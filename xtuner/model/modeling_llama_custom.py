import math
import warnings
from typing import List, Optional, Tuple, Union

from xtuner.utils import IGNORE_INDEX
from mmengine import print_log

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig
from .modeling_llama_pdrop import LlamaPreTrainedModel, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

_CONFIG_FOR_DOC = "LlamaConfig"

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"


        self.interaction_layer_indices = [15, 30, 45]
        self.r = 0.5

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.loop = 1

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        slice_indices: Optional[List] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if (layer_idx + 1) in self.interaction_layer_indices:

                if hidden_states.shape[1] != 1:

                    (
                        hidden_states, 
                        attention_mask, 
                        labels, 
                        slice_indices,
                        position_ids
                    ) = self.prepra_input_labels(
                        hidden_states, 
                        attention_mask, 
                        labels, slice_indices, 
                        layer_idx,
                        position_ids
                    )
                    
                    if self._use_flash_attention_2:
                        # 2d mask is passed through the layers
                        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                    elif self._use_sdpa and not output_attentions:
                        # output_attentions=True can not be supported when using SDPA, and we fall back on
                        # the manual implementation that requires a 4D causal mask in all cases.
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                            attention_mask,
                            (batch_size, seq_length),
                            inputs_embeds,
                            past_key_values_length,
                        )
                    else:
                        # 4d mask is passed through the layers
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                        )
         
                else:
                    # print(f"{'=' * 25} layer_idx: {layer_idx} {'=' * 25}")
                    new_position_ids = []
                    # print(f"position_ids: {position_ids}")
                    # print(f"slice_indices: {slice_indices}")
                    for idx, cur_position_ids in enumerate(position_ids):
                        num_slice_token = slice_indices[0][0][1] - slice_indices[0][0][0] - 1
                        factor = self.interaction_layer_indices.index((layer_idx + 1)) + 1
                        cur_visual_length = int(num_slice_token * (0.5 ** (factor - 1)))
                        # print(f"cur_visual_length: {cur_visual_length}")
                        next_visual_length = int(num_slice_token * (0.5 ** factor))
                        # print(f"next_visual_length: {next_visual_length}")
                        cur_position_ids = cur_position_ids - (cur_visual_length - next_visual_length)
                        # print(f"cur_position_ids: {cur_position_ids}")
                        new_position_ids.append(cur_position_ids)
                    position_ids = new_position_ids
                    # print(f"position_ids: {position_ids}")

        hidden_states = self.norm(hidden_states)

        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"labels: {labels.shape}")

        self.loop += 1

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), labels

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device

        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def vote_and_mix(self, x, r=0.2):
        # print(f"x: {x.shape}")
        bs, n, _ = x.shape

        x_norm = x / x.norm(dim=-1, keepdim=True)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

        for i in range(sim.size(0)):
            sim[i].fill_diagonal_(float('-inf'))

        v_w, v_i = sim.max(1)

        score = torch.zeros((bs, n), device=x.device, dtype=x.dtype).scatter_add_(-1, v_i, v_w)
        r_id = score.argsort(-1)[:, :int(n * (1 - r))]
        p_id = score.argsort(-1)[:,int(n * (1 - r)):]

        w = []

        for i in range(bs):
            w.append(torch.index_select(torch.index_select(sim[i], dim=0, index=p_id[i]), dim=1, index=r_id[i]))

        w = torch.stack(w).to(x.device)
        w = F.softmax(w, dim=-1)

        x_p, x_r = [], []

        for i in range(bs):
            x_p.append(torch.index_select(x[i], dim=0, index=p_id[i]))
            x_r.append(torch.index_select(x[i], dim=0, index=r_id[i]))
        x_p = torch.stack(x_p).to(x.device)
        x_r = torch.stack(x_r).to(x.device)

        x_mix = torch.bmm(w.transpose(1, 2), x_p)

        x_out = x_r + x_mix

        return x_out

    def prepra_input_labels(
        self, 
        inputs_embeds,
        attention_mask,
        labels,
        slice_indices,
        layer_idx,
        position_ids
    ):
        
        new_inputs_embeds = []
        new_attention_mask = []
        new_labels = []
        new_slice_indices = []
        batch_size = inputs_embeds.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size,inputs_embeds.shape[1]), dtype=torch.bool, device=inputs_embeds.device)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full((batch_size,inputs_embeds.shape[1]), IGNORE_INDEX, device=inputs_embeds.device)            

        for batch_idx in range(batch_size):
            # print(f"=====> layer {layer_idx}, batch_idx: {batch_idx} <=====")

            cur_input_embeds = inputs_embeds[batch_idx]
            cur_attention_mask = attention_mask[batch_idx]                
            slice_start, slice_end = slice_indices[batch_idx][0]

            # print(f"cur_input_embeds: {cur_input_embeds.shape}")
            # print(f"cur_attention_mask: {cur_attention_mask.shape}")
            # print(f"slice_start, slice_end: {slice_start, slice_end}")

            if slice_start == -1 and slice_end == -1:
                new_inputs_embeds.append(cur_input_embeds)
                new_attention_mask.append(cur_attention_mask)
                cur_label = labels[batch_idx]
                new_labels.append(cur_label)
                new_slice_indices.append([[slice_start, slice_end]])
                continue

            front_input_embeds = cur_input_embeds[:slice_start + 1]
            tail_input_embeds = cur_input_embeds[slice_end:]
            slice_input_embeds = cur_input_embeds[slice_start + 1:slice_end]

            # print(f"front_input_embeds: {front_input_embeds.shape}")
            # print(f"tail_input_embeds: {tail_input_embeds.shape}")
            # print(f"slice_input_embeds: {slice_input_embeds.shape}")

            front_attention_mask = cur_attention_mask[:slice_start + 1]
            tail_cur_attention_mask = cur_attention_mask[slice_end:]
            slice_cur_attention_mask = cur_attention_mask[slice_start + 1:slice_end]

            # print(f"front_attention_mask: {front_attention_mask.shape}")
            # print(f"tail_cur_attention_mask: {tail_cur_attention_mask.shape}")
            # print(f"slice_cur_attention_mask: {slice_cur_attention_mask.shape}")

            cur_label = labels[batch_idx]
            front_label = cur_label[:slice_start + 1]
            tail_label = cur_label[slice_end:]
            slice_label = cur_label[slice_start + 1:slice_end]

            # print(f"front_label: {front_label.shape}")
            # print(f"tail_label: {tail_label.shape}")
            # print(f"slice_label: {slice_label.shape}")

            preserved_slice_input_embeds = self.vote_and_mix(slice_input_embeds.unsqueeze(0), self.r).squeeze()
            preserved_slice_token_num = preserved_slice_input_embeds.shape[0]

            # print(f"preserved_slice_input_embeds: {preserved_slice_input_embeds.shape}")
            # print(f"preserved_slice_token_num: {preserved_slice_token_num}")

            # if not self.training and self.loop == 1:
            #     print(f"layer {layer_idx} preserved_slice_token_num: {preserved_slice_token_num}")

            new_slice_end = slice_start + 1 + preserved_slice_token_num
            new_slice_indices.append([[slice_start, new_slice_end]])
            # print(f"new_slice_indices: {new_slice_indices}")

            cur_new_input_embeds = torch.cat([
                front_input_embeds, preserved_slice_input_embeds, tail_input_embeds], dim=0)
            cur_new_attention_mask = torch.cat([
                front_attention_mask, slice_cur_attention_mask[:preserved_slice_token_num], tail_cur_attention_mask], dim=0)
            cur_new_labels = torch.cat([
                front_label, slice_label[:preserved_slice_token_num], tail_label], dim=0)

            new_inputs_embeds.append(cur_new_input_embeds)
            new_attention_mask.append(cur_new_attention_mask)
            new_labels.append(cur_new_labels)

            # print(f"cur_new_input_embeds: {cur_new_input_embeds.shape}")
            # print(f"cur_new_attention_mask: {cur_new_attention_mask.shape}")
            # print(f"cur_new_labels: {cur_new_labels.shape}")

        max_len = max(x.shape[0] for x in new_inputs_embeds)
        new_inputs_embeds_padded = []
        new_attention_mask_paded = []
        new_labels_padded = torch.full((batch_size, max_len),
                                IGNORE_INDEX,
                                dtype=new_labels[0].dtype,
                                device=new_labels[0].device)
        new_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for batch_idx in range(batch_size):
            cur_new_embed = new_inputs_embeds[batch_idx]
            cur_attention_mask = new_attention_mask[batch_idx]

            cur_len = cur_new_embed.shape[0]
            new_inputs_embeds_padded.append(
                torch.cat((cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                    dtype=cur_new_embed.dtype,
                                    device=cur_new_embed.device)),
                        dim=0))
            
            cur_len = cur_attention_mask.shape[0]
            new_attention_mask_paded.append(
                torch.cat((cur_attention_mask,
                        torch.zeros((max_len - cur_len),
                                    dtype=cur_attention_mask.dtype,
                                    device=cur_attention_mask.device)),
                        dim=0))

            cur_new_label = new_labels[batch_idx]
            cur_len = cur_new_label.shape[0]
            new_labels_padded[batch_idx, :cur_len] = cur_new_label

            cur_len = cur_attention_mask.sum().item()
            new_position_ids[batch_idx, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_inputs_embeds_padded = torch.stack(new_inputs_embeds_padded, dim=0)
        new_attention_mask_paded = torch.stack(new_attention_mask_paded, dim=0)

        # print(f"new_inputs_embeds_padded: {new_inputs_embeds_padded.shape}")
        # print(f"new_attention_mask_paded: {new_attention_mask_paded.shape}")
        # print(f"new_slice_indices: {new_slice_indices}")
        # print(f"new_position_ids: {new_position_ids.shape}")

        return new_inputs_embeds_padded, new_attention_mask_paded, new_labels_padded, new_slice_indices, new_position_ids

class LlamaForCausalLM_VoteMix(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        slice_indices: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(f"input_ids: {input_ids.shape if input_ids is not None else 'None'}")
        # print(f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
        # print(f"position_ids: {position_ids.shape if position_ids is not None else 'None'}")
        # print(f"past_key_values: {len(past_key_values) if past_key_values is not None else 'None'}")
        # print(f"inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
        # print(f"use_cache: {use_cache}")
        # print(f"output_attentions: {output_attentions}")
        # print(f"output_hidden_states: {output_hidden_states}")
        # print(f"return_dict: {return_dict}")
        # print(f"cache_position: {cache_position.shape if cache_position is not None else 'None'}")
        # print(f"LlamaForCausalLM slice_indices: {slice_indices}")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, labels = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            slice_indices=slice_indices,
            labels=labels
        )

        hidden_states = outputs[0]
        # print(f"hidden_states: {hidden_states.shape}")

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # print(f"labels: {labels.shape}")

            # labels = self.update_labels(labels, slice_indices)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, slice_indices=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if slice_indices is not None:
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "slice_indices": slice_indices
                }
            )    
        else:
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

