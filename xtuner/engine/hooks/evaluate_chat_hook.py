# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils.misc import get_object_from_string
from mmengine import print_log
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.utils import expand2square, load_image
from xtuner.dataset.llava_multisubimages import image_transform, dynamic_preprocess
from xtuner.model.utils import prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)


class EvaluateChatHook(Hook):

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 image_processor=None,
                 dino_image_processor=None,
                 convnext_image_processor=None,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=256,
                 stop_word=None,
                 stop_words=[]):
        self.evaluation_inputs = evaluation_inputs
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]
        self.evaluation_images = evaluation_images
        if isinstance(self.evaluation_images, str):
            self.evaluation_images = [self.evaluation_images]
        if self.evaluation_images is not None:
            assert len(
                self.evaluation_images) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_images) == 1:
                self.evaluation_images = [self.evaluation_images[0]] * len(
                    self.evaluation_inputs)
            self.evaluation_images = [
                load_image(img) for img in self.evaluation_images
            ]
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
            if prompt_template.get('TYPE', '') == 'mindgpt':
                # system = prompt_template.SYSTEM.format(system='你是一个名字叫做理想同学的AI数字生命体。')
                system = prompt_template.SYSTEM.format(system='你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。')
            stop_words += prompt_template.get('STOP_WORDS', [])
        if stop_word is not None:
            # TODO: deprecation, v0.3.0
            warnings.warn(
                ('The `stop_word` argument is deprecated and will be removed '
                 'in v0.3.0, use `stop_words` instead.'), DeprecationWarning)
            stop_words.append(stop_word)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        if dino_image_processor is not None:
            self.dino_image_processor = BUILDER.build(dino_image_processor)
        else:
            self.dino_image_processor = None
        if convnext_image_processor is not None:
            self.convnext_image_processor = BUILDER.build(convnext_image_processor)
        else:
            self.convnext_image_processor = None

        self.stop_criteria = StoppingCriteriaList()
        # default generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

    def _generate_samples(self, runner, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = False
        model.eval()

        if self.evaluation_images is not None:
            for sample_image, sample_input in zip(self.evaluation_images,
                                                  self.evaluation_inputs):
                
                torch.cuda.empty_cache()
                
                if model.num_sub_images > 1:
                    try:
                        input_size = self.image_processor.crop_size['height']
                    except Exception as e:
                        input_size = self.image_processor.crop_size
                
                    images = dynamic_preprocess(sample_image, image_size=input_size, use_thumbnail=True, max_num=model.num_sub_images, 
                                                thumbnail_first=True
                                                )
                    num_pixel_values = len(images)

                    image = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']

                    num_padding = model.num_sub_images + 1 - num_pixel_values
                    if num_padding > 0:
                        image = torch.cat([image, torch.zeros(num_padding, 3, input_size, input_size)], dim=0)

                    image = image.to(device)

                    image_dino = None
                    if self.dino_image_processor:
                        print_log(f"image_dino obtained from dino_image_processor", 'current')
                        image_dino = self.dino_image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                        image_dino = image_dino.to(device)
                    else:
                        print_log(f"image_dino obtained from image", 'current')
                        image_dino = image

                    image_convnext = None
                    if self.convnext_image_processor:
                        print_log(f"image_convnext obtained from convnext_image_processor", 'current')
                        image_convnext = self.convnext_image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                        image_convnext = image_convnext.to(device)
                        

                else:
                    if self.image_processor.image_mean is not None:
                        square_image = expand2square(
                            sample_image,
                            tuple(
                                int(x * 255) for x in self.image_processor.image_mean))
                    else:
                        square_image = sample_image
                    image = self.image_processor.preprocess(
                        square_image, return_tensors='pt')['pixel_values'][0]
                    image = image.to(device)

                    image_dino = None
                    if self.dino_image_processor:
                        print_log(f"image_dino obtained from dino_image_processor", 'current')
                        image_dino = self.dino_image_processor.preprocess(
                            square_image, return_tensors='pt')['pixel_values'][0]
                        image_dino = image_dino.to(device)
                    else:
                        print_log(f"image_dino obtained from image", 'current')
                        image_dino = image

                    image_convnext = None
                    if self.convnext_image_processor:
                        print_log(f"image_convnext obtained from convnext_image_processor", 'current')
                        image_convnext = self.convnext_image_processor.preprocess(square_image, return_tensors='pt')['pixel_values'][0]
                        image_convnext = image_convnext.to(device)

                sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
                # print(f"====> sample_input: {[sample_input]}")
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                # print(f"self.system: {self.system}, self.instruction: {[self.instruction]}, inputs: {[inputs]}")
                chunk_encode = []
                for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                    # print(f"====> chunk: {chunk}")
                    if idx == 0:
                        cur_encode = self.tokenizer.encode(chunk)
                    else:
                        cur_encode = self.tokenizer.encode(
                            chunk, add_special_tokens=False)
                    chunk_encode.append(cur_encode)
                assert len(chunk_encode) == 2
                # print(f"====> chunk_encode: {chunk_encode}")
                input_ids = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    input_ids.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        input_ids.append(IMAGE_TOKEN_INDEX)
                input_ids = torch.tensor(input_ids).to(device)
                # print(f"====> input_ids: {input_ids}, {input_ids.shape}")

                print_log(f"evaluate chat hook image: {image.shape}", 'current')

                if image_dino is not None:
                    print_log(f"evaluate chat hook image_dino: {image_dino.shape}", 'current')
                    print_log(f"siglipdino share one processor: {(image - image_dino).mean() == 0}", 'current')
                if image_convnext is not None:
                    print_log(f"evaluate chat hook image_convnext: {image_convnext.shape}", 'current')

                if image.dtype != model.visual_encoder.dtype:
                    image = image.to(dtype=model.visual_encoder.dtype)

                if model.dino_visual_encoder and image_dino.dtype != model.dino_visual_encoder.dtype:
                    image_dino = image_dino.to(dtype=model.dino_visual_encoder.dtype)
                
                # if model.convnext_visual_encoder and image_convnext.dtype != model.convnext_visual_encoder.stem[0].weight.dtype:
                #     image_convnext = image_convnext.to(dtype=model.convnext_visual_encoder.stem[0].weight.dtype)

                if model.convnext_visual_encoder and image_convnext.dtype != model.convnext_visual_encoder.dtype:
                    image_convnext = image_convnext.to(dtype=model.convnext_visual_encoder.dtype)

                # start_event = torch.cuda.Event(enable_timing=True)
                # end_event = torch.cuda.Event(enable_timing=True)  
                # start_event.record()

                with torch.no_grad():

                    if model.visual_encoder_name == "RADIOModel":
                        if len(image.shape) == 3:
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                visual_outputs = model.visual_encoder(image.unsqueeze(0))
                        else:
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                visual_outputs = model.visual_encoder(image)
                    else:
                        if len(image.shape) == 3:
                            visual_outputs = model.visual_encoder(
                                image.unsqueeze(0), output_hidden_states=True)
                        else:
                            visual_outputs = model.visual_encoder(
                                image, output_hidden_states=True)

                    # end_event.record()
                    # torch.cuda.synchronize()

                    # forward_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
                    # print(f"visual_encoder forward pass time (CUDA): {forward_time:.6f} seconds")

                    # start_event = torch.cuda.Event(enable_timing=True)
                    # end_event = torch.cuda.Event(enable_timing=True)
                    # start_event.record()

                    if model.dino_visual_encoder:
                        if len(image_dino.shape) == 3:
                            dino_outputs = model.dino_visual_encoder(
                                image_dino.unsqueeze(0), output_hidden_states=True)
                        else:
                            dino_outputs = model.dino_visual_encoder(
                                image_dino, output_hidden_states=True)
                            

                    if model.convnext_visual_encoder:
                        if len(image.shape) == 3:
                            # convnext_outputs = model.convnext_visual_encoder.stem(image_convnext.unsqueeze(0))
                            # for stage in model.convnext_visual_encoder.stages:
                            #     convnext_outputs = stage(convnext_outputs)
                            convnext_outputs = model.convnext_visual_encoder(
                                image_convnext.unsqueeze(0), output_hidden_states=True).last_hidden_state                  
                        else:
                            # convnext_outputs = model.convnext_visual_encoder.stem(image_convnext)
                            # for stage in model.convnext_visual_encoder.stages:
                            #     convnext_outputs = stage(convnext_outputs)
                            convnext_outputs = model.convnext_visual_encoder(
                                image_convnext, output_hidden_states=True).last_hidden_state

                    visual_feat_list = []
                    if model.visual_encoder_name == "RADIOModel":
                        vit_pixel_values = visual_outputs[-1].to(model.projector.dtype)
                        visual_feat_list.append(vit_pixel_values)
                    else:
                        vit_pixel_values = visual_outputs.hidden_states[model.visual_select_layer]
                        visual_feat_list.append(vit_pixel_values)

                    if model.dino_visual_encoder:
                        dino_pixel_values = dino_outputs.hidden_states[model.visual_select_layer][:, 1:]
                        visual_feat_list.append(dino_pixel_values)

                    if model.convnext_visual_encoder:
                        bs = vit_pixel_values.shape[0]
                        convnext_pixel_values = convnext_outputs.view(bs, convnext_outputs.shape[1], -1).permute(0, 2, 1)
                        visual_feat_list.append(convnext_pixel_values)

                if len(visual_feat_list) > 1:
                    pixel_values = model.projector(torch.cat(visual_feat_list, dim=-1))
                else:
                    pixel_values = model.projector(visual_feat_list[0])

                # end_event.record()
                # torch.cuda.synchronize()

                # forward_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
                # print(f"dino_visual_encoder forward pass time (CUDA): {forward_time:.6f} seconds")

                # if model.dino_visual_encoder and model.convnext_visual_encoder is None:

                #     if model.projector.__class__.__name__ == 'TokenPackerProjectorModel':

                #         # vit_pixel_values = model.visual_feature_select(visual_outputs, layers=[12,16,22,23], with_cls=False)
                #         # dino_pixel_values = mdoel.visual_feature_select(dino_outputs, layers=[12,16,22,23], with_cls=True)

                #         vit_pixel_values = model.visual_feature_select(visual_outputs, with_cls=False)
                #         dino_pixel_values = model.visual_feature_select(dino_outputs, with_cls=True)

                #         pixel_values = (
                #             torch.cat([vit_pixel_values[0], dino_pixel_values[0]], dim=-1),
                #             torch.cat([vit_pixel_values[1], dino_pixel_values[1]], dim=-1)
                #         )
                        
                #         pixel_values = model.projector(pixel_values)
                #     else:
                #         vit_pixel_values = visual_outputs.hidden_states[model.visual_select_layer]
                #         dino_pixel_values = dino_outputs.hidden_states[model.visual_select_layer][:, 1:]
                #         pixel_values = model.projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))

                #     # vit_pixel_values = visual_outputs.hidden_states[model.visual_select_layer]
                #     # dino_pixel_values = dino_outputs.hidden_states[model.visual_select_layer][:, 1:]
                                        
                #     # if model.projector.__class__.__name__ == 'FusionProjectorModel':
                #     #     pixel_values = model.projector((torch.cat([vit_pixel_values, dino_pixel_values], dim=-1), [num_pixel_values - 1]))
                #     # else:
                #     #     pixel_values = model.projector(torch.cat([vit_pixel_values, dino_pixel_values], dim=-1))

                # elif model.dino_visual_encoder and model.convnext_visual_encoder:

                #     vit_pixel_values = visual_outputs.hidden_states[model.visual_select_layer]
                #     dino_pixel_values = dino_outputs.hidden_states[model.visual_select_layer][:, 1:]

                #     bs = vit_pixel_values.shape[0]
                #     convnext_pixel_values = convnext_outputs.view(bs, convnext_outputs.shape[1], -1).permute(0, 2, 1)

                #     if model.projector.__class__.__name__ == 'FusionProjectorModel':
                #         pixel_values = model.projector((torch.cat([vit_pixel_values, dino_pixel_values, convnext_pixel_values], dim=-1), [num_pixel_values - 1]))
                #     else:
                #         pixel_values = model.projector(torch.cat([vit_pixel_values, dino_pixel_values, convnext_pixel_values], dim=-1))

                # else: 
                #     pixel_values = model.projector(
                #         visual_outputs.hidden_states[model.visual_select_layer][:, 1:])   
                    
                

                if model.num_sub_images> 1:
                    if model.projector_type != "cascadedhoneybee":
                        pixel_values = pixel_values.view(-1, num_pixel_values * pixel_values.shape[1], pixel_values.shape[2])

                    print_log(f"evaluate chat hook pixel_values: {pixel_values.shape}", 'current')

                    mm_inputs = prepare_multi_subimages_inputs_labels_for_multimodal(
                        llm=model.llm,
                        input_ids=input_ids.unsqueeze(0),
                        pixel_values=pixel_values, 
                        num_pixel_values=[num_pixel_values], 
                        projector_num_queries=model.projector_num_queries)
                else:
                    print_log(f"evaluate chat hook pixel_values: {pixel_values.shape}", 'current')
                    
                    mm_inputs = prepare_inputs_labels_for_multimodal(
                        llm=model.llm,
                        input_ids=input_ids.unsqueeze(0),
                        pixel_values=pixel_values)

                # print(f"mm_inputs: {mm_inputs.keys()}")
                # for k, v in mm_inputs.items():
                #     if v is not None:
                #         print(k, v.shape)
                #     else:
                #         print(k, "None")

                generation_output = model.generate(
                    **mm_inputs,
                    max_new_tokens=max_new_tokens,
                    generation_config=self.gen_config,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=self.stop_criteria)
                runner.logger.info(
                    f'\n{"="*50}\nSample output:\n'
                    f'{inputs + self.tokenizer.decode(generation_output[0])}\n{"="*50}\n'
            )
        else:
            for sample_input in self.evaluation_inputs:
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
                input_ids = input_ids.to(device)
                generation_output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    generation_config=self.gen_config,
                    stopping_criteria=self.stop_criteria)
                runner.logger.info(
                    f'\n{"="*50}\nSample output:\n'
                    f'{self.tokenizer.decode(generation_output[0])}\n{"="*50}\n')

        # Cast to training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        runner.logger.info('before_train in EvaluateChatHook.')
        self._generate_samples(runner, max_new_tokens=50)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None or (batch_idx +
                                          1) % self.every_n_iters != 0:
            return
        runner.logger.info('after_train_iter in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_train(self, runner):
        runner.logger.info('after_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in EvaluateChatHook.')
        self._generate_samples(runner)
