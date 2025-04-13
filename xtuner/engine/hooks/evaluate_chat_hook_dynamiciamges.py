# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils.misc import get_object_from_string
from mmengine import print_log
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.utils import expand2square, load_image, expand_original_aspect
from xtuner.dataset.llava_multisubimages import image_transform, dynamic_preprocess, slidingwindow_preprocess
from xtuner.model.utils import prepare_inputs_labels_for_multimodal, prepare_multi_subimages_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)


class DynamicImagesEvaluateChatHook(Hook):

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 dynamic_image_processor=None,
                 square_image_processor=None,
                 dino_image_processor=None,
                 convnext_image_processor=None,
                 max_num=0,
                 min_num=0,
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
                # system = prompt_template.SYSTEM.format(system='你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。')
                system = prompt_template.SYSTEM.format(system='你是由理想汽车智能空间创造的多模态AI助手，名叫理想同学，拥有处理和分析图像的能力。\n你的主要任务是根据用户提供的信息提供准确、有用的回复。')
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

        if dynamic_image_processor is not None:
            self.dynamic_image_processor = BUILDER.build(dynamic_image_processor)
        else:
            self.dynamic_image_processor = None 

        if square_image_processor is not None:
            self.square_image_processor = BUILDER.build(square_image_processor)
        else: 
            self.square_image_processor = None
    
        # if dino_image_processor is not None:
        #     self.dino_image_processor = BUILDER.build(dino_image_processor)
        # else:
        #     self.dino_image_processor = None
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

        self.max_num = max_num
        self.min_num = min_num

        self.is_first_run = True

    def _generate_samples(self, runner, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        if self.is_first_run:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            model.to(device)
            self.is_first_run = False

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        # model.llm.config.use_cache = False
        model.llm.config.use_cache = True
        model.eval()

        if self.evaluation_images is not None:
            for sample_image, sample_input in zip(self.evaluation_images,
                                                  self.evaluation_inputs):
                
                torch.cuda.empty_cache()
                
                if self.dynamic_image_processor is not None:

                    try:
                        input_size = self.dynamic_image_processor.crop_size['height']
                    except Exception as e:
                        input_size = self.dynamic_image_processor.crop_size

                    # input_size = 640

                    # width, height, _ = sample_image.shape
                    # if min(width, height) > 448:
                    #     input_size = math.ceil(min(width, height) / 2)
                    # image_thumbnail, _ = expand_original_aspect(sample_image, image_size=input_size, background_color=tuple(int(x * 255) for x in self.dynamic_image_processor.image_mean))
                    
                    image_thumbnail = self.dynamic_image_processor.preprocess(sample_image, return_tensors='pt')['pixel_values']
                    num_pixel_values = 1
                    image_size = [image_thumbnail.shape[2], image_thumbnail.shape[3]]
                    image_thumbnail = image_thumbnail.to(device)
                    print_log(f"evaluate chat hook image_thumbnail: {image_thumbnail.shape}", 'current')
                    
                    if self.convnext_image_processor is not None:
                        convnext_image_thumbnail = self.convnext_image_processor.preprocess(sample_image, return_tensors='pt')['pixel_values']
                        convnext_image_thumbnail = convnext_image_thumbnail.to(device)
                        print_log(f"evaluate chat hook convnext_image_thumbnail: {convnext_image_thumbnail.shape}", 'current')
                    else:
                        convnext_image_thumbnail = None
                else:
                    num_pixel_values = 0
                    image_thumbnail = None
                    convnext_image_thumbnail = None

                if self.square_image_processor is not None:
                    
                    if self.convnext_image_processor is None:
                        try:
                            input_size = self.square_image_processor.crop_size['height']
                        except Exception as e:
                            input_size = self.square_image_processor.crop_size
                    else:
                        try:
                            input_size = self.convnext_image_processor.crop_size['height']
                        except Exception as e:
                            input_size = self.convnext_image_processor.crop_size

                    # input_size = 640

                    sw_image_tiles = slidingwindow_preprocess(sample_image, image_size=input_size, background_color=tuple(int(x * 255) for x in self.square_image_processor.image_mean))
                    minimonkey_image_tiles = dynamic_preprocess(sample_image, min_num=self.min_num, max_num=self.max_num, image_size=input_size, use_rounded_down_ratio=True)

                    # if sw_image_tiles is None:
                    #     image_tiles2 = self.square_image_processor.preprocess(minimonkey_image_tiles, return_tensors='pt')['pixel_values']
                    #     num_pixel_values += len(minimonkey_image_tiles)
                    #     image_tiles = image_tiles2
                    #     image_tiles = image_tiles.to(device)
                    #     print_log(f"evaluate chat hook minimokey image_tiles: {image_tiles.shape}", 'current')
                    #     if self.convnext_image_processor is not None:
                    #         convnext_image_tiles2 = self.convnext_image_processor.preprocess(minimonkey_image_tiles, return_tensors='pt')['pixel_values']
                    #         convnext_image_tiles = convnext_image_tiles2
                    #         convnext_image_tiles = convnext_image_tiles.to(device)
                    #         print_log(f"evaluate chat hook minimokey convnext_image_tiles: {convnext_image_tiles.shape}", 'current')
                    #     else:
                    #         convnext_image_tiles = None
                    # else:
                    #     image_tiles1 = self.square_image_processor.preprocess(sw_image_tiles, return_tensors='pt')['pixel_values']
                    #     image_tiles2 = self.square_image_processor.preprocess(minimonkey_image_tiles, return_tensors='pt')['pixel_values']
                    #     num_pixel_values += (len(sw_image_tiles) + len(minimonkey_image_tiles))
                    #     image_tiles = torch.cat([image_tiles1, image_tiles2], dim=0)
                    #     image_tiles = image_tiles.to(device)
                    #     print_log(f"evaluate chat hook sw + minimokey ({len(sw_image_tiles)} + {len(minimonkey_image_tiles)}) image_tiles: {image_tiles.shape}", 'current')
                    #     if self.convnext_image_processor is not None:
                    #         convnext_image_tiles1 = self.convnext_image_processor.preprocess(sw_image_tiles, return_tensors='pt')['pixel_values']
                    #         convnext_image_tiles2 = self.convnext_image_processor.preprocess(minimonkey_image_tiles, return_tensors='pt')['pixel_values']
                    #         convnext_image_tiles = torch.cat([convnext_image_tiles1, convnext_image_tiles2], dim=0)
                    #         print_log(f"evaluate chat hook sw + minimokey ({len(sw_image_tiles)} + {len(minimonkey_image_tiles)}) convnext_image_tiles: {convnext_image_tiles.shape}", 'current')
                    #     else:
                    #         convnext_image_tiles = None

                    image_tiles1 = self.square_image_processor.preprocess(sw_image_tiles, return_tensors='pt')['pixel_values']
                    image_tiles2 = self.square_image_processor.preprocess(minimonkey_image_tiles, return_tensors='pt')['pixel_values']
                    image_tiles = torch.cat([image_tiles1, image_tiles2], dim=0)
                    image_tiles = image_tiles.to(device)
                    num_pixel_values += (len(image_tiles1) + len(image_tiles2))
                    print_log(f"evaluate chat hook sw + minimokey ({len(sw_image_tiles)} + {len(minimonkey_image_tiles)}) image_tiles: {image_tiles.shape}", 'current')
                else:
                    image_tiles = None
                    convnext_image_tiles = None

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
                
                # if model.dino_visual_encoder and image_dino.dtype != model.dino_visual_encoder.dtype:
                #     image_dino = image_dino.to(dtype=model.dino_visual_encoder.dtype)
                
                # if model.convnext_visual_encoder and image_convnext.dtype != model.convnext_visual_encoder.stem[0].weight.dtype:
                #     image_convnext = image_convnext.to(dtype=model.convnext_visual_encoder.stem[0].weight.dtype)

                # if model.convnext_visual_encoder and image_convnext.dtype != model.convnext_visual_encoder.dtype:
                #     image_convnext = image_convnext.to(dtype=model.convnext_visual_encoder.dtype)

                # start_event = torch.cuda.Event(enable_timing=True)
                # end_event = torch.cuda.Event(enable_timing=True)  
                # start_event.record()

                with torch.no_grad():

                    # image_thumbnail = image_thumbnail.to(dtype=model.visual_encoder.dtype)
                    
                    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    #     visual_outputs = model.visual_encoder(image_thumbnail)
                           
                    # if model.siglip_visual_encoder:
                    #     image_tiles = image_tiles.to(dtype=model.siglip_visual_encoder.dtype)

                    #     siglip_visual_outputs = model.siglip_visual_encoder(
                    #         image_tiles, output_hidden_states=True)

                    # visual_feat_list = []
                    # vit_pixel_values = visual_outputs[-1].to(model.projector.dtype)

                    # # visual_feat_list.append(vit_pixel_values)
                    # visual_feat_list.append((vit_pixel_values, [wh // model.visual_encoder.patch_size for wh in image_size]))

                    # if model.siglip_visual_encoder:
                    #     siglip_pixel_values = siglip_visual_outputs.hidden_states[model.visual_select_layer].to(model.projector.dtype)
                    #     visual_feat_list.append(siglip_pixel_values)

                    if (image_thumbnail is not None) and (image_tiles is not None):

                        bs = image_thumbnail.shape[0]

                        if (convnext_image_thumbnail is not None) and (convnext_image_tiles is not None):
                            pixel_values = model._forward_visual_encoders(
                                torch.cat([image_thumbnail, image_tiles], dim=0), 
                                torch.cat([convnext_image_thumbnail, convnext_image_tiles], dim=0), 
                                model.fusion_channel)
                        else:
                            pixel_values = model._forward_visual_encoders(torch.cat([image_thumbnail, image_tiles], dim=0), model.fusion_channel)

                        if model.projector_type == 'dualpath' or model.projector_type == 'fusion':
                            thumbnail_pixel_values = pixel_values[:bs, ...]
                            tiles_pixel_values = pixel_values[bs:, ...]
                            # print(f"thumbnail_pixel_values: {thumbnail_pixel_values.shape}")
                            # print(f"tiles_pixel_values: {tiles_pixel_values.shape}")

                            visual_feat_list = [thumbnail_pixel_values, tiles_pixel_values]
                        else:
                            visual_feat_list = [pixel_values]
                    else:
                        if image_thumbnail is not None:
                            image = image_thumbnail
                        else:
                            image = image_tiles

                        bs = image.shape[0] // num_pixel_values

                        image_size = (image.shape[2], image.shape[3])

                        pixel_values = model._forward_visual_encoders(image)
                        visual_feat_list = [pixel_values]

                    if len(visual_feat_list) > 1:                 
                        pixel_values = model.projector(visual_feat_list)
                    else:
                        # pixel_values = model.projector((visual_feat_list[0], (x // model.visual_encoder.patch_size for x in image_size)))
                        pixel_values = model.projector(visual_feat_list[0])
                

                    if model.num_sub_images> 1 and (not model.vote_and_mix):

                        # if model.projector_type != "dualpath":
                        #     thumbnail_pixel_values = pixel_values[:bs, ...]
                        #     tiles_pixel_values = pixel_values[bs:, ...].view(bs, -1, pixel_values.shape[-1])
                        #     pixel_values = torch.cat([thumbnail_pixel_values, tiles_pixel_values], dim=1)

                        if (image_thumbnail is None) or (image_tiles is None):
                            pixel_values = pixel_values.view(bs, -1, pixel_values.shape[2])

                        print_log(f"pixel_values to prepare_multi_subimages_inputs: {pixel_values.shape}", 'current')

                        # projector_num_queries = [visual_feat_list[0][0].shape[1] // 4, visual_feat_list[1].shape[1] // 4]
                        # print(f"projector_num_queries: {projector_num_queries}")

                        projector_num_queries = model.projector_num_queries
                        add_img_slice_special_tokens = model.add_img_slice_special_tokens

                        # print(f"projector_num_queries: {projector_num_queries}")
                        # print(f"add_img_slice_special_tokens: {add_img_slice_special_tokens}")

                        mm_inputs = prepare_multi_subimages_inputs_labels_for_multimodal(
                            llm=model.llm,
                            llm_name=model.llm_name,
                            input_ids=input_ids.unsqueeze(0),
                            pixel_values=pixel_values, 
                            num_pixel_values=[num_pixel_values], 
                            projector_num_queries=projector_num_queries,
                            add_img_slice_special_tokens=add_img_slice_special_tokens,
                            mark_slice_indices=model.mark_slice_indices)
                    else:
                        print_log(f"pixel_values to prepare_inputs: {pixel_values.shape}", 'current')
                        mm_inputs = prepare_inputs_labels_for_multimodal(
                            llm=model.llm,
                            input_ids=input_ids.unsqueeze(0),
                            pixel_values=pixel_values)

                    print_log(f"mm_inputs['inputs_embeds']: {mm_inputs['inputs_embeds'].shape}", 'current')

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
