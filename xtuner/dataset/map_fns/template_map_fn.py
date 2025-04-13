# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from mmengine.utils.misc import get_object_from_string


def template_map_fn(example, template):

    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation.get('input', '')
        if input is None:
            input = ''
        input_text = template.INSTRUCTION.format(input=input, round=i + 1)

        if i == 0 and template.get('TYPE', '') == 'mindgpt':
            # system = template.SYSTEM.format(system='你是一个名字叫做理想同学的AI数字生命体。')
            # system = template.SYSTEM.format(system='你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n\n请根据以下文本写一个合适的回复。')
            system = template.SYSTEM.format(system='你是由理想汽车智能空间创造的多模态AI助手，名叫理想同学，拥有处理和分析图像的能力。\n你的主要任务是根据用户提供的信息提供准确、有用的回复。')
            input_text = system + input_text

        system = single_turn_conversation.get('system', '')
        if system != '' and system is not None:
            system = template.SYSTEM.format(system=system)
            input_text = system + input_text
        single_turn_conversation['input'] = input_text

        if template.get('SUFFIX', None):
            output_text = single_turn_conversation.get('output', '')
            output_text += template.SUFFIX
            single_turn_conversation['output'] = output_text

        # SUFFIX_AS_EOS is False ==> need_eos_token is True
        single_turn_conversation['need_eos_token'] = \
            not template.get('SUFFIX_AS_EOS', False)
        single_turn_conversation['sep'] = template.get('SEP', '')

    return {'conversation': conversation}


def template_map_fn_factory(template):
    if isinstance(template, str):  # for resume
        template = get_object_from_string(template)
    return partial(template_map_fn, template=template)
