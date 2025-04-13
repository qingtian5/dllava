# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN


def llava_image_only_map_fn(example):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            assert DEFAULT_IMAGE_TOKEN in msg['value']
            input += DEFAULT_IMAGE_TOKEN
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def llava_map_fn(example, system=''):

    messages = example['conversations']
    if system != '':
        messages = [{"from": "system", "value": system}] + messages
    input = ''
    conversation = []
    system_dict = {}
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            # if DEFAULT_IMAGE_TOKEN in msg['value']:
            if msg['value'].count(DEFAULT_IMAGE_TOKEN) == 1:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            elif msg['value'].count(DEFAULT_IMAGE_TOKEN) > 1:
                msg['value'] = msg['value'].strip()
            input += msg['value']
        elif msg['from'] == 'system':
            system_dict['system'] = msg['value']
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value'], **system_dict})
            input = ''
        else:
            raise NotImplementedError

    return {'conversation': conversation}
