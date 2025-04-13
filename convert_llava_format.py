import json
import random
import os
from tqdm import tqdm

prompts_en = ['Write a terse but informative summary of the picture.\n<image>',
 '<image>\nRender a clear and concise summary of the photo.',
 '<image>\nShare a concise interpretation of the image provided.',
 'What is in the photo?\n<image>',
 '<image>\nGive a short and clear explanation of the subsequent image.',
 '<image>\nDescribe the image concisely.',
 '<image>\nWrite a terse but informative summary of the picture.',
 'Provide a brief description of the given image.\n<image>',
 'Render a clear and concise summary of the photo.\n<image>',
 'Summarize the visual content of the image.\n<image>',
 'Share a concise interpretation of the image provided.\n<image>',
 '<image>\nWhat is in the photo?',
 '<image>\nProvide a brief description of the given image.',
 '<image>\nGive a brief description of the image.',
 'Give a brief description of the image.\n<image>',
 'What is this?\n<image>',
 '<image>\nWhat is this?',
 'Give a short and clear explanation of the subsequent image.\n<image>',
 "Present a compact description of the photo's key features.\n<image>",
 '<image>\nSummarize the visual content of the image.',
 'Describe the image concisely.\n<image>',
 "<image>\nPresent a compact description of the photo's key features."]

prompts_cn = [
    '对图片进行简洁但信息丰富的总结。\n<image>',
    '<image>\n对照片进行清晰简洁的总结。',
    '<image>\n分享对提供的图片的简洁解释。',
    '照片中有什么？\n<image>',
    '<image>\n对随后的图片给出简短而清晰的解释。',
    '<image>\n简洁地描述这张图片。',
    '<image>\n对图片进行简洁但信息丰富的总结。',
    '对给定的图片提供简短的描述。\n<image>',
    '对照片进行清晰简洁的总结。\n<image>',
    '对图片的视觉内容进行总结。\n<image>',
    '分享对提供的图片的简洁解释。\n<image>',
    '<image>\n照片中有什么？',
    '<image>\n对给定的图片提供简短的描述。',
    '<image>\n对图片进行简短的描述。',
    '对图片进行简短的描述。\n<image>',
    '这是什么？\n<image>',
    '<image>\n这是什么？',
    '对随后的图片给出简短而清晰的解释。\n<image>',
    "对照片的关键特点进行简洁的描述。\n<image>",
    '<image>\n总结图片的视觉内容。',
    '简洁地描述这张图片。\n<image>',
    "<image>\n对照片的关键特点进行简洁的描述。"
]

file = '/mnt/pfs-guan-ssai/cv/yanghongfu/grit_coyo_20m/grit_coyo_20m_all.json'
output_file = 'data/llava_data/LLaVA-Pretrain/grit_coyo_20m_all.json'

file = 'data/llava_data/LLaVA-Pretrain/generated_en_27m_catcaption.json'
output_file = 'data/llava_data/LLaVA-Pretrain/generated_en_27m_catcaption.json'


print(f"read from {file}")

injson = json.load(open(file))
#
#print(injson[0])
#quit()

tmp = []

for i in tqdm(injson):
    new_dict = {
        'image': i['image'],
        'conversations': [{'from': 'human', 'value': random.choice(prompts)}, {'from': 'gpt','value': i['caption']}]
    }
    tmp.append(new_dict)

print(len(tmp))
print(tmp[0])

json.dump(tmp, open(output_file, 'w'))