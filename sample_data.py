import os
import json
from pathlib import Path
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
    #   CLIPImageProcessor,
                          CLIPVisionModel,
                          SiglipImageProcessor, SiglipVisionModel,
                          AutoModel, AutoConfig, AutoProcessor)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b',
    trust_remote_code=True,
    padding_side='right')

output_dir = '/mnt/pfs-mc0p4k/cv/team/lishanshan/data/OV_SFT/Car_sft_data/data/train_data/1122_sft_car_260k_short_a_pure_t'

os.makedirs(output_dir, exist_ok=True)

for jf in list(Path('/mnt/pfs-mc0p4k/cv/team/lishanshan/data/OV_SFT/Car_sft_data/data/train_data/1122_sft_car_260k_short_a_pure_t').rglob('*.json')):
    valid_data = []
    num_tokens = []

    injson = json.load(open(jf))
    for i in tqdm(injson):
        texts = ""
        for c in i['conversations']:
            texts += c['value']

        i['num_tokens'] = len(tokenizer(texts)['input_ids'])

        num_tokens.append(i['num_tokens'])

        if i['num_tokens'] > 2048: continue

        valid_data.append(i)

    print(f"File: {jf.stem}, Max tokens: {max(num_tokens)}, cleand data: {len(valid_data)}")

    with open(f"{output_dir}/{jf.name}", 'w', encoding='utf-8') as file:
        json.dump(valid_data, file, ensure_ascii=False, indent=2)