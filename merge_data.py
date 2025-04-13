import json 
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('/mnt/pfs-guan-ssai/cv/moxu/MindGPT-V6-16B-stage2-26000/')
current_directory = Path('/mnt/pfs-guan-ssai/cv/moxu/data/100m')
json_files = list(current_directory.glob('*.json'))
print(json_files, len(json_files))
all_data = []
for jf in json_files:
    injson = json.load(open(jf))
    tmp = []
    for i in tqdm(injson, desc=jf.stem):
        if i.get('image', None) is not None:
            i['num_tokens'] = len(tokenizer(i['conversations'][1]['value'])['input_ids'])
            tmp.append(i)
    tmp = [i for i in tmp if i['num_tokens'] <= 100]

    print(f"original: {len(injson)} -----> {len(tmp)}")

    all_data += tmp

    print(f"all_data: {len(all_data)}")

json.dump(all_data, open('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/LLaVA-Pretrain/mix_100m_max_tokens_100.json', 'w'))


