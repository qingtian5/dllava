import os
import glob
import argparse
import json
from tqdm import tqdm
from pathlib import Path
# from transformers import AutoTokenizer
import pandas as pd
import concurrent.futures


def process(input_data, filename):
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path='/mnt/pfs-guan-ssai/cv/yanghongfu/Nous-Hermes-2-Yi-34B',
    #     trust_remote_code=True,
    #     padding_side='right')

    json_data = []
    for idx, i in enumerate(tqdm(input_data, desc=Path(filename).stem)):

        # flag = False
        # for c in i['conversations']:
        #     if c['from'] == 'human' and c['value'].count('<image>') > 1: 
        #         flag = True
        #         break
        # if flag: continue

        texts = ''
        for c in i['conversations']:
            texts += c['value'].replace('<image>', '<unk>')

        num_image_tokens = texts.count('<image>')

        # num_tokens = len(tokenizer(texts)['input_ids'])

        # if num_tokens > (4096 - 256 * 5): continue
        json_data.append(i)

    return json_data


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument("--input_single_dataset_path", type=str, default="/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/LLaVA-Pretrain/mix_60m_tokens_10-200.json", help="input data path")
# parser.add_argument("--output_path", type=str, default="/mnt/pfs-guan-ssai/cv/moxu/data/sharding_data/mix_60m_tokens_10-200", help="output data folder path")

parser.add_argument("--input_single_dataset_path", type=str,
                    default="/mnt/pfs-mc0p4k/cv/team/lishanshan/data/OV_SFT/Car_sft_data/data/train_data/1122_llava_type_car_260k_short_a_pure_t_shuffle.json",
                    help="input data path")
parser.add_argument("--output_path", type=str,
                    default="/mnt/pfs-mc0p4k/cv/team/lishanshan/data/OV_SFT/Car_sft_data/data/train_data/1122_sft_car_260k_short_a_pure_t",
                    help="output data folder path")
parser.add_argument("--split_size", type=int, default=20000)
args = parser.parse_args()

print('Start to split the dataset...')
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

if args.input_single_dataset_path == 'list':
    input_single_dataset_path = [
        '/mnt/pfs-mc0p4k/cv/team/yanghongfu/LLaVA-Instruct/llava_v1_5_mix665k.json',
        '/mnt/pfs-mc0p4k/cv/team/yanghongfu/LLaVA-Instruct/llava_mmdu-45k.json',
        '/mnt/pfs-mc0p4k/cv/team/yanghongfu/LLaVA-Instruct/llava_mantis_instruct_max_5_images.json'
    ]
    for fileid, filename in enumerate(input_single_dataset_path):
        # print(filename)
        # json_data += json.load(open(filename))

        # if 'llava_cn_50m_from_wukong_cogvlm2' not in filename: continue

        json_data = []

        output_file = f"{args.output_path}/list_{fileid}_"
        print(f"output_file: {output_file}")

        injson = json.load(open(filename))
        num_records = len(injson)

        # injson = [injson[i::64] for i in range(64)]
        # # 创建一个线程池，最大线程数为 4
        # with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        #     # 提交任务并获取 Future 对象
        #     futures = [pool.submit(process, records, filename) for records in injson]

        #     # 迭代 Future 对象，获取结果
        #     for future in concurrent.futures.as_completed(futures):
        #         try:
        #             result = future.result()  # 获取函数执行结果，如果执行出错，会抛出异常
        #         except Exception as e:
        #             print(f"{future} generated an exception: {e}")
        #         else:
        #             # print(result)  # 处理函数执行结果
        #             json_data += result

        for idx, i in enumerate(tqdm(injson)):
            i['id'] = f"list_{fileid}_{str(idx).zfill(8)}"
            if 'image' in i and isinstance(i['image'], str): i['image'] = [i['image']]

            flag = False
            texts = ''
            for c in i['conversations']:
                if c['from'] == 'gpt' and '<image>' in c['value']: flag = True
                texts += c['value']

            if flag:
                continue

            num_image_tokens = texts.count('<image>')
            num_images = len(i['image']) if 'image' in i else 0

            if num_image_tokens != num_images: continue

            json_data.append(i)

        print(f"{'-' * 100}\nrecords: {num_records} -> {len(json_data)}\n{'-' * 100}")

        for i in tqdm(range(0, len(json_data), args.split_size), desc=Path(filename).stem):
            with open(output_file + str(i // args.split_size) + '.json', "w") as f:
                json.dump(json_data[i:i + args.split_size], f)

        # print('Done')

else:
    print(args.input_single_dataset_path)
    json_data = json.load(open(args.input_single_dataset_path))

    print(f"records: {len(json_data)}")

    for i in tqdm(range(0, len(json_data), args.split_size)):
        with open(args.output_path + '/' + os.path.splitext(os.path.basename(args.input_single_dataset_path))[
            0] + '_' + str(i // args.split_size) + '.json', "w") as f:
            json.dump(json_data[i:i + args.split_size], f)
    print('Done')
