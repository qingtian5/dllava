import concurrent.futures
import os
import io
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import random
import pandas as pd
from PIL import Image
from pathlib import Path

# from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
# from xtuner.dataset.utils import Packer, encode_fn
# from xtuner.utils import PROMPT_TEMPLATE

# 定义处理单个文件的函数
def process_file(files):

    for file_path in files:

        # print(f"file_path: {file_path}")

        df = pd.read_parquet(file_path)

        file_name = f"{file_path.parent.name}_{file_path.stem}"

        # print(f"file_name: {file_name}")

        records = []

        t = 0

        for index, row in tqdm(df.iterrows(), desc=file_name): 

            try:

                id = file_name + "_" + "{:08d}".format(index)
                
                images = []

                for idx, image_bytes_array in enumerate(row['images']):

                    
                    image_bytes = image_bytes_array['bytes']
                    image_path = image_bytes_array['path']

                    if image_bytes is None and image_path is None: continue

                    if image_bytes is not None:
                        image_ouput_path = f'/mnt/pfs-mc0p4k/cv/team/yanghongfu/the_cauldron/images/{id}_{idx}.jpg'
                        image_stream = io.BytesIO(image_bytes)
                        image = Image.open(image_stream).convert('RGB')
                        image.save(image_ouput_path)

                    if image_path is not None:

                        if 'COCO' in image_path:
                            image_ouput_path = image_path.replace('/fsx/m4/datasets/downloads/extracted/19661dd042ca9f1e30d4843440822fb38f18fc5d649662da48018561ddec94e2/', '/mnt/pfs-guan-ssai/cv/moxu/data/images/')

                        if 'CLEVR_v1.0' in image_path:
                            image_ouput_path = image_path.replace('/fsx/m4/datasets/downloads/extracted/3c4c03ad359586cd332583e3a61e1ef5808cc52f30cef52648847fd19d477eac/', '/mnt/pfs-mc0p4k/cv/team/yanghongfu/the_cauldron/images/')

                    images.append(image_ouput_path)

                conversations = []

                for idx, qa in enumerate(row['texts']):
                    if idx == 0:
                        if len(images) == 1:
                            conversations.append({'from': 'human', 'value': f"<image>\n{qa['user']}"})
                        elif len(images) > 1:
                            conversations.append({'from': 'human', 'value': '<image>\n' * len(images) + qa['user']})
                        else:
                            conversations.append({'from': 'human', 'value': qa['user']})
                    else:
                        conversations.append({'from': 'human', 'value': qa['user']})
                    conversations.append({'from': 'gpt', 'value': qa['assistant']})

                if len(images) > 0:
                    example = {
                        'image': images[0] if len(images) == 1 else images,
                        'conversations': conversations,
                        'id': id 
                    }
                else:
                    example = {
                        'conversations': conversations,
                        'id': id 
                    } 

                records.append(example)

                if (len(images) == 0 or len(images) > 2 or len(conversations) > 2 or image_path is not None) and t == 0: 
                    print(example, flush=True)
                    t += 1

            except Exception as e:
                print(f"ERROR in {id}: {e}", flush=True)
                return f"ERROR in {file_path}"

        json.dump(records, open(file_path.with_suffix('.json'), 'w'))

    return True


def merge_images_horizontally(image1_path, image2_path, output_path=None):
    # 打开两张图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 确定合并后图片的宽度
    merged_width = 1024 * 2
    height = 1024

    # 调整图片1的大小
    image1 = image1.resize((merged_width // 2, height))

    # 调整图片2的大小
    image2 = image2.resize((merged_width // 2, height))

    # 创建一个新的空白图片，用于存放合并后的结果
    merged_image = Image.new('RGB', (merged_width, height))

    # 将图片1粘贴到左边
    merged_image.paste(image1, (0, 0))

    # 将图片2粘贴到右边
    merged_image.paste(image2, (image1.width, 0))

    # 保存合并后的图片
    merged_image.save(output_path)


def merge_two_images(records):
    
    # for i in tqdm(records):

    #     if isinstance(i['image'], str): continue 

    #     try:
    #         assert len(i['image']) == 2

    #         merge_image_output_path = i['image'][0].replace('_0', "")
    #         merge_images_horizontally(i['image'][0], i['image'][1], merge_image_output_path)

    #         i['image'] = merge_image_output_path

    #         i['conversations'][0]['value'] = i['conversations'][0]['value'].replace('<image>\n<image>\n', '<image>\n')

    #     except Exception as e:
    #         print(f"{e} in {i}")

    # return records

    return [i for i in records if 'nlvr2' in i['id']]

if __name__ == "__main__":

    num_workers = 64

    json_file = '/mnt/pfs-mc0p4k/cv/team/yanghongfu/the_cauldron/the_cauldron_merge_two_images.json'

    print(f'load json {json_file}')
    injson = json.load(open(json_file))
    print(f"{len(injson)} records")
    injson = [injson[i::num_workers] for i in range(num_workers)]

    # directory_path = Path('/mnt/pfs-mc0p4k/cv/team/yanghongfu/the_cauldron/clevr_math')
    # parquet_files = list(directory_path.rglob('*.parquet'))

    # print(f"parquet_files: {len(parquet_files)}", flush=True)

    # files = [parquet_files[i::num_workers] for i in range(num_workers)]


    res = []

    # 创建一个线程池，最大线程数为 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        # 提交任务并获取 Future 对象
        futures = [pool.submit(merge_two_images, records) for records in injson]

        # 迭代 Future 对象，获取结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # 获取函数执行结果，如果执行出错，会抛出异常
            except Exception as e:
                print(f"{future} generated an exception: {e}", flush=True)
            else:
                # print(result, flush=True)  # 处理函数执行结果

                res += result

    print(f"res: {len(res)}")

    json.dump(res, open('/mnt/pfs-mc0p4k/cv/team/yanghongfu/the_cauldron/nlvr2.json', 'w'))

