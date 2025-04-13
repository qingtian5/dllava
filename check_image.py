import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm

import concurrent.futures

import warnings
warnings.filterwarnings("error", category=UserWarning)

def detele_iccfile(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)


def check_image(records):
    valid = []

    for i in tqdm(records):
        image_path = i['image']
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("图像空")
            else:
                detele_iccfile(image_path)  
        except Exception as e:
            # 处理异常
            print(f"{image_path} 出现错误: {e}")
            try:
                image_pil = Image.open(image_path).convert('RGB')
                output_image = str(Path(image_path).with_suffix(".jpg"))
                image_pil.save(output_image)
                detele_iccfile(output_image)  
                i['image'] = output_image
            except Exception as e:
                print(e)
                continue
        else:
            valid.append(i)

    return valid

if __name__ == "__main__":

    injson = json.load(open('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_with_allava_laion_vflan_evol_sharegpt4v.json'))

    res = []

    max_workers = 16

    injson_lists = [injson[i::max_workers] for i in range(max_workers)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        # 提交任务并获取 Future 对象
        futures = [pool.submit(check_image, records) for records in injson_lists]

        # 迭代 Future 对象，获取结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # 获取函数执行结果，如果执行出错，会抛出异常
            except Exception as e:
                print(f"{future} generated an exception: {e}")
            else:
                # print(result)  # 处理函数执行结果
                res += result

    print(f"res: {len(res)}")

    json.dump(res, open('/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_with_allava_laion_vflan_evol_sharegpt4v_clean.json'))