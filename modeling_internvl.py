from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    print(f"orig_width, orig_height: {orig_width, orig_height}, aspect_ratio: {aspect_ratio}")

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    print(f"target_ratios: {target_ratios}")

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    print(f"target_aspect_ratio: {target_aspect_ratio}")

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]\

    print(f"target_width, target_height: {target_width, target_height}")
    print(f"blocks: {blocks}")

    # resize the image
    resized_img = image.resize((target_width, target_height))
    print(f"resized_img: {resized_img.size}")
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        print(f"box: {box}")
        # split the image
        split_img = resized_img.crop(box)
        split_img.save(f"/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/split_imgs/split_img{i}.jpg")
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "/mnt/pfs-guan-ssai/cv/yanghongfu/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_8bit=True).eval()

# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`
pixel_values = load_image('/mnt/pfs-guan-ssai/cv/rxd/data/xiaohongshu/images/part-0001/xhs_notes.2019-06-24/00004/5d0f4784000000002601b846_303725181561314359_00001.jpg', max_num=6).to(torch.bfloat16).cuda()
print(f"pixel_values: {pixel_values.shape}")


generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# single-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(question, response)

# # multi-round single-image conversation
# question = "请详细描述图片" # Please describe the picture in detail
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)

# question = "请根据图片写一首诗" # Please write a poem according to the picture
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)

# # multi-round multi-image conversation
# pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = "详细描述这两张图片" # Describe the two pictures in detail
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)

# question = "这两张图片的相同点和区别分别是什么" # What are the similarities and differences between these two pictures
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)

# # batch inference (single image per sample)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# image_counts = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ["Describe the image in detail."] * len(image_counts)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              image_counts=image_counts,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(question)
#     print(response)
