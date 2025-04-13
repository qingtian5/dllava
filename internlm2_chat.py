import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, StoppingCriteriaList
# from xtuner.tools.utils import get_stop_criteria, get_streamer
import time

# from modelscope import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('/mnt/pfs-guan-ssai/cv/yanghongfu/internlm2-chat-7b', trust_remote_code=True)
# # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
# model = AutoModelForCausalLM.from_pretrained('/mnt/pfs-guan-ssai/cv/yanghongfu/internlm2-chat-7b', torch_dtype=torch.float16, trust_remote_code=True).cuda()
# model = model.eval()

# max_new_tokens = 1024
# gen_config = GenerationConfig(
#     max_new_tokens=max_new_tokens,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.75,
#     top_k=40,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id
#     if tokenizer.pad_token_id is not None else
#     tokenizer.eos_token_id,
# )

# stop_criteria = StoppingCriteriaList()
# Streamer = get_streamer(model)
# streamer = Streamer(
#     tokenizer) if Streamer is not None else None


# inputs ="""根据图片回答问题，图中是否有人存在"""
# input_ids = tokenizer.encode(inputs, return_tensors='pt').to(device=model.device)
# print(input_ids)

# generation_output = model.generate(
#     inputs=input_ids,
#     max_new_tokens=max_new_tokens,
#     generation_config=gen_config,
#     streamer=streamer,
#     bos_token_id=tokenizer.bos_token_id,
#     stopping_criteria=stop_criteria)

# print(
#     f'\n{"="*50}\nSample output:\n'
#     f'输入：{inputs}\n输出：{tokenizer.decode(generation_output[0])}\n{"="*50}\n'
# )

prompt = """
任务：判断输入句子是否为大语言模型训练数据的噪声数据。请输出一个0-10的整数分数，用<score></score>包裹分数。
评分规则：不通顺语句、大量重复文本内容都视为噪声。只要句子出现大量重复文本内容，分数都必须低于3。
输入句子：{}
输出：
"""

inputs = prompt.format('而且,还可以看到,今年公司营业成本全面上升,这可能反应了一个行业;一个巨大的上市公司，前面有很多基金公司和其他公司') 

# model_name = "/mnt/pfs-guan-ssai/cv/yanghongfu/DeepSeek-V2-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # `max_memory` should be set based on your devices
# max_memory = {i: "75GB" for i in range(8)}
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "system", "content": "You are a helpful assistant that helps me determine noisy data, and only outputs scores without any analysis"},
#     {"role": "user", "content": inputs}
# ]
# input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

# print(f'{"="*50}\nSample output:\n{inputs + result}\n{"="*50}')

# tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/cv/yanghongfu/internlm2-chat-1_8b-sft", trust_remote_code=True)
# # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
# model = AutoModelForCausalLM.from_pretrained("/mnt/pfs-guan-ssai/cv/yanghongfu/internlm2-chat-1_8b-sft", torch_dtype=torch.float16, trust_remote_code=True).cuda()
# model = model.eval()

# response, history = model.chat(tokenizer, inputs, history=[])
# print(f'{"="*50}\nSample output:\n{inputs + response}\n{"="*50}')


model_path = '/mnt/pfs-guan-ssai/cv/yanghongfu/Yi-1.5-6B-Chat/'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    # torch_dtype='auto'
).eval()

# model = model.cuda()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": inputs}
]


for i in range(50):
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_new_tokens=200)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f'{"="*50}\nSample output:\n{inputs + response}\n{"="*50}')



# device = "cuda:0" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained(
#     "/mnt/pfs-guan-ssai/cv/yanghongfu/Qwen1.5-MoE-A2.7B-Chat",
#     # "qwen/Qwen1.5-0.5B-Chat",
#     # torch_dtype="auto",
#     # device_map="auto"
# )
# model = model.to(device)

# tokenizer = AutoTokenizer.from_pretrained(
#     "/mnt/pfs-guan-ssai/cv/yanghongfu/Qwen1.5-MoE-A2.7B-Chat"
#     # "qwen/Qwen1.5-0.5B-Chat",
# )

# prompt = """
# 任务：判断输入句子是否为噪声数据。请输出一个0-10的整数分数，10为句子没有噪声，0为句子存在大量噪声，不需要分析
# 评分规则：不通顺语句、大量重复文本内容都视为噪声。
# 这是一个高分例子：输入句子：这张图片展示了一位女士和一只狗在海滩上。女士坐在沙滩上，面带微笑，似乎在与狗互动。狗则坐在她面前，伸出前爪好像在试图与女士握手。输出：<score>10</score>
# 这是一个低分例子：输入句子：白色柳条椅子，粉色花朵，白色桌子，粉色花朵，樱花，粉色花朵，白色柳条桌子，粉色花朵，白色柳条椅子，粉色花朵，白色桌子，樱花，粉色花朵，白色柳条桌子，粉色花朵，白色柳条椅子，粉色花朵。输出：<score>0</score>
# 输入句子：图中有一张白色带条纹的椅子，一个穿黄色衣服的女人，一个白色包，一个黑色带金色的表，一个白色笔记本，一个红色的水杯，一个绿色的植物，一个金色的戒指，一个黑色的戒指，一个白色的钱包，一个蓝色的笔，一个黑色的笔，一个放在桌子上的勺子，一个黄色的勺子，一个白色的勺子，一个放在桌子上的叉子，一个黄色的叉子，一个放在桌子上的杯子，一个黄色的杯子，一个放在桌子上的碗，一个黄色的碗，一个放在桌子上的筷子，一个黄色的筷子，一个放在桌子上的手机，一个白色的手机，一个放在桌子上的勺子，一个黄色的勺子，一个放在桌子上的叉子，一个黄色的叉子
# 输出：
# """

# prompt = """
# 输入句子: 绿叶嫩芽， 绿叶嫩芽， 绿叶嫩芽， 绿叶嫩芽， 绿叶嫩芽 绿叶嫩芽 绿叶嫩芽 绿叶嫩芽 绿叶嫩芽绿叶嫩芽绿叶嫩芽绿叶嫩芽 绿叶嫩芽 
# 输入句子是否重复
# 高品质椰子鞋男椰子真爆白满天星侧透满天星亚洲限定。
# """

# messages = [
#     {"role": "system", "content": "你是一个大模型训练助手，帮助我判断噪声数据"},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# start = time.time()
# scores = []
# for _ in range(10):
#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(response)
#     scores.append(response)
# print(time.time() - start)
# print(scores, len([i for i in scores if float(i) < 5]))