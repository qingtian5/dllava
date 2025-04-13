from openai import OpenAI
import re

client = OpenAI(api_key="sk-1a8d1039b59b499c8dc933c54bce22d4", base_url="https://api.deepseek.com/")

prompt = """
任务：判断输入句子是否为噪声数据。请输出一个0-10的整数分数，10为句子没有噪声，0为句子存在大量噪声，不需要分析
评分规则：不通顺语句、大量重复文本内容都视为噪声。只要句子出现噪声，分数都必须低于3。
这是一个高分例子：输入句子：这张图片展示了一位女士和一只狗在海滩上。女士坐在沙滩上，面带微笑，似乎在与狗互动。狗则坐在她面前，伸出前爪好像在试图与女士握手。输出：<score>10</score>
这是一个低分例子：输入句子：白色柳条椅子，粉色花朵，白色桌子，粉色花朵，樱花，粉色花朵，白色柳条桌子，粉色花朵，白色柳条椅子，粉色花朵，白色桌子，樱花，粉色花朵，白色柳条桌子，粉色花朵，白色柳条椅子，粉色花朵。输出：<score>0</score>
输入句子：{}
输出：
"""

text = '在村庄旁的高地上，站着一个面带微笑、背着包的中年女人。她穿着一件红色上衣，双腿交叉，似乎在享受宁静的村庄景色。背景中可见一些建筑物，包括小屋和其他建筑，环境宁静而宜人。女人的微笑显示出她可能在思考愉快的事情或者欣赏周围的风景。整个场景散发着宁静和满足的氛围。'
input = prompt.format(text) 
print(input)


# for i in range(10):
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个大模型训练助手，帮助我判断噪声数据"},
        {"role": "user", "content": input},
    ]
)

print(response.choices[0].message.content)

match = re.search(r'<score>(\d+)</score>', response.choices[0].message.content)

print(float(match.group(1)))

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_name = "/mnt/pfs-guan-ssai/cv/yanghongfu/DeepSeek-V2-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # `max_memory` should be set based on your devices
# # max_memory = {i: "75GB" for i in range(8)}
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(result)
