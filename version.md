# Version
- v0.1.15: mindgpt v6 采用 [unused] 格式 prompt

```
[unused0]system
你是一个名字叫做理想同学的AI数字生命体。[unused1]
[unused0]user
<image>
Please describe this picture[unused1]
[unused0]assistant
The image features a wooden bridge...
```

demo  pipeline: chat_gradio.py --prompt-template midngpt

- v0.1.16: mindgpt v6 采用 eos 格式 prompt

```python
>>> inputs = 'USER: [unused10]\n用中文回答下面问题：请描述一下这张照片 ASSISTANT:'
    # 输入：用中文回答下面问题：请描述一下这张照片
>>> tokenizer = AutoTokenizer.from_pretrained('/mnt/pfs-guan-ssai/cv/moxu/mindgpt-v6/',trust_remote_code=True,encode_special_tokens=True,additional_special_tokens=["[unused10]"])
>>> inputs_ids = tokenizer.encode(inputs)
    # inputs_ids: [1, 3148, 1001, 29901, 29871, 94886, 29871, 13, 30406, 65865, 46146, 67094, 41417, 30383, 31088, 87882, 39081, 319, 1799, 9047, 13566, 29901]
>>> tokenizer.decode(inputs_ids)
    # '<s> USER: [unused10] \n用中文回答下面问题：请描述一下这张照片 ASSISTANT:'
```

运行 demo pipeline: python chat_gradio.py --prompt-template vicuna

