## 🛠️ 快速上手

### 推理
- 参考scripts/inference.sh


### 训练
- xtuner train <job_name>
#### mindgpt2.0
- 参考 scripts/lizrun_moe.sh
<!-- - 参考lizrun.sh
- 任务超参数调整路径 xtuner/configs/llava/~/finetune(pretrain)/~.py
- 比如，job_name = llava_internlm2_chat_20b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune，则应该编辑xtuner/configs/llava/internlm2_chat_20b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_20b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py
- 目前实验过的所有llm, visual encoder组合都可以在路径中找到 -->

#### 常用训练集
- image_folder: /mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/data/llava_data/llava_images
##### pretrain
- bcs558k: /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/data/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json 

##### finetune
- mix665k: /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json

#### 开始训练
- lizrun start -c lizrun.sh -n 1 -i reg-ai.chehejia.com/ssai/lizr/cu120/py310/pytorch:2.2.1-xtuner

#### pth_to_hf
- xtuner convert pth_to_hf <job_name> <./epoch_~.pth> <./epoch_~_hf>

### 评测
**TODO: VLMEVAL**
- xtuner mmbench <llm_path> --visual-encoder <vit_path> --prompt-template <> --data-path <~.tsv> --work-dir <> --llava <./epoch_~_hf>
- prompt template 请和训练时的参数配置保持一致
- work-dir 结果保存路径
- tsv评测文件
- mingpt2.0，参考以下代码
```
xtuner mmbench  /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/sft-mind-gpt-v7moe-1010_dpo_dpo-0923_2560_n12b3e2_1010-seed42-ckpt-3354 \
--visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/siglip-so400m-patch14-384 \
--dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
--llava /mnt/pfs-mc0p4k/cv/team/yanghongfu/work_dirs/llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages/sft-mindgpt-v7moe-siglipdino-448-sw-minimonkey-img-slice-300m-stage2-cambrian-cauldron-great-intergration-50m-ft_iter_13900_hf \
--prompt-template mindgpt \
--data-path /mnt/pfs-mc0p4k/cv/team/zhanghongyuan/mindgpt-v-yanghongfu/eval_tsv/MMBench_DEV_EN.tsv \
--dynamic-image-size 448 \
--num-sub-images 6 \
--projector-num-queries 256 64 \
--work-dir /mnt/pfs-mc0p4k/cv/team/zhanghongyuan/mindgpt-v-yanghongfu/workdirs/1234
```

```
  'MMBench_DEV_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv", 
  'MMBench_TEST_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv", 
  'MMBench_DEV_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv", 
  'MMBench_TEST_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv", 
  'CCBench': "https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv", 
  'MME': "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv", 
  'SEEDBench_IMG': "https://opencompass.openxlab.space/utils/VLMEval/SEEDBench_IMG.tsv", 
```

### 微调 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing)

XTuner 支持微调大语言模型。数据集预处理指南请查阅[文档](./docs/zh_cn/user_guides/dataset_prepare.md)。

  ```shell
  # 单卡
  xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  # 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  ```

  - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

  - 更多示例，请查阅[文档](./docs/zh_cn/user_guides/finetune.md)。

- **步骤 2**，将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```
