#!/bin/bash

cd /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/mindgpt-v

pip install -e '.[all]'
pip install blobfile

LLM_MODEL=/mnt/pfs-guan-ssai/cv/sunhaoyi/sft-mind-gpt-v7moe-1010_dpo_dpo-0923_2560_n12b3e2_1010-seed42-ckpt-3354.back
MODEL_HF=/mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs/llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages/sft-mindgpt-v7moe-siglipdino-448-sw-minimonkey-img-slice-300m-stage2-cambrian-cauldron-great-intergration-50m-ft_iter_25296_hf
MODEL_NAME=MindGPT-3OV-base-1111

xtuner mmbench ${LLM_MODEL} \
--visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
--dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
--llava ${MODEL_HF} \
--prompt-template mindgpt \
--data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OV_Stage3/1101_plant_zice.tsv \
--dynamic-image-size 448 \
--num-sub-images 6 \
--projector-num-queries 256 64 \
--work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/Plant_zice/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fox_ocr_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/Fox/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fudanvi_ctr_hard_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/FudanVL_hard/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fudanvi_ctr_normal_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/FudanVL_normal/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/OCRBench.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/OCRBench/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/TextVQA_VAL.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/TextVQA/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/ChartQA_TEST.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/ChartQA/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/MMBench_DEV_CN.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/MMBench_DEV_CN/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/MMBench_DEV_EN.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/MMBench_DEV_EN/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template mindgpt \
# --data-path /mnt/pfs-mc0p4k/cv/team/zhanghongyuan/mindgpt-v-yanghongfu/eval_tsv/MMStar.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/MindGPT-3OV/MMStar/${MODEL_NAME}
