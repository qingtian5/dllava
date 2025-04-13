#!/bin/bash

cd /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/mindgpt-v

pip install -e '.[all]'
pip install blobfile

LLM_MODEL=/mnt/pfs-mc0p4k/cv/team/lishanshan/model/Qwen2.5-7B-Instruct
MODEL_HF=/mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/llava_qwen2_5_7b_instruct_siglipdino_448_e1_gpu8_pretrain/1127_qwen2_5_7b_ins_665k_sft/epoch_1_hf
MODEL_NAME=new_encoder_qwen2_5_7b_1127_665k

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OV_Stage3/1101_plant_zice.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/Plant_zice/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fox_ocr_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/Fox/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fudanvi_ctr_hard_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/FudanVL_hard/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/fudanvi_ctr_normal_benchmark.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/FudanVL_normal/${MODEL_NAME}

xtuner mmbench ${LLM_MODEL} \
--visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
--dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
--llava ${MODEL_HF} \
--prompt-template qwen_chat \
--data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/OCRBench.tsv \
--dynamic-image-size 448 \
--num-sub-images 6 \
--projector-num-queries 256 64 \
--max-new-tokens 128 \
--work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/OCRBench/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/TextVQA_VAL.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/TextVQA/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/ChartQA_TEST.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --max-new-tokens 128 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/ChartQA/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/MMBench_DEV_CN.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/MMBench_DEV_CN/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/tsv_data/OCR_2.0/MMBench_DEV_EN.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/MMBench_DEV_EN/${MODEL_NAME}

# xtuner mmbench ${LLM_MODEL} \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-mc0p4k/cv/team/yanghongfu/mindgpt3ov/models/dinov2-large \
# --llava ${MODEL_HF} \
# --prompt-template qwen_chat \
# --data-path /mnt/pfs-mc0p4k/cv/team/zhanghongyuan/mindgpt-v-yanghongfu/eval_tsv/MMStar.tsv \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --projector-num-queries 256 64 \
# --work-dir /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/test_results/Qwen2_5_7b_New/MMStar/${MODEL_NAME}

echo "DONE"

while true
do
    sleep 1800
done
