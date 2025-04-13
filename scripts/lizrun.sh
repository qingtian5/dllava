TH_ORI=${0%/*}
WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
cd ${WORK_PATH}
pip install blobfile


 pip install timm webdataset
 pip install -e '.[all]'
 # pip install torch torchvision torchaudio
 pip install transformers==4.40.2
 pip install deepspeed

MASTER_IP=""
if [ "${RANK}" == "0" ];then
  while [[ "$MASTER_IP" == "" ]]
    do
        MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
        # MASTER_IP=127.0.0.1
        sleep 1
    done
else
  ## Convert DNS to IP for torch
  MASTER_IP=`getent hosts ${MASTER_ADDR} | awk '{print $1}'` # Ethernet
fi

echo "MASTER_IP: ${MASTER_IP}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"

# while true
# do
#     sleep 1800
# done

# echo "NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} PORT=${MASTER_PORT} ADDR=${MASTER_IP} NODE_RANK=${RANK} xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2"
# NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} PORT=${MASTER_PORT} ADDR=${MASTER_IP} NODE_RANK=${RANK} xtuner train llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2

# NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} PORT=${MASTER_PORT} ADDR=${MASTER_IP} NODE_RANK=${RANK} xtuner train llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2

# NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2

# Train
# NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2

# cd /mnt/pfs-guan-ssai/cv/sunhaoyi/xtuner-main/work_dirs/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune/svit_qa
# xtuner convert pth_to_hf llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune ./epoch_1.pth ../svit_qa_hf

# cd /mnt/pfs-guan-ssai/cv/sunhaoyi/xtuner-main/
# CUDA_VISIBLE_DEVICES=4 xtuner mmbench /mnt/pfs-guan-ssai/cv/yanghongfu/internlm2-chat-7b \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/.cache/clip-vit-large-patch14-336 \
# --llava ./work_dirs/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune/comvint_mix_hf \
# --prompt-template internlm2_chat \
# --data-path MMBench_DEV_EN.tsv \
# --work-dir ./work_dirs

#  0 MMBench_DEV_CN 0 MMBench_DEV_EN 1 CCBench 2 MME 3 MMMU_DEV_VAL 4 MMVet 5 MathVista_MINI
#  6 LLaVABench 7 AI2D_TEST 8 OCRBench 9 HallusionBench 10 MMStar
#  vlm SEEDBench_IMG
# cd /mnt/pfs-guan-ssai/cv/sunhaoyi/xtuner-new
# xtuner mmbench /mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b \
# --visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
# --dino-visual-encoder /mnt/pfs-guan-ssai/cv/cjy/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
# --llava ./work_dirs/llava_dolphin_yi_34b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune_multisubimages/dolphin_yi_34b_honeybee_256_hf \
# --dynamic-image-size 448 \
# --num-sub-images 6 \
# --prompt-template qwen_chat \
# --data-path LiAutoCabin.tsv \
# --work-dir ./dolphin_yi_34b_honeybee_256

# clip_eva
#cd /mnt/pfs-guan-ssai/cv/sunhaoyi/xtuner-new
#xtuner mmbench /mnt/pfs-guan-ssai/cv/moxu/MindGPT-V6-16B-stage2-26000/ \
#--visual-encoder /mnt/pfs-guan-ssai/cv/cjy/models/mindvit/2024_04_21/MindViT-EVA02-l-14-336-20240421 \
#--llava ./work_dirs/llava_mindgptv6_base_16b_qlora_eva_vit_large_p14_336_lora_e1_gpu8_finetune \
#--dynamic-image-size 224 \
#--prompt-template vicuna \
#--data-path LiAutoBench.tsv \
#--work-dir ./mindgpt_V_xlsx

# siglip + dinov2
#cd /mnt/pfs-guan-ssai/cv/Nucky/project/xtuner-new
#python xtuner/tools/mmbench.py /mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b \
#--visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
#--dino-visual-encoder /mnt/pfs-guan-ssai/cv/cjy/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
#--llava /mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/work_dirs_guan/dolphinyi34b-siglipdino-100m-ft-0.1_epoch_1_hf \
#--dynamic-image-size 448 \
#--num-sub-images 6 \
#--prompt-template qwen_chat \
#--data-path $1 \
#--work-dir ./sft_result/dolphin-2.9.1-yi-1.5-34b/0710

#cd /mnt/pfs-guan-ssai/cv/Nucky/project/xtuner-new
#python xtuner/tools/mmbench.py /mnt/pfs-guan-ssai/cv/sunhaoyi/dolphin-2.9.1-yi-1.5-34b \
#--visual-encoder /mnt/pfs-guan-ssai/cv/yanghongfu/siglip-so400m-patch14-384 \
#--dino-visual-encoder /mnt/pfs-guan-ssai/cv/cjy/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
#--llava /mnt/pfs-guan-ssai/cv/Nucky/project/xtuner-new/work_dirs/dolphin-2.9.1-yi-1.5-34b_0801_logo_1/object_hf_1 \
#--dynamic-image-size 448 \
#--num-sub-images 6 \
#--prompt-template qwen_chat \
#--data-path $1 \
#--work-dir ./sft_result/dolphin-

cd /mnt/pfs-guan-ssai/cv/yanghongfu/temp/mindgpt-v-yanghongfu
while true; do
    if pip install -e '.[all]'; then
        echo "Command executed successfully!"
        break
    else
        echo "Command failed, retrying..."
        sleep 1  # 等待 5 秒后重试
    fi
done
bash /mnt/pfs-guan-ssai/cv/yanghongfu/temp/mindgpt-v-yanghongfu/scripts/inference.sh
