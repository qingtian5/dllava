#!/bin/bash
PATH_ORI=${0%/*}
WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
cd ${WORK_PATH}

# export NCCL_IB_HCA=^mlx5_0,^mlx5_1,mlx5_2,^mlx5_3,mlx5_4,^mlx5_5,mlx5_6,^mlx5_7,mlx5_8

# pip config set global.index-url https://mirrors.aliyun.com/pypi/simplepytho
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/triton-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/pyarrow-15.0.2-cp310-cp310-manylinux_2_28_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/torch-2.2.1-cp310-cp310-manylinux1_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/bitsandbytes-0.43.0-py3-none-manylinux_2_24_x86_64.whl
# pip install /mnt/pfs-guan-ssai/cv/moxu/xtuner-main/whl/opencv_python-4.9.0.80-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

cd /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/mindgpt-v

pip install timm webdataset
pip install -e '.[all]'
# pip install torch torchvision torchaudio
pip install transformers==4.40.2
pip install deepspeed
# git clone -b v2.3.2 https://github.com/HazyResearch/flash-attention /opt/flash-attention
# cd /opt/flash-attention && python setup.py install

pip list | grep transformers

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

cfg=stage3_llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages

cd /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/stage3_llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages_new
xtuner convert pth_to_hf ${cfg} \
 /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/stage3_llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages_new/1122_sft_car_260k_short_a_pure_t_shuffle/iter_558.pth \
 /mnt/pfs-mc0p4k/cv/team/lishanshan/data/code_repo/Friday_SFT/xtuner-new-lishanshan/work_dirs/stage3_llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages_new/1122_sft_car_260k_short_a_pure_t_shuffle/iter_558_hf