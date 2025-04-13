#!/bin/bash
PATH_ORI=${0%/*}
WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
cd ${WORK_PATH}

export NCCL_IB_HCA=^mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH


# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# pip install git+https://github.com/TRI-ML/prismatic-vlms
# pip install webdataset
# pip install -e '.[all]'
# while true; do
#     if pip install -e '.[all]'; then
#         echo "Command executed successfully!"
#         break
#     else
#         echo "Command failed, retrying..."
#         sleep 1  # 等待 5 秒后重试
#     fi
# done
# pip install timm open_clip_torch
# pip install transformers==4.40.2
# pip install deepspeed
# git clone -b v2.3.2 https://github.com/HazyResearch/flash-attention /opt/flash-attention
# cd /opt/flash-attention && python setup.py install

test_xtuner() {
    xtuner > /dev/null 2>&1  # 执行 xtuner 命令，输出重定向到 /dev/null
    return $?  # 返回命令的退出状态
}

# 循环直到 xtuner 成功输出
while true; do
    if test_xtuner; then
        echo "xtuner is working correctly."
        break  # 如果命令成功，退出循环
    else
        echo "xtuner failed. Reinstalling..."
        pip install -e '.[all]'  # 重新执行安装
    fi
done

pip install https://test-space-internal-cache.s3.bj.bcebos.com/cache/ssai-training/litiktoken/litiktoken-0.0.1-py3-none-any.whl
pip install blobfile==2.1.1

pip list | grep transformers

MASTER_IP=""
while [[ "$MASTER_IP" == "" ]]
do
  MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
  sleep 1
done

echo "MASTER_IP: ${MASTER_IP}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"

cp /mnt/pfs-guan-ssai/cv/yanghongfu/xtuner-main/loops_custom.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/loops.py
echo "cp loops_custom.py for resume done "

export hidden_drop_out=0.02

export aux_loss_coef=0.005

export z_loss_coef=0.001

export use_capacity_factor=True
export moe_expert_capacity_factor=2.0
export moe_token_drop_policy=probs
# export moe_token_drop_policy=position

# cfg=llava_mindgpt_moe_siglipdino_448_e1_gpu8_pretrain_thumbnail
# cfg=llava_mindgpt_moe_siglipdino_448_e1_gpu8_pretrain_multisubimages
# cfg=llava_mindgpt_moe_siglipdino_lora_e1_gpu8_finetune_multisubimages

cfg=llava_mindgpt_moe_siglipdino_448_e1_gpu8_pretrain_multisubimages
echo ${cfg}
LOG_FILE=logs_mindgpt/${cfg}_${RANK}.log
echo ${LOG_FILE}
NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} PORT=${MASTER_PORT} ADDR=${MASTER_IP} NODE_RANK=${RANK} xtuner train ${cfg} --deepspeed deepspeed_zero2 2>&1 | tee ${LOG_FILE}

# python gpu.py

# while true
# do
#     sleep 1800
# done