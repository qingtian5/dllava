#!/bin/bash
PATH_ORI=${0%/*}
WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
cd ${WORK_PATH}

# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# pip install git+https://github.com/TRI-ML/prismatic-vlms
# pip install webdataset
# pip install -e '.[all]'
while true; do
    if pip install -e '.[all]'; then
        echo "Command executed successfully!"
        break
    else
        echo "Command failed, retrying..."
        sleep 1  # 等待 5 秒后重试
    fi
done
# pip install timm open_clip_torch
# pip install transformers==4.40.2
# pip install deepspeed
# git clone -b v2.3.2 https://github.com/HazyResearch/flash-attention /opt/flash-attention
# cd /opt/flash-attention && python setup.py install

pip install https://test-space-internal-cache.s3.bj.bcebos.com/cache/ssai-training/litiktoken/litiktoken-0.0.1-py3-none-any.whl
pip install blobfile==2.1.1

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

bash scripts/inference.sh