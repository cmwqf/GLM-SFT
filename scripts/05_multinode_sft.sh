#!/usr/bin/env bash
# GLM-4.5-Air 多机多卡 SFT 训练 (AWS EC2 集群)
#
# verl 的 SFT trainer 是纯 torchrun SPMD 模式，多机训练用标准 torchrun rendezvous。
# 需要在【每一台】节点上都运行本脚本，参数区分 node_rank。
#
# 用法（在每台机器上执行）:
#   MASTER_ADDR=<主节点内网IP> NNODES=<机器数> NODE_RANK=<本机编号> \
#       bash scripts/05_multinode_sft.sh <每台GPU数>
#
# 示例（2 台机器，每台 8 卡）:
#   # 主节点 (node 0, 内网 IP 假设 172.31.10.1):
#   MASTER_ADDR=172.31.10.1 NNODES=2 NODE_RANK=0 bash scripts/05_multinode_sft.sh 8
#   # 从节点 (node 1):
#   MASTER_ADDR=172.31.10.1 NNODES=2 NODE_RANK=1 bash scripts/05_multinode_sft.sh 8
#
# 前置条件（每台机器都要满足）:
#   1. 代码、模型、数据在所有节点路径一致（推荐用 FSx/EFS 共享盘，或 rsync 到每台）
#   2. verl 已安装
#   3. 节点间内网互通，安全组放开所有 TCP/UDP（同一 SG 内互信）
#   4. 数据已用 03_prepare_data.py 准备好

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$#" -lt 1 ]; then
    echo "用法: MASTER_ADDR=<IP> NNODES=<N> NODE_RANK=<R> bash scripts/05_multinode_sft.sh <每台GPU数>"
    exit 1
fi

NPROC_PER_NODE=$1

# ---- 多机参数（通过环境变量传入）----
MASTER_ADDR=${MASTER_ADDR:?"必须设置 MASTER_ADDR（主节点内网 IP）"}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:?"必须设置 NNODES（总机器数）"}
NODE_RANK=${NODE_RANK:?"必须设置 NODE_RANK（本机编号，主节点为 0）"}

# ---- 训练配置 ----
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/models/GLM-4.5-Air}
DATA_DIR="${PROJECT_DIR}/output/data"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_1000.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_1000.parquet}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/output/checkpoints/glm45air-sft-multinode}"

USE_LORA=${USE_LORA:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
MAX_LENGTH=${MAX_LENGTH:-4096}
MICRO_BSZ=${MICRO_BSZ:-1}
GLOBAL_BSZ=${GLOBAL_BSZ:-64}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}

# ---- AWS EFA / NCCL 网络配置 ----
# EFA (Elastic Fabric Adapter) 是 AWS 高速节点互联网卡。
# 如果实例类型支持 EFA（p4d/p5/p5e 等），保持下面设置以获得最佳带宽。
# 如果实例没有 EFA（如普通 g5），设置 USE_EFA=0 走 TCP。
USE_EFA=${USE_EFA:-1}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
# 指定内网网卡（AWS 通常是 ens5 / eth0，用 `ip -o -4 addr` 查看）
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens5}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}

if [ "${USE_EFA}" = "1" ]; then
    # 启用 EFA (需要安装 aws-ofi-nccl 插件)
    export FI_PROVIDER=${FI_PROVIDER:-efa}
    export FI_EFA_USE_DEVICE_RDMA=${FI_EFA_USE_DEVICE_RDMA:-1}
    export NCCL_PROTO=${NCCL_PROTO:-simple}
else
    # 纯 TCP，禁用 InfiniBand
    export NCCL_IB_DISABLE=1
fi

echo ""
echo "=========================================="
echo "  GLM-4.5-Air 多机 SFT 训练"
echo "  节点: NODE_RANK=${NODE_RANK} / NNODES=${NNODES}"
echo "  主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  本机 GPU 数: ${NPROC_PER_NODE}"
echo "  总 GPU 数: $((NNODES * NPROC_PER_NODE))"
echo "  EFA: ${USE_EFA}, 网卡: ${NCCL_SOCKET_IFNAME}"
echo "=========================================="
echo ""

extra_args=()
if [ "${USE_LORA}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=all-linear"
    )
fi

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.train_batch_size=${GLOBAL_BSZ} \
    data.micro_batch_size_per_gpu=${MICRO_BSZ} \
    data.max_length=${MAX_LENGTH} \
    data.truncation=left \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=8192 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    engine=fsdp \
    engine.dtype=bfloat16 \
    optim.lr=${LR} \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name=glm45air-sft \
    trainer.experiment_name=glm45air-multinode \
    trainer.total_epochs=${EPOCHS} \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger='["console"]' \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    "${extra_args[@]}"
