#!/usr/bin/env bash
# 从主节点一键拉起整个集群的多机训练（通过 SSH）
#
# 免去手动登录每台机器。主节点读取 hostfile，SSH 到每台机器执行 05_multinode_sft.sh。
#
# 用法:
#   1. 编辑 scripts/hosts.txt，每行一个节点内网 IP（第一行是主节点）
#   2. 确保主节点可以免密 SSH 到所有节点（含自己）
#   3. bash scripts/06_launch_cluster.sh <每台GPU数>
#
# 前置:
#   - 所有节点代码/模型/数据路径一致（推荐共享盘 FSx/EFS）
#   - 主节点到各节点已配置 SSH 免密（ssh-copy-id 或同一 key）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hosts.txt}"

if [ "$#" -lt 1 ]; then
    echo "用法: bash scripts/06_launch_cluster.sh <每台GPU数>"
    echo "  需要先编辑 ${HOSTFILE}（每行一个节点内网IP，第一行为主节点）"
    exit 1
fi

NPROC_PER_NODE=$1

if [ ! -f "$HOSTFILE" ]; then
    echo "错误: hostfile 不存在: $HOSTFILE"
    echo "请创建该文件，每行一个节点内网 IP。"
    exit 1
fi

# 读取节点列表（忽略空行和注释）
mapfile -t NODES < <(grep -vE '^\s*#|^\s*$' "$HOSTFILE")
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}

# 远程项目目录（各节点上代码所在路径，默认与主节点相同）
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-$PROJECT_DIR}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# 透传给训练脚本的环境变量
PASS_ENV="MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} \
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/models/GLM-4.5-Air} \
USE_LORA=${USE_LORA:-1} MAX_LENGTH=${MAX_LENGTH:-4096} \
MICRO_BSZ=${MICRO_BSZ:-1} GLOBAL_BSZ=${GLOBAL_BSZ:-64} \
LR=${LR:-1e-5} EPOCHS=${EPOCHS:-1} \
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens5} USE_EFA=${USE_EFA:-1}"

echo "=========================================="
echo "  集群启动"
echo "  节点数: ${NNODES}"
echo "  主节点: ${MASTER_ADDR}"
echo "  每台GPU: ${NPROC_PER_NODE}"
echo "  节点列表: ${NODES[*]}"
echo "=========================================="

LOG_DIR="${PROJECT_DIR}/output/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    LOG_FILE="${LOG_DIR}/node_${i}_${NODE}.log"
    echo ">>> 在节点 ${i} (${NODE}) 上启动 (日志: ${LOG_FILE})"

    # 主节点(i==0)可以本地执行；这里统一走 SSH 保持一致
    ssh ${SSH_OPTS} "${SSH_USER}@${NODE}" \
        "cd ${REMOTE_PROJECT_DIR} && ${PASS_ENV} NODE_RANK=${i} \
         bash scripts/05_multinode_sft.sh ${NPROC_PER_NODE}" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "所有节点已启动，等待训练完成..."
echo "查看日志: tail -f ${LOG_DIR}/node_0_*.log"
echo ""

# 等待所有节点
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

if [ "$FAIL" = "0" ]; then
    echo "训练完成，所有节点正常退出。"
else
    echo "有节点异常退出，请检查 ${LOG_DIR} 下的日志。"
    exit 1
fi
