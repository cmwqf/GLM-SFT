#!/usr/bin/env bash
# GLM-4.5-Air SFT 测试训练（1000 条数据）
#
# 用法:
#   bash scripts/04_test_sft.sh <GPU数>
#
# 示例:
#   bash scripts/04_test_sft.sh 8           # 全量参数
#   USE_LORA=1 bash scripts/04_test_sft.sh 4  # LoRA 模式
#
# 在运行前，请先执行:
#   1. bash scripts/01_check_env.sh          # 确认环境
#   2. python scripts/03_prepare_data.py     # 准备数据

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$#" -lt 1 ]; then
    echo "用法: bash scripts/04_test_sft.sh <GPU数>"
    echo ""
    echo "环境变量:"
    echo "  USE_LORA=1          使用 LoRA (推荐，节省显存)"
    echo "  MAX_LENGTH=4096     最大序列长度"
    echo "  MICRO_BSZ=1         每卡 micro batch size"
    echo "  LR=1e-5             学习率"
    echo "  EPOCHS=1            训练轮数"
    exit 1
fi

NPROC=$1

# ---- 配置 ----
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/models/GLM-4.5-Air}
DATA_DIR="${PROJECT_DIR}/output/data"
TRAIN_FILE="${DATA_DIR}/train_1000.parquet"
VAL_FILE="${DATA_DIR}/val_1000.parquet"
SAVE_DIR="${PROJECT_DIR}/output/checkpoints/glm45air-sft-test"

USE_LORA=${USE_LORA:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}

MAX_LENGTH=${MAX_LENGTH:-4096}
MICRO_BSZ=${MICRO_BSZ:-1}
GLOBAL_BSZ=${GLOBAL_BSZ:-16}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}
# ---- 配置结束 ----

# 检查前置条件
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    echo "请先下载模型"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python scripts/03_prepare_data.py"
    exit 1
fi

# 构建额外参数
extra_args=()

if [ "${USE_LORA}" = "1" ]; then
    echo ">>> 使用 LoRA 模式 (rank=${LORA_RANK}, alpha=${LORA_ALPHA})"
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=all-linear"
    )
else
    echo ">>> 使用全量参数微调模式"
fi

echo ""
echo "=========================================="
echo "  GLM-4.5-Air SFT 测试训练"
echo "  GPU 数: ${NPROC}"
echo "  模型:   ${MODEL_PATH}"
echo "  数据:   ${TRAIN_FILE}"
echo "  LoRA:   ${USE_LORA}"
echo "  序列长: ${MAX_LENGTH}"
echo "  Batch:  global=${GLOBAL_BSZ}, micro=${MICRO_BSZ}/gpu"
echo "  LR:     ${LR}"
echo "  Epochs: ${EPOCHS}"
echo "=========================================="
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
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
    trainer.project_name=glm45air-sft-test \
    trainer.experiment_name=glm45air-1k-test \
    trainer.total_epochs=${EPOCHS} \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=${NPROC} \
    "${extra_args[@]}"
