#!/usr/bin/env bash
# GLM-4.5-Air full SFT with verl, using the DCP sharded-load path (meta-init ->
# FSDP2 fully_shard -> torch.distributed.checkpoint.load). No rank0 full-load, no
# broadcast; peak per-rank memory ~ model_size / world_size. Scales single-node -> 48 GPU.
#
# Prereq (once): convert the HF checkpoint to a DCP checkpoint:
#   python3.11 scripts/convert_hf_to_dcp.py \
#       --hf-path /data/models/GLM-4.5-Air --out-dir /data/models/GLM-4.5-Air-dcp
# And apply the verl patch:  bash verl_patch/apply_patch.sh
set -euo pipefail

# ---- cluster topology (defaults = single node, 8 GPU) ----
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- paths ----
MODEL_PATH="${MODEL_PATH:-/data/models/GLM-4.5-Air}"
export VERL_DCP_CKPT_PATH="${VERL_DCP_CKPT_PATH:-/data/models/GLM-4.5-Air-dcp}"   # <-- enables DCP load
DATA_DIR="${DATA_DIR:-/home/ec2-user/sft-data}"
OUT_DIR="${OUT_DIR:-/home/ec2-user/sft-output/glm45air-sft}"

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
echo "[run] nodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC_PER_NODE world=$WORLD_SIZE"
echo "[run] VERL_DCP_CKPT_PATH=$VERL_DCP_CKPT_PATH"

if [[ ! -d "$VERL_DCP_CKPT_PATH" ]]; then
  echo "ERROR: DCP checkpoint $VERL_DCP_CKPT_PATH missing. Run scripts/convert_hf_to_dcp.py first." >&2
  exit 1
fi

torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train_1000.parquet" \
  data.val_files="$DATA_DIR/val_1000.parquet" \
  data.messages_key=messages \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=4096 \
  data.truncation=left \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=8192 \
  data.ignore_input_ids_mismatch=True \
  model.path="$MODEL_PATH" \
  +model.override_config.attn_implementation=sdpa \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  model.trust_remote_code=true \
  engine=fsdp \
  engine.dtype=bfloat16 \
  optim.lr=1e-5 \
  optim.lr_scheduler_type=cosine \
  optim.lr_warmup_steps_ratio=0.1 \
  optim.weight_decay=0.01 \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.project_name=sft-glm45air \
  trainer.experiment_name="glm45air-1k-fullsft-${WORLD_SIZE}gpu-dcp" \
  trainer.total_epochs=1 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  +trainer.logger=[console,wandb] \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes="$NNODES"
