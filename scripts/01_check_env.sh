#!/usr/bin/env bash
# 环境检查：GPU、驱动、PyTorch、NCCL、verl
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "========== 环境检查 =========="
echo ""

# 1. GPU 检测
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    pass "检测到 ${GPU_COUNT} 张 GPU"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    echo ""

    # 检查 GPU 之间是否可以 P2P
    echo "--- GPU 拓扑 ---"
    nvidia-smi topo -m 2>/dev/null || warn "无法获取 GPU 拓扑"
    echo ""
else
    fail "nvidia-smi 不可用，未检测到 GPU"
fi

# 2. CUDA 版本
echo "--- CUDA ---"
if command -v nvcc &>/dev/null; then
    pass "nvcc: $(nvcc --version | grep 'release' | awk '{print $6}')"
elif [ -f /usr/local/cuda/version.txt ]; then
    pass "CUDA: $(cat /usr/local/cuda/version.txt)"
elif nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    pass "CUDA (driver): ${CUDA_VER}"
else
    warn "未检测到 CUDA"
fi
echo ""

# 3. PyTorch
echo "--- PyTorch ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version (torch): {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}, {props.total_mem / 1024**3:.1f} GB')
" 2>&1 && pass "PyTorch OK" || fail "PyTorch 导入失败"
echo ""

# 4. NCCL
echo "--- NCCL ---"
python3 -c "
import torch.distributed as dist
print(f'NCCL available: {dist.is_nccl_available()}')
if hasattr(torch.cuda.nccl, 'version'):
    print(f'NCCL version: {torch.cuda.nccl.version()}')
" 2>&1 && pass "NCCL OK" || warn "NCCL 检查失败"
echo ""

# 5. verl
echo "--- verl ---"
python3 -c "import verl; print(f'verl version: {verl.__version__}')" 2>&1 \
    && pass "verl OK" || fail "verl 未安装，请运行: cd /home/ubuntu/verl && pip install -e ."
echo ""

# 6. 其他依赖
echo "--- 关键依赖 ---"
for pkg in transformers datasets pandas pyarrow hydra-core; do
    python3 -c "import importlib; m=importlib.import_module('$pkg'.replace('-','_')); v=getattr(m,'__version__','?'); print(f'  $pkg: {v}')" 2>/dev/null \
        && pass "$pkg" || fail "$pkg 缺失"
done
echo ""

# 7. 模型文件
echo "--- 模型文件 ---"
MODEL_DIR="/home/ubuntu/models/GLM-4.5-Air"
if [ -d "$MODEL_DIR" ]; then
    SAFETENSOR_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
    if [ "$SAFETENSOR_COUNT" -gt 0 ]; then
        MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
        pass "模型目录存在，${SAFETENSOR_COUNT} 个 safetensors 文件，总大小 ${MODEL_SIZE}"
    else
        fail "模型目录存在但没有 safetensors 权重文件，请下载模型"
    fi
else
    fail "模型目录不存在: $MODEL_DIR"
fi
echo ""

# 8. 数据文件
echo "--- 数据文件 ---"
DATA_FILE="/home/ubuntu/datasets/model-data-training/glm_chatml/train.jsonl"
if [ -f "$DATA_FILE" ]; then
    LINE_COUNT=$(wc -l < "$DATA_FILE")
    pass "数据文件存在，${LINE_COUNT} 条记录"
else
    fail "数据文件不存在: $DATA_FILE"
fi
echo ""

# 9. 磁盘和内存
echo "--- 系统资源 ---"
echo "磁盘:"
df -h /home/ubuntu | tail -1
echo "内存:"
free -h | head -2
echo "Swap:"
free -h | tail -1
echo ""

echo "========== 检查完成 =========="
