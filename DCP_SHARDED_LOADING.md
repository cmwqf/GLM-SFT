# GLM-SFT — GLM-4.5-Air 全参 SFT 的 DCP 分片加载补丁

用 [verl](https://github.com/volcengine/verl) 对 **GLM-4.5-Air（107B MoE）** 做多机多卡全参 SFT 时，
verl 默认的权重加载方式在大模型（尤其 MoE）上会导致 **CPU OOM / FSDP 初始化崩溃**，8 卡就不稳定，
根本无法扩到 48 卡。本仓库提供一套**离线 DCP fusion + 训练侧分片流式加载**的方案，把加载期的
**每卡峰值内存从 `O(整模型)` 降到 `O(整模型 / world_size)`**。

---

## 1. 问题：verl 默认加载在大 MoE 上必炸

verl 的 FSDP engine（`verl/workers/engine/fsdp/transformer_impl.py`）默认走 **rank0 full-load + broadcast**：

```
rank0:      from_pretrained() 把整模型 (~214GB bf16) 全量加载到 CPU
其他 rank:  init_empty_weights() 建 meta 空壳
FSDP wrap:  sync_module_states=True → rank0 逐层 broadcast 到所有 rank，然后才 shard
```

在 GLM-4.5-Air 上的实际后果（本项目复现）：

- rank0 CPU 必须装下完整 214GB；广播是巨大的串行瓶颈；
- FSDP init 阶段频繁 `ChildFailedError` / `SIGTERM`，8 卡就反复重启十几次；
- 48 卡跨 6 节点时，rank0 节点承受全部广播压力，NCCL 再优化也没用。

### 为什么「所有 rank 都 from_pretrained」也不行

一个常见的错误修法是让每个 rank 各自 `from_pretrained(..., low_cpu_mem_usage=True)`。这是**三重 materialization**：

| 环节 | 代价 |
|------|------|
| `from_pretrained` 全量构建 | 每 rank 一份完整模型 |
| MoE expert 融合（per-expert → fused）| 融合瞬间 per-expert + fused 两份同时驻留 |
| FSDP wrap flatten/shard | wrap 前每 rank 必须先持有整模型 |

`low_cpu_mem_usage` 只消除双份分配、`sync_module_states=False` 只省广播，**都不触及「FSDP 之前不能有 full model」这条根本约束**。
8 rank × 214GB = 1.7TB 卡在内存边缘，任一尖峰即 OOM；48 卡同理。

**根因**：在没解决权重加载布局之前就做了 full model instantiation，顺序本身错了 —— 分片必须**先于**加载发生。

---

## 2. 方案：离线 fusion + 训练侧 DCP 分片流式加载

正确的不变量：**任何 rank 在任何时刻都不能持有完整模型，峰值 = `O(model / world_size)`。**

要满足它，加载顺序必须反过来：`meta-init → 先分片 → 再把权重灌进分片`。唯一的硬骨头是 MoE 的
per-expert（物理层，`experts.0.gate_proj.weight`）↔ fused（逻辑层，`experts.gate_up_proj` 堆叠张量）
布局差异 —— 这个融合逻辑埋在 transformers 的 `from_pretrained` 里。

我们把这块高风险逻辑**一次性、离线**地做掉：

### Stage 1（离线，一次性）— HF → DCP fused checkpoint
单进程 `from_pretrained`（复用 HF 自己的融合，正确性 by construction）→ `torch.distributed.checkpoint.save`
存成 **fused 布局**的 DCP 分片 checkpoint。单进程只驻留 1 份模型，与训练/FSDP 路径完全隔离。
非持久 buffer（rotary `inv_freq` / `original_inv_freq`，仅 2 个、各 32 元素）也一并存入，训练侧无需重算。

### Stage 2（每次训练）— meta → FSDP2 → DCP 流式 load
```
meta-init（0 内存，所有 rank）
  → FSDP2 fully_shard（在 meta 上定分片计划，每 rank 只留 1/N 空 DTensor）
  → to_empty（只 materialize 自己那 1/N 的空存储）
  → dcp.load（DCP 只读每 rank 的 shard，从 world-size 无关的 checkpoint reshard 灌入 DTensor）
```
无 rank0 full-load、无 broadcast。48 卡下每卡加载峰值 ≈ `214GB / 48 + 单文件 buffer ≈ 9GB`。

---

## 3. 目录结构

```
GLM-SFT/
├── scripts/
│   ├── 07_convert_hf_to_dcp.py      # Stage 1：HF safetensors → DCP fused checkpoint（单进程）
│   ├── 08_verify_dcp_checksums.py   # 离线校验：DCP round-trip 是否保真（单进程 CPU，逐张量对拍）
│   ├── 09_test_fsdp2_dcp_load.py    # 端到端：8/48 卡 meta→fully_shard→dcp.load→forward/backward
│   └── 10_run_sft_dcp.sh            # 训练启动脚本（单机/多机通用，靠 VERL_DCP_CKPT_PATH 开启）
├── verl_patch/
│   ├── transformer_impl.py          # 打好补丁的 verl FSDP engine 完整文件
│   ├── transformer_impl.dcp.patch   # 对应的 unified diff（便于 review）
│   └── apply_patch.sh               # 部署脚本（自动备份原文件 + py_compile 校验）
└── DCP_SHARDED_LOADING.md           # 本文档
```
（`scripts/01_*`~`06_*` 为集群/环境/数据/多机的既有脚本，见 `scripts/MULTINODE.md`。）

---

## 4. 使用

```bash
# (1) 离线转换（一次性，~1 分钟加载 + 数分钟落盘）
python3.11 scripts/07_convert_hf_to_dcp.py \
    --hf-path /data/models/GLM-4.5-Air \
    --out-dir /data/models/GLM-4.5-Air-dcp

# (2) 校验转换保真（可选但推荐，单进程 CPU）
python3.11 scripts/08_verify_dcp_checksums.py \
    --hf-path /data/models/GLM-4.5-Air \
    --dcp-dir /data/models/GLM-4.5-Air-dcp
# 期望：VERIFY RESULT: PASS

# (3) 部署补丁到已安装的 verl
bash verl_patch/apply_patch.sh
# 或指定目录： bash verl_patch/apply_patch.sh /path/to/site-packages/verl/workers/engine/fsdp

# (4) 端到端加载测试（8 卡，走真实训练路径）
VERL_LOGGING_LEVEL=WARN torchrun --standalone --nproc_per_node=8 \
    scripts/09_test_fsdp2_dcp_load.py \
    --hf-path /data/models/GLM-4.5-Air --dcp-dir /data/models/GLM-4.5-Air-dcp

# (5) 训练（单机 8 卡）
bash scripts/10_run_sft_dcp.sh
# 多机（例：6 节点 × 8 卡 = 48 卡），每个节点执行：
NNODES=6 NODE_RANK=$RANK NPROC_PER_NODE=8 MASTER_ADDR=<node1_ip> \
    bash scripts/10_run_sft_dcp.sh
```

补丁通过环境变量 **`VERL_DCP_CKPT_PATH`** 开启：设置了就走 DCP 分片加载（强制 FSDP2），
不设置则完全退回 verl 原行为（非侵入、可回退）。

---

## 5. 补丁做了什么

`verl_patch/transformer_impl.py` 只在 `VERL_DCP_CKPT_PATH` 存在时改变行为：

- `_build_module`：所有 rank 在 **meta** 上 `from_config`（不加载任何权重），干掉 rank0 full-load 与
  `get_init_weight_context_manager` 的 rank0-only init hack。
- `_build_fsdp_module` → `_build_fsdp_module_dcp`：`apply_fsdp2`（复用 verl 自带的 FSDP2 wrap）→
  `to_empty` → `dcp.load`（params + buffers 就地流式灌入）→ `set_model_state_dict`。无 `sync_module_states` 广播。
- 非 DCP 路径（未设环境变量）保持 verl 原逻辑不变。

---

## 6. 验证结果（GLM-4.5-Air, 8×H200）

- **转换**：`106.85B params`，737 tensors（735 params/持久buffer + 2 非持久 rotary buffer），DCP 落盘 ~200GB。
- **离线 checksum 对拍**（`08_verify_dcp_checksums.py`）：`checked 737 tensors, mismatches: 0` → `VERIFY RESULT: PASS`。
- **8 卡端到端**（`09_test_fsdp2_dcp_load.py`）：
  - `dcp.load done (735 sharded + 2 buffers)`
  - `WEIGHT CHECK: PASS`（全局参数和位级一致，`rel=2.97e-16`，全部 finite）
  - `FORWARD: finite=True` / `BACKWARD: grads_finite=True`
  - **每卡显存 ~27GB = 214GB / 8**（加载期），峰值不变量成立。
- **真实 verl SFT**（打补丁后）：`Before FSDP, memory allocated: 0.00GB` —— FSDP 之前无 full model；
  训练期每卡 ~53GB（分片参数 + Adam 优化器状态 + 梯度），**CPU 仅 ~64GB**（对照原 rank0 路径需 214GB+）。
- **加载速度**：单进程转换加载 735 分片 ~20s；对照 verl 原 8-rank rank0-broadcast 加载需 ~6 min 且频繁在 FSDP init 崩溃。

> 环境：torch 2.11 / verl（FSDP2 + `torch.distributed.checkpoint`）/ H200 141GB。

### 关于显存容量（重要）

本补丁解决的是**加载期**的爆内存/崩溃问题，让加载峰值降到 `O(model/N)`。它**不改变训练期**
全参优化器的显存需求。GLM-4.5-Air（107B）全参 AdamW SFT 的分片显存约为：

```
fp32 master 参数 4B + Adam(m,v) 各 4B = 12 B/param（分片）  + bf16 参数 2B + 梯度 …
≈ 12 × 107e9 / N  字节
  N=8  → ~160 GB/卡  > 141 GB(H200)  ⟹ 训练期 OOM（在首个 optimizer step 分配 Adam 状态时）
  N=48 → ~35  GB/卡  ⟹ 可容纳
```

即：**8 卡放不下 107B 全参 Adam**（加载能过、会在 `optimizer.step()` 的 Adam 状态分配处 OOM）。
这正是需要 48 卡的原因；而 48 卡下 verl 原始 rank0-broadcast 加载会先崩，**本补丁是让 48 卡加载可行的前提**。
若要在少量卡上做全参训练，可开 `engine.offload_policy=true`（CPU offload 优化器/参数，牺牲速度）。
