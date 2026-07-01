# AWS EC2 多机多卡 SFT 指南

verl 的 SFT trainer 是纯 **torchrun SPMD** 模式，多机训练就是标准 PyTorch 多机 `torchrun`。
下面是在 AWS EC2 集群上跑 GLM-4.5-Air 多机 SFT 的完整流程。

## 1. 集群前置条件

| 项目 | 要求 |
|------|------|
| 实例类型 | 建议 p4d/p5/p5e（带 EFA 高速互联）；g5/g6 也可但走 TCP，跨机带宽低 |
| 网络 | 所有节点同一 VPC/子网；安全组内互信（放开 all traffic within SG）|
| 代码/模型/数据 | **所有节点路径必须一致**，强烈推荐挂共享盘（FSx for Lustre / EFS）|
| SSH | 主节点可免密 SSH 到所有节点（含自己）|
| 软件 | 每台机器都装好 verl、PyTorch、NCCL、EFA 驱动 |

### 共享存储 vs 各自拷贝
- **推荐 FSx/EFS**：模型和数据只放一份，所有节点直接读，最省事
- 若无共享盘：用 `rsync` 把代码/模型/数据同步到每台机器的相同路径

## 2. 关键网络配置

### 找到内网网卡名
```bash
ip -o -4 addr show   # 通常是 ens5 或 eth0
```
把网卡名传给 `NCCL_SOCKET_IFNAME`（脚本默认 ens5）。

### EFA（高速互联，p4d/p5 系列）
若实例支持 EFA，需要安装 aws-ofi-nccl 插件，脚本会自动启用：
```bash
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
```
验证 EFA 是否就绪：`fi_info -p efa`

若实例**不支持 EFA**（如 g5），设 `USE_EFA=0` 走 TCP。

## 3. 先验证多机 NCCL 连通性

在跑训练前，务必先测多机通信是否正常（跨机 NCCL 最容易出问题）：

```bash
# 主节点 (node 0):
MASTER_ADDR=<主节点IP> NNODES=2 NODE_RANK=0 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=<主节点IP> --master_port=29500 \
  scripts/02_test_nccl.py

# 从节点 (node 1):
MASTER_ADDR=<主节点IP> NNODES=2 NODE_RANK=1 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
  --master_addr=<主节点IP> --master_port=29500 \
  scripts/02_test_nccl.py
```

看带宽测试结果：EFA 应该有几十~上百 GB/s 的 bus bandwidth；TCP 只有几 GB/s。

## 4. 启动训练

### 方式 A：一键启动（推荐）
从主节点用 SSH 启动器拉起所有节点：
```bash
cp scripts/hosts.txt.example scripts/hosts.txt
# 编辑 hosts.txt 填入各节点内网 IP（第一行为主节点）

bash scripts/06_launch_cluster.sh 8    # 每台 8 卡
```

### 方式 B：手动在每台机器执行
```bash
# 主节点 (node 0):
MASTER_ADDR=172.31.10.1 NNODES=2 NODE_RANK=0 bash scripts/05_multinode_sft.sh 8
# 从节点 (node 1):
MASTER_ADDR=172.31.10.1 NNODES=2 NODE_RANK=1 bash scripts/05_multinode_sft.sh 8
```

## 5. 常用可调参数（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `NNODES` | 必填 | 机器总数 |
| `NODE_RANK` | 必填 | 本机编号，主节点=0 |
| `MASTER_ADDR` | 必填 | 主节点内网 IP |
| `NCCL_SOCKET_IFNAME` | ens5 | 内网网卡名 |
| `USE_EFA` | 1 | 是否用 EFA（无 EFA 设 0）|
| `USE_LORA` | 1 | LoRA 模式，省显存 |
| `GLOBAL_BSZ` | 64 | 全局 batch size（多机可调大）|
| `MAX_LENGTH` | 4096 | 序列截断长度 |

## 6. 常见问题

- **卡在 rendezvous / 连不上主节点**：检查安全组是否放开节点间 TCP（尤其 master_port 29500），网卡名是否正确
- **NCCL timeout**：`NCCL_SOCKET_IFNAME` 设错了，用 `ip -o -4 addr` 确认
- **EFA 不生效**：`fi_info -p efa` 无输出说明 EFA 驱动/插件没装好，先设 `USE_EFA=0` 跑通再优化
- **各节点权重不一致报错**：确认模型/数据在所有节点是同一份（共享盘或 rsync 校验）
- **全量微调 OOM**：GLM-4.5-Air 是 MoE，全量微调显存需求极大，测试阶段优先 `USE_LORA=1`
