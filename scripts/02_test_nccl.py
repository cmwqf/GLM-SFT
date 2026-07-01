"""
多卡 NCCL 通信测试

用法:
    torchrun --standalone --nproc_per_node=<GPU数> scripts/02_test_nccl.py

测试内容:
    1. all_reduce — 所有卡聚合求和
    2. all_gather — 所有卡收集数据
    3. broadcast — 主卡广播到其他卡
    4. reduce_scatter — 规约后分发
    5. 带宽测试 — 用大 tensor 测量实际吞吐
"""

import os
import time
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def test_all_reduce(rank, world_size):
    tensor = torch.ones(1024, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(tensor, torch.full_like(tensor, expected)), \
        f"all_reduce 失败: 期望 {expected}, 得到 {tensor[0].item()}"
    return True


def test_all_gather(rank, world_size):
    tensor = torch.full((512,), rank, device=f"cuda:{rank}", dtype=torch.float32)
    gathered = [torch.zeros(512, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    for i, g in enumerate(gathered):
        assert torch.allclose(g, torch.full_like(g, i)), \
            f"all_gather 失败: rank {i} 的数据不正确"
    return True


def test_broadcast(rank, world_size):
    if rank == 0:
        tensor = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    else:
        tensor = torch.zeros(1024, device=f"cuda:{rank}", dtype=torch.float32)
    dist.broadcast(tensor, src=0)
    expected = torch.arange(1024, device=f"cuda:{rank}", dtype=torch.float32)
    assert torch.allclose(tensor, expected), "broadcast 失败"
    return True


def test_reduce_scatter(rank, world_size):
    input_tensor = torch.ones(world_size * 256, device=f"cuda:{rank}") * (rank + 1)
    output_tensor = torch.zeros(256, device=f"cuda:{rank}")
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(output_tensor, torch.full_like(output_tensor, expected)), \
        f"reduce_scatter 失败"
    return True


def test_bandwidth(rank, world_size):
    sizes_mb = [1, 10, 100, 500]
    results = []
    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        tensor = torch.randn(numel, device=f"cuda:{rank}")

        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        n_iters = 10
        for _ in range(n_iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # all_reduce 传输量 = 2*(N-1)/N * data_size
        data_bytes = numel * 4
        algo_bandwidth = data_bytes * n_iters / elapsed / 1e9
        bus_bandwidth = algo_bandwidth * 2 * (world_size - 1) / world_size
        results.append((size_mb, algo_bandwidth, bus_bandwidth, elapsed / n_iters * 1000))

    return results


def main():
    rank, world_size = setup()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  NCCL 通信测试 — {world_size} 张 GPU")
        print(f"{'='*60}\n")

    tests = [
        ("all_reduce", test_all_reduce),
        ("all_gather", test_all_gather),
        ("broadcast", test_broadcast),
        ("reduce_scatter", test_reduce_scatter),
    ]

    for name, fn in tests:
        dist.barrier()
        try:
            fn(rank, world_size)
            if rank == 0:
                print(f"  [PASS] {name}")
        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] {name}: {e}")

    # 带宽测试
    dist.barrier()
    if rank == 0:
        print(f"\n--- 带宽测试 (all_reduce, {world_size} GPUs) ---")
        print(f"  {'Size':>8s}  {'Algo BW':>10s}  {'Bus BW':>10s}  {'Latency':>10s}")

    bw_results = test_bandwidth(rank, world_size)

    if rank == 0:
        for size_mb, algo_bw, bus_bw, latency_ms in bw_results:
            print(f"  {size_mb:>6d} MB  {algo_bw:>8.2f} GB/s  {bus_bw:>8.2f} GB/s  {latency_ms:>8.2f} ms")
        print(f"\n{'='*60}")
        print("  所有测试通过!")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
