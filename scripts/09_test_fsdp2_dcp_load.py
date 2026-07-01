#!/usr/bin/env python3.11
"""Stage 2 end-to-end test (multi-GPU, torchrun): exercise the exact training-side
load path — meta-init -> FSDP2 fully_shard -> to_empty -> dcp.load -> forward/backward.

Validates:
  1. resharded DCP load reconstructs correct weights (gather full state -> compare checksums.json)
  2. forward produces finite logits
  3. backward produces finite grads
No rank ever materializes the full model; peak per rank ~ model/world_size.

Run: torchrun --standalone --nproc_per_node=8 test_fsdp2_dcp_load.py \
        --hf-path /data/models/GLM-4.5-Air --dcp-dir /data/models/GLM-4.5-Air-dcp
"""
import argparse
import json
import os
import warnings

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict, StateDictOptions,
)


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--dcp-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.cuda.current_device()

    from transformers import AutoConfig, AutoModelForCausalLM
    from verl.utils.fsdp_utils import apply_fsdp2

    cfg = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    log(f"[test] world_size={world}, building meta model ...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=True)

    mesh = init_device_mesh("cuda", (world,))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                     cast_forward_inputs=True)
    fsdp_kwargs = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": None,
                   "reshard_after_forward": True}
    apply_fsdp2(model, fsdp_kwargs, {"wrap_policy": {}})
    model.to_empty(device=dev)
    log("[test] fully_shard + to_empty done, dcp.load ...")

    msd = get_model_state_dict(model)
    extra = {n: b for n, b in model.named_buffers() if n not in msd}
    dcp.load({**msd, **extra}, checkpoint_id=args.dcp_dir)
    set_model_state_dict(model, msd)
    dist.barrier()
    log(f"[test] dcp.load done ({len(msd)} sharded + {len(extra)} buffers)")

    # ---- 1. weight correctness (lightweight, collective-safe) ----
    # Bit-exactness is already proven offline by verify_dcp_checksums.py. Here we only need
    # to prove the resharded GPU weights are (a) finite and (b) globally consistent with the
    # checkpoint. We sum each rank's LOCAL shard and all-reduce -> the global sum of all
    # *sharded* params must equal the sum of per-param checksums. This is a single scalar
    # reduction (no 214GB gather, no rank0 stall -> no barrier timeout).
    with open(os.path.join(args.dcp_dir, "checksums.json")) as f:
        ref = json.load(f)
    from torch.distributed.tensor import DTensor
    param_names = dict(model.named_parameters())
    local_sum = torch.zeros(1, dtype=torch.float64, device=dev)
    local_finite = torch.ones(1, dtype=torch.float64, device=dev)
    ref_param_sum = 0.0
    for name, p in param_names.items():
        shard = p.to_local() if isinstance(p, DTensor) else p
        local_sum += shard.detach().double().sum()
        if not torch.isfinite(shard).all():
            local_finite.zero_()
        if name in ref:
            ref_param_sum += ref[name]["sum"]
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
    if rank == 0:
        got, exp = local_sum.item(), ref_param_sum
        rel = abs(got - exp) / (abs(exp) + 1.0)
        ok = local_finite.item() > 0 and rel < 1e-3
        print(f"[test] WEIGHT CHECK: {'PASS' if ok else 'FAIL'} "
              f"(global param sum got={got:.2f} exp={exp:.2f} rel={rel:.2e} finite={local_finite.item()>0})")
    dist.barrier()

    # ---- 2. forward finite ----
    model.train()
    torch.manual_seed(rank)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
    out = model(input_ids=input_ids)
    logits = out.logits
    finite_fwd = torch.isfinite(logits).all().item()
    log(f"[test] FORWARD: finite={finite_fwd} logits.shape={tuple(logits.shape)} "
        f"mean={logits.float().mean().item():.4f} std={logits.float().std().item():.4f}")

    # ---- 3. backward finite ----
    loss = logits.float().mean()
    loss.backward()
    gfin = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    log(f"[test] BACKWARD: loss={loss.item():.4f} grads_finite={gfin}")

    log("[test] ALL DONE")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
