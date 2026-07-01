#!/usr/bin/env python3.11
"""Stage 1 (offline, single process): convert a HuggingFace safetensors checkpoint
into a DCP (torch.distributed.checkpoint) sharded checkpoint in the model's *fused*
parameter layout.

Why: for large MoE models (e.g. GLM-4.5-Air, 107B, per-expert safetensors), the
training-time load must never materialize the full model on any rank. The correct
pipeline is  meta-init -> FSDP2 fully_shard -> dcp.load  (peak per rank ~ model/N).
But `dcp.load` needs a checkpoint keyed by the model's *fused* param names
(`experts.gate_up_proj`), whereas HF safetensors are *per-expert*
(`experts.0.gate_proj.weight`). The fusion logic lives inside transformers'
`from_pretrained`. So we do the fusion exactly once here, by letting HF load the
model canonically, then dump the resulting state_dict as a DCP checkpoint.

This script is a single-process job (the only sanctioned use of from_pretrained):
it holds ONE full CPU copy of the model, fully isolated from the FSDP training path.

Outputs under <out_dir>:
  - DCP checkpoint files (.metadata + __0_0.distcp ...)
  - checksums.json : per-tensor {shape, dtype, sum, absmean} for offline validation
"""
import argparse
import json
import os
import warnings

import torch
import torch.distributed.checkpoint as dcp


def tensor_checksum(t: torch.Tensor) -> dict:
    tf = t.detach().to(torch.float64)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sum": float(tf.sum().item()),
        "absmean": float(tf.abs().mean().item()),
    }


def build_full_state_dict(model) -> dict:
    """params + persistent buffers (via state_dict) + non-persistent buffers
    (rotary inv_freq etc., which are NOT in state_dict but ARE needed after a
    meta-init since we skip HF's __init__ buffer computation on the training side)."""
    sd = dict(model.state_dict())  # params + persistent buffers (real tensors)
    existing = set(sd.keys())
    n_extra = 0
    for name, buf in model.named_buffers():
        if name not in existing:
            sd[name] = buf.detach().clone()
            n_extra += 1
    print(f"[convert] state_dict entries: {len(sd)} (non-persistent buffers added: {n_extra})")
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True, help="HuggingFace model dir (safetensors)")
    ap.add_argument("--out-dir", required=True, help="output DCP checkpoint dir")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    torch_dtype = getattr(torch, args.dtype)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"[convert] loading HF model from {args.hf_path} (dtype={args.dtype}) ...")
    cfg = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        config=cfg,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[convert] model loaded: {n_params / 1e9:.2f}B params, class={model.__class__.__name__}")

    sd = build_full_state_dict(model)

    # per-tensor checksums for offline validation (cheap, before we free the model)
    print("[convert] computing checksums ...")
    checksums = {name: tensor_checksum(t) for name, t in sd.items()}
    with open(os.path.join(args.out_dir, "checksums.json"), "w") as f:
        json.dump(checksums, f)
    print(f"[convert] wrote checksums.json ({len(checksums)} tensors)")

    # DCP save, single process (no distributed group). Each tensor stored as a
    # single full chunk -> dcp.load can reshard it to any world size / sharding.
    print(f"[convert] dcp.save -> {args.out_dir} ...")
    dcp.save(sd, checkpoint_id=args.out_dir, no_dist=True)
    print("[convert] DONE")


if __name__ == "__main__":
    main()
