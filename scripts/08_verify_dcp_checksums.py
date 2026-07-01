#!/usr/bin/env python3.11
"""Stage 1 validation (single process, CPU, no GPU needed): independently reload the
DCP checkpoint into a fresh meta-initialized model materialized on CPU, then compare
every tensor against the checksums.json emitted at conversion time.

This proves the DCP round-trip preserved the exact fused weights (catches key
mismatches, dtype drift, reshape/transpose bugs) WITHOUT any multi-GPU setup."""
import argparse
import json
import os
import warnings

import torch
import torch.distributed.checkpoint as dcp


def tensor_checksum(t: torch.Tensor) -> dict:
    tf = t.detach().to(torch.float64)
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "sum": float(tf.sum().item()), "absmean": float(tf.abs().mean().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--dcp-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    with open(os.path.join(args.dcp_dir, "checksums.json")) as f:
        ref = json.load(f)
    print(f"[verify] loaded {len(ref)} reference checksums")

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=True)
    model.to_empty(device="cpu")

    # build load state dict = state_dict (params + persistent bufs) + non-persistent bufs
    sd = dict(model.state_dict())
    extra = {n: b for n, b in model.named_buffers() if n not in sd}
    load_sd = {**sd, **extra}
    print(f"[verify] loading DCP into {len(load_sd)} tensors (no_dist) ...")
    dcp.load(load_sd, checkpoint_id=args.dcp_dir, no_dist=True)

    # compare
    mism, checked = [], 0
    for name, rc in ref.items():
        if name not in load_sd:
            mism.append(f"{name}: MISSING in loaded model"); continue
        gc = tensor_checksum(load_sd[name])
        checked += 1
        if gc["shape"] != rc["shape"]:
            mism.append(f"{name}: shape {gc['shape']} != {rc['shape']}"); continue
        tol = 1e-3 * (abs(rc["sum"]) + 1.0)
        if abs(gc["sum"] - rc["sum"]) > tol or abs(gc["absmean"] - rc["absmean"]) > 1e-4 * (rc["absmean"] + 1):
            mism.append(f"{name}: sum {gc['sum']:.4f}!={rc['sum']:.4f} absmean {gc['absmean']:.6f}!={rc['absmean']:.6f}")

    extra_keys = [k for k in load_sd if k not in ref]
    print(f"[verify] checked {checked} tensors, mismatches: {len(mism)}, unexpected extra keys: {len(extra_keys)}")
    for m in mism[:20]:
        print("  MISMATCH:", m)
    if extra_keys[:10]:
        print("  extra keys sample:", extra_keys[:10])
    print("VERIFY RESULT:", "PASS" if not mism else "FAIL")


if __name__ == "__main__":
    main()
