#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minachist
# Full text: LICENSE.MIT in the parent directory.
"""Quantize the MTP draft head's routed experts to INT4 for 3 x RTX 3090.

The checkpoint keeps the MTP module in BF16 (4.69 GiB of its 4.86 GiB are the
draft layer's 512 routed experts). Three cards have room for them only at INT4
g128, the scheme the body's experts use. This writes three files next to, not
into, the checkpoint; compose.yaml mounts them over the originals:

  config.json                      adds config group "group_mtp" for mtp.*.mlp.experts.*
  model.safetensors.index.json     points the new tensors at the file below
  model_extra_tensors.safetensors  the MTP tensors, experts as weight_packed/scale/shape

Round-to-nearest, symmetric, group size 128, with the per-group clip
(1.00 .. 0.70 of the max) that gives the least squared error. No calibration
data. Deterministic on a given device; a GPU and a CPU pick a different clip for
24 of the 19.7 million groups (near ties, one candidate apart).

  python3 make_mtp_int4.py /path/to/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound OUT_DIR

Needs torch and safetensors (the vLLM image has both). Uses a GPU if one is
visible (about a minute), otherwise the CPU (longer). Reads ~4.9 GB, writes 1.5 GB.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXTRA = "model_extra_tensors.safetensors"
GS = 128


def quantize(w: torch.Tensor, dev: torch.device):
    """w [out, in] -> (q int8 in [-8, 7], scale fp16 [out, in / GS], relative RMSE)"""
    w = w.to(dev, torch.float32)
    o, i = w.shape
    g = w.view(o, i // GS, GS)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    best_s = best_e = None
    for r in torch.linspace(1.0, 0.70, 16):
        s = (amax * r / 7.5).to(torch.float16).to(torch.float32)
        q = torch.clamp(torch.round(g / s), -8, 7)
        e = ((q * s - g) ** 2).sum(-1, keepdim=True)
        if best_e is None:
            best_s, best_e = s, e
        else:
            m = e < best_e
            best_s = torch.where(m, s, best_s)
            best_e = torch.where(m, e, best_e)
    q = torch.clamp(torch.round(g / best_s), -8, 7).to(torch.int8).view(o, i)
    rel = (best_e.sum() / (g**2).sum()).sqrt().item()
    return q, best_s.view(o, i // GS).to(torch.float16), rel


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """compressed-tensors pack-quantized layout: value + 8, eight per int32,
    element j of each group of eight in bits 4j..4j+3."""
    u = (q.to(torch.int64) + 8).view(q.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=q.device)
    v = (u << shifts).sum(-1)  # 0 .. 2**32 - 1
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out, errs = {}, []
    with safe_open(os.path.join(src, EXTRA), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.startswith("mtp.") and ".mlp.experts." in k and k.endswith(".weight"):
                base = k[: -len(".weight")]
                q, s, rel = quantize(t, dev)
                out[base + ".weight_packed"] = pack_int4(q).cpu().contiguous()
                out[base + ".weight_scale"] = s.cpu().contiguous()
                out[base + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
                errs.append(rel)
            else:
                out[k] = t
    if not errs:
        sys.exit(f"no BF16 MTP experts in {src}/{EXTRA}; already converted?")
    print(f"{len(errs)} expert matrices, relative RMSE mean {sum(errs) / len(errs):.4f} max {max(errs):.4f}")
    save_file(out, os.path.join(out_dir, EXTRA), metadata={"format": "pt"})
    os.chmod(os.path.join(out_dir, EXTRA), 0o644)

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    for k in [k for k in wm if k.startswith("mtp.") and ".mlp.experts." in k and k.endswith(".weight")]:
        del wm[k]
    for k in out:
        wm[k] = EXTRA
    json.dump(idx, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    qc = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
    group = json.loads(json.dumps(qc["config_groups"]["group_1"]))
    w = group["weights"]
    assert (w["num_bits"], w["group_size"], w["symmetric"]) == (4, GS, True), w
    group["targets"] = ["re:mtp\\..*\\.mlp\\.experts\\..*"]
    qc["config_groups"]["group_mtp"] = group
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print("wrote", ", ".join(sorted(os.listdir(out_dir))), "in", out_dir)


if __name__ == "__main__":
    main()
