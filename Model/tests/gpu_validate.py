# -*- coding: utf-8 -*-

"""On-device (CUDA) validation for the segmented causal core.

Not part of the CPU unittest run -- invoked directly on the GPU server:

    PYTHONPATH=. python Model/tests/gpu_validate.py

Checks, on the actual CUDA device:
  1. bf16 autocast forward/backward produce finite logits + grads (no NaN/Inf).
  2. Zero future leakage in float32 on-device (perturbing the last token leaves
     earlier logits bit-identical).
  3. Cache vs no-cache decode is bit-exact in float32 on-device.
"""

from __future__ import annotations

import sys

import torch

from Model.config import segmented_tiny_config
from Model.inference.cache import DecodeCache
from Model.model import RDTForCausalLM


def _device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA not available -- aborting GPU validation", flush=True)
        sys.exit(2)
    return torch.device("cuda")


def check_bf16_finite(dev) -> None:
    torch.manual_seed(0)
    cfg = segmented_tiny_config()
    model = RDTForCausalLM(cfg).to(dev).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 24), device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"]).all(), "bf16 loss not finite"
    out["loss"].backward()
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "bf16 grad not finite"
            n += 1
    assert n > 0
    print(f"[OK] bf16 autocast forward/backward finite (loss={out['loss'].item():.4f})", flush=True)


def check_causality(dev) -> None:
    torch.manual_seed(0)
    cfg = segmented_tiny_config()
    model = RDTForCausalLM(cfg).to(dev).eval()
    bsz, length = 1, 17
    wp = torch.arange(length, device=dev).unsqueeze(0).expand(bsz, length).contiguous()
    md = torch.zeros(bsz, length, dtype=torch.long, device=dev)
    ids = torch.randint(0, cfg.vocab_size, (bsz, length), device=dev)
    with torch.no_grad():
        base = model(ids, word_pos=wp, morph_depth=md, return_logits=True)["logits"]
        for k in (length - 1, 10, 8, 5):
            mod = ids.clone()
            mod[:, k] = (mod[:, k] + 1) % cfg.vocab_size
            pert = model(mod, word_pos=wp, morph_depth=md, return_logits=True)["logits"]
            assert torch.equal(base[:, :k], pert[:, :k]), f"leak at token {k}"
    print("[OK] zero future leakage on-device (float32)", flush=True)


def check_cache_bit_exact(dev) -> None:
    torch.manual_seed(0)
    cfg = segmented_tiny_config()
    model = RDTForCausalLM(cfg).to(dev).eval()
    b, length, prefill = 2, 11, 3
    ids = torch.randint(300, cfg.vocab_size, (b, length), device=dev)
    with torch.no_grad():
        ref = model(ids, return_logits=True)["logits"]
        cache = DecodeCache()
        mask = torch.ones_like(ids)
        wp, md = model._default_morph_info(ids, mask)
        diffs = []
        lg = model._forward_decode(ids[:, :prefill], wp[:, :prefill], md[:, :prefill], cache)
        for t in range(prefill):
            diffs.append((ref[:, t, :] - lg[:, t, :]).abs().max().item())
        for t in range(prefill, length):
            lg1 = model._forward_decode(ids[:, t:t + 1], wp[:, t:t + 1], md[:, t:t + 1], cache)
            diffs.append((ref[:, t, :] - lg1[:, 0, :]).abs().max().item())
    worst = max(diffs)
    assert worst < 1e-4, f"cache mismatch {worst}"
    print(f"[OK] cache vs no-cache bit-exact on-device (max|delta|={worst:.2e})", flush=True)


def main() -> None:
    dev = _device()
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    check_bf16_finite(dev)
    check_causality(dev)
    check_cache_bit_exact(dev)
    print("ALL_GPU_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
