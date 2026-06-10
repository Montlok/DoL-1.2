# -*- coding: utf-8 -*-

"""Smoke checks for the two-stage RDT core.

Covers shapes, backward, the equal-length (no-downsample) path, the
same-direction orientation probe, and a five-mode drift-control ablation table.
Run from the repo root:

    PYTHONPATH=. python3 smoke_two_stage.py
"""

from __future__ import annotations

import dataclasses

import torch

from Model.config import two_stage_tiny_config
from Model.two_stage import TwoStageCore, TwoStageOutput


DRIFT_MODES = ["none", "norm", "decay", "both", "mhc"]


def _make_cfg(drift_mode: str):
    return dataclasses.replace(two_stage_tiny_config(), recurrent_drift_mode=drift_mode)


def _inputs(bsz: int, seq_len: int, d_model: int):
    e0 = torch.randn(bsz, seq_len, d_model)
    # Split each row into two contiguous words (a left and a right half) so the
    # orientation probe sees same-word adjacent pairs. A half-split avoids the
    # ``seq_len // 2`` divide-by-zero (seq_len < 2) and never yields >2 word ids.
    half = (seq_len + 1) // 2
    word_pos = (torch.arange(seq_len) >= half).long().unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)
    attn_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    return e0, word_pos, morph_depth, attn_mask


def smoke_shapes_and_backward() -> None:
    torch.manual_seed(0)
    cfg = two_stage_tiny_config()
    core = TwoStageCore(cfg)

    bsz, seq_len = 2, 16
    e0, word_pos, morph_depth, attn_mask = _inputs(bsz, seq_len, cfg.d_model)
    e0.requires_grad_(True)

    h, info = core(
        e0,
        word_pos=word_pos,
        morph_depth=morph_depth,
        attn_mask=attn_mask,
    )

    assert h.shape == (bsz, seq_len, cfg.d_model), h.shape
    assert info["steps_used"] == cfg.recurrent_steps
    assert info["global_semantic"].shape == (bsz, cfg.d_model)

    h.sum().backward()
    assert e0.grad is not None and torch.isfinite(e0.grad).all()

    print("[shapes/backward]")
    print(f"  {tuple(e0.shape)} -> {tuple(h.shape)}; grad_norm={e0.grad.norm().item():.4f}")


def smoke_no_downsample_equal_length() -> None:
    torch.manual_seed(0)
    cfg = two_stage_tiny_config()
    assert cfg.two_stage_downsample is False
    core = TwoStageCore(cfg)
    core.eval()

    bsz, seq_len = 2, 20
    e0, word_pos, morph_depth, attn_mask = _inputs(bsz, seq_len, cfg.d_model)

    with torch.no_grad():
        out = TwoStageOutput(
            hidden=core(e0, word_pos=word_pos, morph_depth=morph_depth)[0],
            global_semantic=torch.zeros(bsz, cfg.d_model),
        )

    assert out.hidden.shape[1] == seq_len, "two-stage core must stay equal-length"
    assert out.compressed is None and out.seg_ids is None

    print("[no-downsample]")
    print(f"  seq_len preserved: in={seq_len} out={out.hidden.shape[1]}")


def smoke_orientation_probe() -> None:
    torch.manual_seed(0)
    cfg = two_stage_tiny_config()
    core = TwoStageCore(cfg)
    core.eval()

    bsz, seq_len = 2, 16
    e0, word_pos, morph_depth, attn_mask = _inputs(bsz, seq_len, cfg.d_model)

    with torch.no_grad():
        h, _info = core(e0, word_pos=word_pos, morph_depth=morph_depth, attn_mask=attn_mask)

    orient = TwoStageCore.slice_orientation(h, word_pos, attn_mask=attn_mask)
    assert torch.isfinite(orient)
    assert -1.0 <= orient.item() <= 1.0

    print("[orientation-probe]")
    print(f"  same-word adjacent cosine: {orient.item():.4f}")


def smoke_drift_ablation() -> None:
    bsz, seq_len = 2, 16

    print("[drift-ablation]")
    print(f"  {'mode':<6} {'out_norm':>10} {'grad_norm':>10} {'orient':>8}")

    for mode in DRIFT_MODES:
        torch.manual_seed(0)
        cfg = _make_cfg(mode)
        core = TwoStageCore(cfg)

        e0, word_pos, morph_depth, attn_mask = _inputs(bsz, seq_len, cfg.d_model)
        e0.requires_grad_(True)

        h, _info = core(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
        )
        h.sum().backward()

        orient = TwoStageCore.slice_orientation(h.detach(), word_pos, attn_mask=attn_mask)

        assert torch.isfinite(h).all(), f"{mode} produced non-finite output"
        assert torch.isfinite(e0.grad).all(), f"{mode} produced non-finite grad"

        print(
            f"  {mode:<6} {h.norm().item():>10.3f} "
            f"{e0.grad.norm().item():>10.3f} {orient.item():>8.4f}"
        )


def main() -> None:
    smoke_shapes_and_backward()
    smoke_no_downsample_equal_length()
    smoke_orientation_probe()
    smoke_drift_ablation()
    print("\nOK")


if __name__ == "__main__":
    main()
