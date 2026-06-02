# -*- coding: utf-8 -*-

"""Stage-attribution + scaling profiler for the segmented core.

The aggregate fwd+bwd benchmark (gpu_bench.py) is dominated by the **sequential
NaiveSSM scan**: every Mamba layer is an O(L) python/cuda loop, so with the
NaiveSSM fallback the Mamba layers — not attention — set the wall clock. The
segmented core adds ``segmented_local_layers`` extra Mamba layers on top of the
shared stage-1 encoder, which is why its *total* NaiveSSM time is higher even
though the part it actually optimises (the recurrent-depth **attention/RDT**
refinement) is far cheaper.

This profiler isolates the axis the segmented core is designed to save: the RDT
refinement cost. ``recurrent_steps`` re-runs the stage-2 attention block R times,
so RDT cost scales O(R * n^2) in the number of refined positions n. Running it
on ``n_seg = L / L_B`` block summaries instead of all ``L`` tokens cuts that to
O(R * (L / L_B)^2) — a L_B^2 reduction that grows with sequence length.

Run on the GPU server (not collected by unittest):

    PYTHONPATH=. python Model/tests/gpu_profile.py
"""

from __future__ import annotations

import math
import time

import torch

from Model.model import RDTForCausalLM
from Model.tests.gpu_bench import _base


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync()
    return (time.time() - t0) / iters * 1e3  # ms/iter


def profile_stages(bsz=1, seq=512, segment_len=8):
    """Attribute one segmented forward to stage-1 / stage-2(RDT) / stage-3."""
    dev = torch.device("cuda")
    cfg = _base("segmented", segment_len=segment_len)
    model = RDTForCausalLM(cfg).to(dev).eval()
    core = model.recurrent
    d = cfg.d_model
    e0 = torch.randn(bsz, seq, d, device=dev)
    word_pos = torch.arange(seq, device=dev).unsqueeze(0).expand(bsz, seq)
    morph = torch.zeros(bsz, seq, dtype=torch.long, device=dev)

    with torch.no_grad():
        backbone = core._run_stage1(e0, word_pos=word_pos, morph_depth=morph,
                                    attn_mask=None, causal=True)
        bidx = core._boundary_indices(seq, dev)
        n_seg = bidx.shape[0]
        summaries = backbone.index_select(1, bidx)
        seg_wp = torch.arange(n_seg, device=dev).unsqueeze(0).expand(bsz, n_seg)
        seg_mo = torch.zeros(bsz, n_seg, dtype=torch.long, device=dev)

        t_s1 = _time(lambda: core._run_stage1(e0, word_pos=word_pos,
                                              morph_depth=morph, attn_mask=None,
                                              causal=True))
        t_s2 = _time(lambda: core._refine(summaries, word_pos=seg_wp,
                                          morph_depth=seg_mo,
                                          total_steps=cfg.recurrent_steps,
                                          bptt_window=None))
        ctx = core._scatter_prev_context(summaries, seq)
        h = e0 + core.ctx_proj(ctx)
        t_s3 = _time(lambda: core._run_local(h, word_pos=word_pos,
                                             morph_depth=morph, attn_mask=None,
                                             causal=True))

    print(f"  b{bsz} s{seq} L_B{segment_len} (n_seg={n_seg}):", flush=True)
    print(f"    stage1 mamba x{cfg.stage1_mamba_layers}   : {t_s1:7.2f} ms", flush=True)
    print(f"    stage2 RDT   x{cfg.recurrent_steps} steps : {t_s2:7.2f} ms  <-- segmented optimises this", flush=True)
    print(f"    stage3 local x{cfg.segmented_local_layers}   : {t_s3:7.2f} ms", flush=True)
    del model
    torch.cuda.empty_cache()


def profile_rdt_scaling():
    """RDT-refinement cost: segmented (n_seg summaries) vs two_stage (L tokens).

    Isolates stage-2 only so the shared NaiveSSM stage-1 cost does not mask the
    quadratic attention saving. Reports ms/iter for the recurrent refinement at
    growing sequence length.
    """
    dev = torch.device("cuda")
    print("\n# RDT-refinement cost (stage-2 only), R=8 steps", flush=True)
    print(f"{'seq':>6}{'two_stage(L tok)':>20}{'segmented(L/8 seg)':>22}{'speedup':>10}", flush=True)
    for seq in (512, 1024, 2048, 4096):
        # two_stage refines over all L tokens
        ts_cfg = _base("two_stage")
        ts = RDTForCausalLM(ts_cfg).to(dev).eval().recurrent
        d = ts_cfg.d_model
        x = torch.randn(1, seq, d, device=dev)
        wp = torch.arange(seq, device=dev).unsqueeze(0)
        mo = torch.zeros(1, seq, dtype=torch.long, device=dev)
        try:
            with torch.no_grad():
                t_ts = _time(
                    lambda ts_core=ts: ts_core._refine_plain(
                        x,
                        word_pos=wp,
                        morph_depth=mo,
                        attn_mask=None,
                        causal=True,
                        total_steps=8,
                        bptt_window=None,
                    ),
                    iters=5,
                    warmup=2,
                )
        except RuntimeError:
            t_ts = float("nan")
        del ts
        torch.cuda.empty_cache()

        # segmented refines over n_seg = seq/8 summaries
        sg_cfg = _base("segmented", segment_len=8)
        sg = RDTForCausalLM(sg_cfg).to(dev).eval().recurrent
        n_seg = seq // 8
        xs = torch.randn(1, n_seg, d, device=dev)
        wps = torch.arange(n_seg, device=dev).unsqueeze(0)
        mos = torch.zeros(1, n_seg, dtype=torch.long, device=dev)
        with torch.no_grad():
            t_sg = _time(
                lambda sg_core=sg: sg_core._refine(
                    xs,
                    word_pos=wps,
                    morph_depth=mos,
                    total_steps=8,
                    bptt_window=None,
                ),
                iters=5,
                warmup=2,
            )
        del sg
        torch.cuda.empty_cache()

        sp = t_ts / t_sg if t_sg > 0 else float("nan")
        ts_str = "OOM" if math.isnan(t_ts) else f"{t_ts:7.2f} ms"
        print(f"{seq:>6}{ts_str:>20}{t_sg:>17.2f} ms{sp:>9.1f}x", flush=True)


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    print("# Stage attribution (forward, no_grad) — confirms NaiveSSM Mamba dominates", flush=True)
    for seq in (512, 1024):
        profile_stages(bsz=1, seq=seq, segment_len=8)
    profile_rdt_scaling()
    print("\nPROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
