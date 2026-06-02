# -*- coding: utf-8 -*-

"""VRAM / throughput benchmark for the segmented core on a single GPU.

Invoked directly on the GPU server (not part of the unittest run):

    PYTHONPATH=. python Model/tests/gpu_bench.py

Sweeps segment_len, kv_share_budget and random-r, reporting peak VRAM and
forward+backward tokens/s. Also compares the segmented core against an
equal-config two_stage baseline to quantify the compute saving from running the
attention/RDT refinement over n_seg block summaries instead of all n_tok tokens.

Runs with the realistic training-memory settings (grad_ckpt_recurrent +
truncated BPTT) so the numbers reflect how the model is actually trained, and
uses the NaiveSSM backend so it runs without the official mamba-ssm build.
"""

from __future__ import annotations

import time

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM

BPTT_WINDOW = 4


def _base(core_type: str, segment_len: int = 8, **kw) -> RDTConfig:
    return RDTConfig(
        d_model=1024,
        n_heads=16,
        head_dim=64,
        kv_lora_rank=256,
        rope_head_dim=32,
        nope_head_dim=32,
        ffn_hidden=4096,
        ffn_multiple=256,
        n_prelude=2,
        n_coda=2,
        recurrent_steps=8,
        max_seq_len=4096,
        use_official_mamba=False,
        grad_ckpt_recurrent=True,
        core_type=core_type,
        stage1_mamba_layers=6,
        stage2_attn_layers=2,
        segment_len=segment_len,
        segmented_local_layers=2,
        recurrent_drift_mode="none",
        **kw,
    )


def _run(cfg: RDTConfig, bsz: int, seq: int, iters: int = 5):
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = RDTForCausalLM(cfg).to(dev).train()
    ids = torch.randint(0, cfg.vocab_size, (bsz, seq), device=dev)

    def _step():
        loss = model(ids, labels=ids, bptt_window=BPTT_WINDOW)["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(2):  # warmup
        _step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    tps = bsz * seq * iters / dt
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model
    return tps, peak


def main() -> None:
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    print(f"grad_ckpt_recurrent=True bptt_window={BPTT_WINDOW}", flush=True)

    configs = [
        ("segmented L_B=4", _base("segmented", segment_len=4)),
        ("segmented L_B=8", _base("segmented", segment_len=8)),
        ("segmented L_B=8 kvshare=2", _base("segmented", segment_len=8, kv_share_budget=2)),
        ("segmented L_B=8 random-r", _base("segmented", segment_len=8,
                                           recurrent_random_r=True,
                                           recurrent_r_min=2, recurrent_r_max=8)),
        ("two_stage (token-level RDT)", _base("two_stage")),
    ]

    for bsz, seq in [(4, 512), (2, 1024), (8, 512)]:
        print(f"\n# batch={bsz} seq={seq}", flush=True)
        print(f"{'config':<42}{'tok/s':>12}{'peakVRAM(GB)':>14}", flush=True)
        for name, cfg in configs:
            try:
                tps, peak = _run(cfg, bsz, seq)
                print(f"{name:<42}{tps:>12.0f}{peak:>14.2f}", flush=True)
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e) else "ERR"
                print(f"{name:<42}{msg:>12}  {str(e)[:48]}", flush=True)
                torch.cuda.empty_cache()

    print("\nBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
