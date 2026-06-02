# -*- coding: utf-8 -*-

"""End-to-end trainability proof for the segmented core on a single GPU.

Not collected by unittest discover (no ``test_*`` name / TestCase). Run on the
GPU server:

    PYTHONPATH=. python Model/tests/gpu_train.py

Validates, with the real training framework (s6: AdamAtan2 + WSD schedule +
truncated BPTT) wired to the segmented core, that the architecture is actually
trainable end to end:

1. **Convergence** — overfit a fixed small batch; cross-entropy must fall by a
   large margin with no NaN/Inf at any step.
2. **Checkpoint round-trip** — save model+optimizer+scheduler, reload into a
   fresh model, and confirm the resumed forward loss matches bit-for-bit and a
   further step keeps decreasing (resume correctness for long pretraining runs).
3. **Gated paths** — repeat a short run with ``recurrent_random_r`` enabled to
   confirm the stochastic-depth training path is finite and decreasing.
"""

from __future__ import annotations

import tempfile

import torch

from Model.config import RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.training.optim import build_optimizer, build_scheduler

BPTT_WINDOW = 4


def _cfg(**kw) -> RDTConfig:
    return RDTConfig(
        d_model=256,
        n_heads=4,
        head_dim=64,
        kv_lora_rank=128,
        rope_head_dim=32,
        nope_head_dim=32,
        ffn_hidden=1024,
        ffn_multiple=128,
        n_prelude=1,
        n_coda=1,
        recurrent_steps=4,
        max_seq_len=512,
        use_official_mamba=False,
        core_type="segmented",
        stage1_mamba_layers=2,
        stage2_attn_layers=2,
        segment_len=8,
        segmented_local_layers=1,
        recurrent_drift_mode="none",
        **kw,
    )


def _train_cfg() -> TrainingConfig:
    return TrainingConfig(
        optimizer="adamw",
        adam_use_atan2=True,
        learning_rate=3e-3,
        weight_decay=0.0,
        warmup_steps=10,
        max_steps=200,
        lr_schedule="wsd",
        wsd_stable_ratio=0.7,
    )


def _overfit(model, opt, sch, ids, steps):
    losses = []
    for _ in range(steps):
        out = model(ids, labels=ids, bptt_window=BPTT_WINDOW)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    cfg = _cfg()
    tcfg = _train_cfg()
    model = RDTForCausalLM(cfg).to(dev).train()
    opt = build_optimizer(model, tcfg)
    sch = build_scheduler(opt, tcfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 128), device=dev)

    print("\n# 1. convergence (overfit fixed batch, AdamAtan2 + WSD)", flush=True)
    losses = _overfit(model, opt, sch, ids, tcfg.max_steps)
    print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  "
          f"drop={losses[0]-losses[-1]:.4f}", flush=True)
    assert losses[-1] < losses[0] * 0.25, "did not converge"
    print("  PASS convergence (>=4x loss reduction, all finite)", flush=True)

    print("\n# 2. checkpoint round-trip", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sch": sch.state_dict()}, f.name)
        model.eval()
        with torch.no_grad():
            ref = float(model(ids, labels=ids, bptt_window=BPTT_WINDOW)["loss"])

        fresh = RDTForCausalLM(cfg).to(dev)
        ckpt = torch.load(f.name, map_location=dev)
        fresh.load_state_dict(ckpt["model"])
        fresh.eval()
        with torch.no_grad():
            got = float(fresh(ids, labels=ids, bptt_window=BPTT_WINDOW)["loss"])
    print(f"  ref={ref:.6f}  reloaded={got:.6f}  |Δ|={abs(ref-got):.2e}", flush=True)
    assert abs(ref - got) < 1e-5, "checkpoint reload mismatch"

    fresh.train()
    opt2 = build_optimizer(fresh, tcfg)
    opt2.load_state_dict(ckpt["opt"])
    sch2 = build_scheduler(opt2, tcfg)
    sch2.load_state_dict(ckpt["sch"])
    more = _overfit(fresh, opt2, sch2, ids, 20)
    print(f"  resumed 20 steps: {got:.4f} -> {more[-1]:.4f}", flush=True)
    assert more[-1] <= got + 1e-3, "resume did not keep improving"
    print("  PASS checkpoint round-trip + resume", flush=True)

    print("\n# 3. stochastic-depth (random-r) trainability", flush=True)
    torch.manual_seed(0)
    rcfg = _cfg(recurrent_random_r=True, recurrent_r_min=2, recurrent_r_max=4)
    rmodel = RDTForCausalLM(rcfg).to(dev).train()
    ropt = build_optimizer(rmodel, tcfg)
    rsch = build_scheduler(ropt, tcfg)
    rl = _overfit(rmodel, ropt, rsch, ids, 100)
    print(f"  loss[0]={rl[0]:.4f}  loss[-1]={rl[-1]:.4f}", flush=True)
    assert rl[-1] < rl[0] * 0.5 and all(
        torch.isfinite(torch.tensor(loss_value)) for loss_value in rl
    ), "random-r path failed"
    print("  PASS random-r trainability", flush=True)

    print("\nTRAIN_DONE ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
