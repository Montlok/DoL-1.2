# -*- coding: utf-8 -*-

"""Alignment evaluation CLI for post-trained RDT models.

Two modes, mirroring :mod:`Model.posttrain.eval`:

- ``--gate`` (default on): the bit-exact decode-cache consistency invariant on a
  random batch — the RL-critical guarantee that sampling log-probs equal scoring
  log-probs. Exits non-zero if it drifts past ``--gate-tol``.
- quality reports over a responses JSONL (``--responses``): each line is
  ``{"response": str, "reference": str|null}``. Prints verifiable-reward stats,
  Mongolian script purity, and ``<think>`` format compliance.

Usage::

    python -m scripts.eval_align --config tiny --gate
    python -m scripts.eval_align --responses gen.jsonl \\
        --exact-weight 1.0 --purity-weight 0.5 --format-weight 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    RDTConfig,
    base_config,
    pretrain_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Model.posttrain.eval import (  # noqa: E402
    decode_cache_consistency,
    format_compliance_rate,
    purity_report,
    reward_report,
)
from Model.posttrain.rewards import RewardConfig  # noqa: E402

CONFIG_CHOICES = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
}


def _tiny_cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8, rope_head_dim=4,
        nope_head_dim=4, ffn_hidden=64, ffn_multiple=32, n_prelude=2, n_coda=2,
        recurrent_steps=3, mamba_d_state=8, mamba_expand=2, mamba_headdim=8,
        use_official_mamba=False, max_seq_len=64, core_type="two_stage",
        stage1_mamba_layers=3, stage2_attn_layers=2, recurrent_drift_mode="mhc",
    )


def _load_responses(path: str) -> tuple[list[str], list[str | None]]:
    responses: list[str] = []
    references: list[str | None] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        responses.append(obj["response"])
        references.append(obj.get("reference"))
    return responses, references


def main() -> int:
    ap = argparse.ArgumentParser(description="RDT alignment eval")
    ap.add_argument("--config", choices=list(CONFIG_CHOICES) + ["gate_tiny"],
                    default="gate_tiny")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gate", action="store_true", help="run decode-cache gate")
    ap.add_argument("--gate-tol", type=float, default=1e-4)
    ap.add_argument("--gate-seq-len", type=int, default=11)
    ap.add_argument("--recurrent-steps", type=int, default=None)
    ap.add_argument("--responses", default=None, help="JSONL of responses to score")
    ap.add_argument("--exact-weight", type=float, default=0.0)
    ap.add_argument("--numeric-weight", type=float, default=0.0)
    ap.add_argument("--purity-weight", type=float, default=0.0)
    ap.add_argument("--purity-min-ratio", type=float, default=0.0)
    ap.add_argument("--format-weight", type=float, default=0.0)
    ap.add_argument("--purity-threshold", type=float, default=0.8)
    args = ap.parse_args()

    rc = 0

    if args.gate or args.responses is None:
        cfg = _tiny_cfg() if args.config == "gate_tiny" else CONFIG_CHOICES[args.config]()
        model = RDTForCausalLM(cfg).eval()
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state.get("model", state))
        torch.manual_seed(0)
        ids = torch.randint(300, cfg.vocab_size, (2, args.gate_seq_len))
        diff = decode_cache_consistency(
            model, ids, recurrent_steps=args.recurrent_steps
        )
        ok = diff < args.gate_tol
        print(f"[gate] decode_cache_consistency max_diff={diff:.3e} "
              f"tol={args.gate_tol:.1e} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            rc = 1

    if args.responses:
        responses, references = _load_responses(args.responses)
        reward_cfg = RewardConfig(
            exact_match_weight=args.exact_weight,
            numeric_match_weight=args.numeric_weight,
            purity_weight=args.purity_weight,
            purity_min_ratio=args.purity_min_ratio,
            format_weight=args.format_weight,
        )
        rep = reward_report(responses, references, reward_cfg)
        pur = purity_report(responses, min_ratio=args.purity_threshold)
        fmt = format_compliance_rate(responses)
        print(f"[reward] n={int(rep['n'])} mean={rep['reward_mean']:.4f} "
              f"min={rep['reward_min']:.4f} max={rep['reward_max']:.4f}")
        print(f"[purity] mean={pur['purity_mean']:.4f} "
              f"frac_below_{args.purity_threshold}={pur['frac_below']:.4f}")
        print(f"[format] think_compliance={fmt:.4f}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
