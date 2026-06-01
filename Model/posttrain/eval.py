# -*- coding: utf-8 -*-

"""Alignment evaluation suite for post-trained RDT models.

Two kinds of checks, deliberately kept separate:

1. **Correctness invariants** (must hold *exactly*, bit-exact within fp tol):
   the RL-critical guarantee that full-forward log-probs equal the incremental
   decode-cache log-probs used during sampling. If this drifts, every advantage
   and KL term in DPO/GRPO is silently wrong, so it is a hard gate.

2. **Quality reports** (scalar dashboards, *not* pass/fail): verifiable-reward
   breakdowns, Mongolian script purity, and ``<think>`` format compliance over a
   batch of decoded responses. These summarize alignment quality without a
   learned reward model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from Model.inference.cache import DecodeCache
from Model.model import RDTForCausalLM
from Model.posttrain.logprobs import logits_to_token_logprobs, sequence_logprobs
from Model.posttrain.rewards import (
    RewardConfig,
    format_reward,
    mongolian_script_ratio,
    reward_for,
)


def decode_cache_consistency(
    model: RDTForCausalLM,
    input_ids: torch.Tensor,
    prefill: int = 2,
    recurrent_steps: int | None = None,
) -> float:
    """Max abs diff between full-forward and incremental-decode log-probs.

    This pins the sampling/scoring self-consistency invariant. A value within
    ~1e-4 means the policy log-probs used by RL match what the cache produced
    during generation. Returns the scalar max difference so callers can gate.
    """
    model.eval()
    with torch.no_grad():
        full = sequence_logprobs(model, input_ids, recurrent_steps=recurrent_steps)

        cache = DecodeCache()
        mask = torch.ones_like(input_ids)
        wp, md = model._default_morph_info(input_ids, mask)
        length = input_ids.shape[1]
        steps = [
            model._forward_decode(
                input_ids[:, :prefill], wp[:, :prefill], md[:, :prefill], cache
            )
        ]
        for t in range(prefill, length):
            steps.append(
                model._forward_decode(
                    input_ids[:, t : t + 1], wp[:, t : t + 1], md[:, t : t + 1], cache
                )
            )
        logits = torch.cat(steps, dim=1)
        incr = logits_to_token_logprobs(logits[:, :-1, :], input_ids[:, 1:])
    return float((full - incr).abs().max())


def reward_report(
    responses: Sequence[str],
    references: Sequence[str | None] | None,
    cfg: RewardConfig,
) -> dict[str, float]:
    """Mean / min / max of the combined verifiable reward over responses."""
    if references is None:
        references = [None] * len(responses)
    scores = torch.tensor(
        [reward_for(r, ref, cfg) for r, ref in zip(responses, references)],
        dtype=torch.float32,
    )
    return {
        "n": float(len(responses)),
        "reward_mean": float(scores.mean()) if len(scores) else 0.0,
        "reward_min": float(scores.min()) if len(scores) else 0.0,
        "reward_max": float(scores.max()) if len(scores) else 0.0,
    }


def purity_report(
    responses: Sequence[str], min_ratio: float = 0.8
) -> dict[str, float]:
    """Mongolian script purity: mean ratio and fraction below ``min_ratio``."""
    ratios = [mongolian_script_ratio(r) for r in responses]
    if not ratios:
        return {"purity_mean": 0.0, "frac_below": 0.0}
    below = sum(1 for r in ratios if r < min_ratio) / len(ratios)
    return {
        "purity_mean": sum(ratios) / len(ratios),
        "frac_below": below,
    }


def format_compliance_rate(responses: Sequence[str]) -> float:
    """Fraction of responses with a single well-formed think block + answer."""
    if not responses:
        return 0.0
    return sum(format_reward(r) for r in responses) / len(responses)


__all__ = [
    "decode_cache_consistency",
    "format_compliance_rate",
    "purity_report",
    "reward_report",
]
