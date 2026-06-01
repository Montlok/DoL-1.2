# -*- coding: utf-8 -*-

"""Verifiable / rule-based rewards for GRPO.

Mongolian + STEM alignment is a great fit for *verifiable* rewards: no learned
reward model is needed, which removes a whole class of reward-hacking and
distribution-shift failures. Rewards here are pure functions of the decoded
response text (and an optional reference answer):

- ``exact_match_reward``: STEM answer correctness (normalized string / numeric).
- ``mongolian_script_ratio`` / ``language_purity_reward``: fraction of script
  that is Mongolian (Cyrillic + traditional Mongolian block), penalizing
  code-switching / script leakage.
- ``format_reward``: well-formed ``<think>...</think>`` then a final answer.

``RewardConfig`` linearly combines components; ``compute_rewards`` returns a
per-response tensor ready for group normalization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Cyrillic (Mongolian uses Cyrillic) + traditional Mongolian script block.
_CYRILLIC = (0x0400, 0x04FF)
_MONGOLIAN = (0x1800, 0x18AF)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _in_range(ch: str, rng: tuple[int, int]) -> bool:
    return rng[0] <= ord(ch) <= rng[1]


def mongolian_script_ratio(text: str) -> float:
    """Fraction of letters that are Mongolian (Cyrillic or traditional)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    mong = sum(
        1 for c in letters if _in_range(c, _CYRILLIC) or _in_range(c, _MONGOLIAN)
    )
    return mong / len(letters)


def language_purity_reward(text: str, min_ratio: float = 0.0) -> float:
    """Reward in ``[0, 1]`` equal to the Mongolian script ratio.

    ``min_ratio`` hard-zeros responses below a purity floor (useful to strongly
    discourage script leakage).
    """
    ratio = mongolian_script_ratio(text)
    return 0.0 if ratio < min_ratio else ratio


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_reward(response: str, reference: str) -> float:
    """1.0 if normalized strings match, else 0.0."""
    return 1.0 if _normalize_answer(response) == _normalize_answer(reference) else 0.0


def numeric_match_reward(response: str, reference: str, tol: float = 1e-6) -> float:
    """1.0 if the last number in the response equals the reference number."""
    resp_nums = _NUM_RE.findall(response)
    ref_nums = _NUM_RE.findall(reference)
    if not resp_nums or not ref_nums:
        return 0.0
    return 1.0 if abs(float(resp_nums[-1]) - float(ref_nums[-1])) <= tol else 0.0


def format_reward(text: str) -> float:
    """1.0 if there is a single well-formed think block followed by content."""
    blocks = _THINK_RE.findall(text)
    if len(blocks) != 1:
        return 0.0
    after = _THINK_RE.sub("", text, count=1).strip()
    return 1.0 if after else 0.0


@dataclass
class RewardConfig:
    exact_match_weight: float = 0.0
    numeric_match_weight: float = 0.0
    purity_weight: float = 0.0
    purity_min_ratio: float = 0.0
    format_weight: float = 0.0


def reward_for(
    response: str,
    reference: str | None,
    cfg: RewardConfig,
) -> float:
    total = 0.0
    if cfg.exact_match_weight and reference is not None:
        total += cfg.exact_match_weight * exact_match_reward(response, reference)
    if cfg.numeric_match_weight and reference is not None:
        total += cfg.numeric_match_weight * numeric_match_reward(response, reference)
    if cfg.purity_weight:
        total += cfg.purity_weight * language_purity_reward(
            response, cfg.purity_min_ratio
        )
    if cfg.format_weight:
        total += cfg.format_weight * format_reward(response)
    return total


def compute_rewards(
    responses: Sequence[str],
    references: Sequence[str | None] | None,
    cfg: RewardConfig,
) -> torch.Tensor:
    """Per-response scalar rewards as a float tensor ``[N]``."""
    if references is None:
        references = [None] * len(responses)
    return torch.tensor(
        [reward_for(r, ref, cfg) for r, ref in zip(responses, references)],
        dtype=torch.float32,
    )


__all__ = [
    "RewardConfig",
    "compute_rewards",
    "exact_match_reward",
    "numeric_match_reward",
    "format_reward",
    "language_purity_reward",
    "mongolian_script_ratio",
    "reward_for",
]
