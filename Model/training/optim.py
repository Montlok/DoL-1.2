# -*- coding: utf-8 -*-

"""Optimizer + LR schedule builders."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import torch
import torch.nn as nn

from Model.config import TrainingConfig

try:  # optional: keep optim usable even if rmsnorm import fails for any reason
    from Model.layers.rmsnorm import RMSNorm as _RMSNorm
except Exception:  # pragma: no cover - defensive
    _RMSNorm = None

_NORM_TYPES: tuple[type, ...] = (
    (nn.LayerNorm, nn.GroupNorm, _RMSNorm) if _RMSNorm is not None else (nn.LayerNorm, nn.GroupNorm)
)


def param_groups_with_no_decay(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups.

    Norms, biases, embeddings, and tensors marked with ``_no_weight_decay``
    (e.g. mamba ``dt_bias``, ``A_log``, ``D``) skip weight decay.
    """

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        is_norm = isinstance(module, _NORM_TYPES)

        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)

            no_wd = getattr(param, "_no_weight_decay", False)
            if (
                no_wd
                or is_norm
                or name.endswith("bias")
                or isinstance(module, nn.Embedding)
                or param.ndim <= 1
            ):
                no_decay.append(param)
            else:
                decay.append(param)

    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    model: nn.Module,
    cfg: TrainingConfig,
) -> torch.optim.Optimizer:
    groups = param_groups_with_no_decay(model, cfg.weight_decay)
    if not groups:
        raise ValueError("model has no trainable parameters")

    if cfg.optimizer.lower() != "adamw":
        raise ValueError(f"unsupported optimizer: {cfg.optimizer}")

    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine decay to ``min_lr_ratio * lr``."""

    warmup = max(0, cfg.warmup_steps)
    decay_total = max(1, cfg.lr_decay_steps or cfg.max_steps)
    min_ratio = cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, decay_total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def recurrent_steps_for_step(
    step: int,
    cfg: TrainingConfig,
    target_steps: int,
) -> int:
    """Recurrent depth for one optimizer step.

    Applies the optional ramp curriculum (start → target), then — when
    ``cfg.recurrent_steps_sampling == "poisson"`` — replaces the fixed depth
    with a log-normal Poisson draw centered on the ramped target (Geiping et
    al. 2025, "Scaling up Test-Time Compute with Latent Reasoning"):

        tau ~ Normal(log(t - 1) - sigma^2 / 2, sigma)
        r = 1 + Poisson(exp(tau)),  clamped to [min, max]

    so ``E[r] ≈ t``. Training across depths is what makes the model usable
    at depths other than the default at inference time; fixed-depth training
    measurably degrades both shallower and deeper evaluation.

    The draw is seeded from ``(cfg.seed, step)`` only — never the rank — so
    every rank unrolls the same depth. Under FSDP each recurrent step issues
    its own all-gathers; rank-divergent depths would deadlock collectives.
    """

    target = _ramp_target(step, cfg, target_steps)

    if cfg.recurrent_steps_sampling != "poisson":
        return target

    lo = cfg.recurrent_steps_min
    hi = cfg.recurrent_steps_max
    if hi is None:
        hi = max(2 * target_steps, lo)

    sigma = cfg.recurrent_steps_sigma
    mu = max(target - 1, 1)

    rng = random.Random((cfg.seed << 32) ^ (step * 0x9E3779B97F4A7C15))
    tau = rng.gauss(math.log(mu) - 0.5 * sigma * sigma, sigma)
    r = 1 + _poisson_sample(rng, math.exp(tau))

    return max(lo, min(r, hi))


def _ramp_target(step: int, cfg: TrainingConfig, target_steps: int) -> int:
    """Optional recurrent-depth curriculum: ramp from start → target."""

    start = cfg.recurrent_steps_start
    if start is None or cfg.recurrent_steps_ramp <= 0 or start >= target_steps:
        return target_steps
    progress = min(1.0, max(0.0, step / cfg.recurrent_steps_ramp))
    return int(round(start + (target_steps - start) * progress))


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth Poisson sampler; fine for the small lambdas used here."""

    if lam <= 0.0:
        return 0
    threshold = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def _unused_iterable(_: Iterable[nn.Parameter]) -> None:  # pragma: no cover
    return None


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "param_groups_with_no_decay",
    "recurrent_steps_for_step",
]
