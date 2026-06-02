# -*- coding: utf-8 -*-

"""Optimizer + LR schedule builders.

Optimizers:
* ``adamw`` (default) — torch AdamW, optionally with the **Adam-atan2** update
  (arXiv:2407.05872): the eps-guarded ratio ``m / (sqrt(v) + eps)`` is replaced
  by ``atan2(m, b*sqrt(v))``, which is scale-invariant and immune to bf16
  underflow in the denominator. Drop-in; ``adam_eps`` becomes unused.
* ``muon`` — **experimental**, Moonlight/Muon (arXiv:2502.16982): orthogonalize
  the momentum of 2-D weight matrices via a Newton–Schulz iteration; everything
  else (embeddings, lm_head, norms, biases) stays on AdamW. Caveat for this
  model: shared recurrent weights receive ~r× amplified gradients, so Muon's
  interaction with the RDT loop is an open question — opt-in only.

Schedules: ``cosine`` (default) and ``wsd`` (warmup-stable-decay,
MiniCPM arXiv:2404.06395).
"""

from __future__ import annotations

import math
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


def _split_params_by_decay(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(decay, no_decay)`` parameter lists using the repo policy."""

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

    return decay, no_decay


def param_groups_with_no_decay(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups.

    Norms, biases, embeddings, and tensors marked with ``_no_weight_decay``
    (e.g. mamba ``dt_bias``, ``A_log``, ``D``) skip weight decay.
    """

    decay, no_decay = _split_params_by_decay(model)

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

    name = cfg.optimizer.lower()

    if name == "adamw":
        return _build_adamw(groups, cfg)

    if name == "muon":
        return _build_muon(model, cfg)

    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def _build_adamw(groups: list[dict], cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.adam_use_atan2:
        return AdamAtan2(
            groups,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def _build_muon(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Hybrid Muon: 2-D hidden weights on Muon, the rest on AdamW(/atan2).

    The routing preserves :func:`param_groups_with_no_decay`: tensors that would
    skip weight decay stay on AdamW with ``weight_decay=0``, and non-2-D tensors
    that should decay stay on AdamW with the configured decay. Muon only gets
    2-D decay-eligible hidden matrices; embeddings and output heads stay on the
    adaptive optimizer.
    """

    muon_params: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []

    head_ids: set[int] = set()
    for attr in ("lm_head", "reverse_head"):
        head = getattr(model, attr, None)
        if head is not None:
            head_ids |= {id(p) for p in head.parameters(recurse=False)}
    embed_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }

    decay, no_decay = _split_params_by_decay(model)
    for param in decay:
        if param.ndim == 2 and id(param) not in embed_ids and id(param) not in head_ids:
            muon_params.append(param)
        else:
            adam_decay.append(param)
    adam_no_decay.extend(no_decay)

    sub: list[torch.optim.Optimizer] = []
    if muon_params:
        sub.append(
            Muon(
                muon_params,
                lr=cfg.learning_rate,
                momentum=cfg.muon_momentum,
                weight_decay=cfg.weight_decay,
                ns_steps=cfg.muon_ns_steps,
            )
        )
    adam_groups: list[dict] = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": cfg.weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    if adam_groups:
        sub.append(_build_adamw(adam_groups, cfg))

    if not sub:
        raise ValueError("model has no trainable parameters")
    return CombinedOptimizer(sub)


class AdamAtan2(torch.optim.Optimizer):
    """AdamW with the atan2 update (arXiv:2407.05872), decoupled weight decay.

    ``a``/``b`` are the paper's shape constants; ``a=1.27`` keeps the update
    magnitude close to Adam in the small-gradient regime while ``atan2`` removes
    the eps hyper-parameter and the bf16 underflow failure mode.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        a: float = 1.27,
        b: float = 1.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            a, b = group["a"], group["b"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to(dtype=torch.float32)
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m_hat = m / bc1
                denom = (v / bc2).sqrt_()

                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = torch.atan2(m_hat, b * denom).to(dtype=p.dtype)
                p.add_(update, alpha=-lr * a)

        return loss


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton–Schulz orthogonalization (Keller Jordan / Moonlight)."""

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(dtype=torch.float32)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.t()
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.t()
    return x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weight matrices (experimental — see module docstring)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2-D weight matrices")
                grad = p.grad.to(dtype=torch.float32)
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=torch.float32
                    )
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                ortho = zeropower_via_newtonschulz5(buf, ns_steps).to(dtype=p.dtype)
                # RMS-matching scale so the effective step size is comparable
                # across differently-shaped matrices (Moonlight).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    """Drives several optimizers as one (shared ``param_groups`` view).

    Exposes the concatenation of the sub-optimizers' ``param_groups`` (the same
    dict objects), so ``LambdaLR`` scales every group and checkpoint
    save/restore round-trips through ``state_dict``.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        self.optimizers = optimizers
        groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(groups, {})
        self._sync_param_groups()
        self._sync_state()

    def _sync_param_groups(self) -> None:
        self.param_groups = [
            group for opt in self.optimizers for group in opt.param_groups
        ]

    def _sync_state(self) -> None:
        self.state.clear()
        for opt in self.optimizers:
            self.state.update(opt.state)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        self._sync_state()
        return loss

    def state_dict(self):
        self._sync_state()
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sub in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sub)
        self._sync_param_groups()
        self._sync_state()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR schedule: warmup then ``cosine`` or ``wsd`` decay to min_lr_ratio."""

    warmup = max(0, cfg.warmup_steps)
    decay_total = max(1, cfg.lr_decay_steps or cfg.max_steps)
    min_ratio = cfg.min_lr_ratio

    if cfg.lr_schedule == "wsd":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _wsd_lambda(warmup, decay_total, min_ratio, cfg)
        )

    def cosine_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, decay_total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda)


def _wsd_lambda(warmup: int, decay_total: int, min_ratio: float, cfg: TrainingConfig):
    """Warmup → stable plateau (lr×1) → short decay tail to min_ratio."""

    span = max(1, decay_total - warmup)
    decay_start = warmup + int(round(span * cfg.wsd_stable_ratio))
    decay_span = max(1, decay_total - decay_start)
    shape = cfg.wsd_decay_shape

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if step < decay_start:
            return 1.0
        progress = min(1.0, max(0.0, (step - decay_start) / decay_span))
        if shape == "linear":
            factor = 1.0 - progress
        elif shape == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # "1-sqrt" (MiniCPM)
            factor = 1.0 - math.sqrt(progress)
        return min_ratio + (1.0 - min_ratio) * factor

    return lr_lambda


def recurrent_steps_for_step(
    step: int,
    cfg: TrainingConfig,
    target_steps: int,
) -> int:
    """Optional recurrent-depth curriculum: ramp from start → target."""

    start = cfg.recurrent_steps_start
    if start is None or cfg.recurrent_steps_ramp <= 0 or start >= target_steps:
        return target_steps
    progress = min(1.0, max(0.0, step / cfg.recurrent_steps_ramp))
    return int(round(start + (target_steps - start) * progress))


def _unused_iterable(_: Iterable[nn.Parameter]) -> None:  # pragma: no cover
    return None


__all__ = [
    "AdamAtan2",
    "CombinedOptimizer",
    "Muon",
    "build_optimizer",
    "build_scheduler",
    "param_groups_with_no_decay",
    "recurrent_steps_for_step",
    "zeropower_via_newtonschulz5",
]
