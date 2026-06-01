# -*- coding: utf-8 -*-

"""Group Relative Policy Optimization (GRPO) for the RDT model.

GRPO (Shao et al., 2024) replaces PPO's value model with a *group-relative*
baseline: for each prompt, sample a group of responses, score them with a
(here verifiable) reward, and normalize rewards within the group to form
advantages. The token-level surrogate is PPO-style with ratio clipping plus a
k3 KL penalty toward a frozen reference.

Rigor rules enforced here:
- Advantages are normalized **within each prompt group** (zero mean / unit std).
- The loss is reduced over **completion tokens only** (token-aligned mask); the
  prompt and any externally injected ``<tool_result>`` spans never contribute,
  matching the SFT masking contract.
- Policy, ``old`` (sampling), and reference log-probs must all be produced at
  the same latent depth (``recurrent_steps``) so ratios and KL are coherent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from Model.model import RDTForCausalLM
from Model.posttrain.logprobs import token_logprobs_with_mask


def group_normalized_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize rewards within each prompt group to zero mean / unit std.

    Args:
        rewards: ``[N]`` where ``N = num_prompts * group_size``, group-contiguous.
        group_size: responses sampled per prompt.

    Returns:
        ``[N]`` advantages.
    """
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    groups = rewards.view(-1, group_size)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, keepdim=True)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1)


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    recurrent_steps: int | None = None
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float | None = None


def grpo_loss(
    policy_token_logp: torch.Tensor,
    old_token_logp: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_token_logp: torch.Tensor | None = None,
    cfg: GRPOConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level GRPO surrogate loss.

    Args:
        policy_token_logp: ``[B, L]`` current-policy per-token log-probs.
        old_token_logp: ``[B, L]`` sampling-policy log-probs (no grad).
        advantages: ``[B]`` per-sequence advantages (broadcast over tokens).
        completion_mask: ``[B, L]`` 1 on response tokens to optimize.
        ref_token_logp: ``[B, L]`` frozen reference log-probs for the KL term.

    Returns:
        ``(loss, metrics)``.
    """
    cfg = cfg or GRPOConfig()
    mask = completion_mask.to(policy_token_logp.dtype)

    ratio = torch.exp(policy_token_logp - old_token_logp)
    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pg = -torch.min(surr1, surr2)

    if ref_token_logp is not None and cfg.kl_coef:
        # k3 (unbiased, non-negative) KL estimator.
        diff = ref_token_logp - policy_token_logp
        kl = torch.exp(diff) - diff - 1.0
    else:
        kl = torch.zeros_like(pg)

    per_token = pg + cfg.kl_coef * kl
    denom = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / denom

    metrics = {
        "loss": float(loss.detach()),
        "pg": float((pg * mask).sum().detach() / denom),
        "kl": float((kl * mask).sum().detach() / denom),
        "ratio_mean": float((ratio * mask).sum().detach() / denom),
        "adv_mean": float(advantages.mean().detach()),
    }
    return loss, metrics


def sample_group(
    model: RDTForCausalLM,
    prompt_ids: torch.Tensor,
    cfg: GRPOConfig,
    eos_id: int | None = None,
    pad_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``cfg.group_size`` continuations for a single prompt.

    Args:
        prompt_ids: ``[P]`` 1-D prompt token ids.

    Returns:
        ``(sequences[G, P+n], completion_mask[G, P+n])`` where the mask is 1 on
        sampled tokens (everything after the prompt that is not padding).
    """
    if prompt_ids.dim() != 1:
        raise ValueError("prompt_ids must be 1-D")
    p = prompt_ids.shape[0]
    batch = prompt_ids.unsqueeze(0).expand(cfg.group_size, -1).contiguous()
    seqs = model.generate(
        batch,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_id=eos_id,
        pad_id=pad_id,
        recurrent_steps=cfg.recurrent_steps,
    )
    mask = torch.zeros_like(seqs, dtype=torch.float32)
    mask[:, p:] = 1.0
    if pad_id is not None:
        mask[:, p:] = (seqs[:, p:] != pad_id).float()
    return seqs, mask


def grpo_step(
    policy: RDTForCausalLM,
    reference: RDTForCausalLM | None,
    prompts: Sequence[torch.Tensor],
    reward_fn: Callable[[Sequence[str], int], torch.Tensor],
    decode: Callable[[torch.Tensor], str],
    optimizer: torch.optim.Optimizer,
    cfg: GRPOConfig | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
) -> dict[str, float]:
    """One GRPO update over a batch of prompts.

    For each prompt: sample a group, snapshot the sampling log-probs (``old``),
    score the group with ``reward_fn`` (decoded via ``decode``), normalize
    advantages within the group, then take a clipped policy-gradient step with a
    KL penalty toward ``reference`` (frozen). Reverse loss must be disabled on
    the policy during alignment.

    ``reward_fn(responses, prompt_index) -> [group_size]`` keeps reward logic
    (references, verifiers) outside the optimizer.
    """
    cfg = cfg or GRPOConfig()
    policy.reverse_loss_enabled = False

    losses = []
    agg: dict[str, float] = {}

    for idx, prompt in enumerate(prompts):
        seqs, mask = sample_group(policy, prompt, cfg, eos_id=eos_id, pad_id=pad_id)
        responses = [decode(seqs[g, prompt.shape[0]:]) for g in range(seqs.shape[0])]
        rewards = reward_fn(responses, idx).float()
        adv = group_normalized_advantages(rewards, cfg.group_size)

        with torch.no_grad():
            old_logp, _ = token_logprobs_with_mask(
                policy, seqs, mask, recurrent_steps=cfg.recurrent_steps
            )
            if reference is not None:
                ref_logp, _ = token_logprobs_with_mask(
                    reference, seqs, mask, recurrent_steps=cfg.recurrent_steps
                )
            else:
                ref_logp = None
        policy_logp, shifted_mask = token_logprobs_with_mask(
            policy, seqs, mask, recurrent_steps=cfg.recurrent_steps
        )

        loss, metrics = grpo_loss(
            policy_logp, old_logp, adv, shifted_mask,
            ref_token_logp=ref_logp, cfg=cfg,
        )
        losses.append(loss)
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v

    total = torch.stack(losses).mean()
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    n = len(prompts)
    out = {k: v / n for k, v in agg.items()}
    out["loss"] = float(total.detach())
    return out


__all__ = [
    "GRPOConfig",
    "grpo_loss",
    "grpo_step",
    "group_normalized_advantages",
    "sample_group",
]
