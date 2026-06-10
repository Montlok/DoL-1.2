# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from Model.blocks import RecurrentBlock
from Model.layers.rmsnorm import RMSNorm


class RecurrentCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.block = RecurrentBlock(cfg)
        self.inject = cfg.inject_embedding
        self.inject_scale = cfg.inject_scale
        self.use_act = cfg.use_act
        self.grad_ckpt = bool(getattr(cfg, "grad_ckpt_recurrent", False))

        if self.inject_scale < 0:
            raise ValueError("inject_scale must be non-negative")
        if cfg.recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive")

        if self.use_act:
            if cfg.act_max_steps <= 0:
                raise ValueError("act_max_steps must be positive")
            if not (0.0 < cfg.act_threshold <= 1.0):
                raise ValueError("act_threshold must be in (0, 1]")
            self.halt_proj = nn.Linear(cfg.d_model, 1)
            # Normalize before the halt head: the hidden-state norm grows
            # across recurrent steps, which would otherwise saturate the
            # sigmoid and make halting depth-dependent at init.
            self.halt_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)

    def forward(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        steps: int | None = None,
        bptt_window: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self._check_inputs(e0, word_pos, morph_depth, attn_mask)

        if self.use_act:
            return self._forward_act(
                e0=e0,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
            )

        return self._forward_fixed(
            e0=e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
            steps=steps,
            bptt_window=bptt_window,
        )

    def _forward_fixed(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        steps: int | None,
        bptt_window: int | None,
    ) -> tuple[torch.Tensor, dict]:
        total_steps = int(steps if steps is not None else self.cfg.recurrent_steps)

        if total_steps <= 0:
            raise ValueError("steps must be positive")

        if bptt_window is not None:
            if bptt_window <= 0:
                raise ValueError("bptt_window must be positive")
            bptt_window = min(bptt_window, total_steps)

        h = e0

        for idx in range(total_steps):
            if bptt_window is not None and idx < total_steps - bptt_window:
                h = h.detach()

            if self.inject:
                h = h + self.inject_scale * e0

            h = self._run_block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
            )

        return h, {
            "steps_used": total_steps,
            "ponder_cost": e0.new_tensor(0.0),
        }

    def _run_block(
        self,
        h: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if self.grad_ckpt and self.training and h.requires_grad:
            def _fn(h_in):
                return self.block(
                    h_in,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, h, use_reentrant=False)
        return self.block(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

    def _forward_act(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> tuple[torch.Tensor, dict]:
        bsz, seq_len, _ = e0.shape
        dtype = e0.dtype

        h = e0
        output = torch.zeros_like(e0)

        halt_accum = torch.zeros(bsz, seq_len, device=e0.device, dtype=torch.float32)
        updates = torch.zeros(bsz, seq_len, device=e0.device, dtype=torch.float32)
        running = torch.ones(bsz, seq_len, device=e0.device, dtype=torch.float32)

        if attn_mask is not None:
            running = running * attn_mask.to(device=e0.device, dtype=torch.float32)

        for _idx in range(self.cfg.act_max_steps):
            if self.inject:
                h = h + self.inject_scale * e0

            h = self._run_block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
            )

            p = torch.sigmoid(self.halt_proj(self.halt_norm(h))).squeeze(-1).float()
            p = p * running

            new_halt = halt_accum + p
            reached = ((new_halt >= self.cfg.act_threshold).float()) * running

            remainder = (1.0 - halt_accum).clamp(min=0.0) * reached
            active_weight = p * (1.0 - reached)
            weight = active_weight + remainder

            output = output + weight.to(dtype).unsqueeze(-1) * h
            updates = updates + running

            halt_accum = new_halt
            running = running * (1.0 - reached)

        # NOTE: previously this used `running.sum().item() == 0` to break
        # early, but that forces a host sync every step. We unroll the full
        # act_max_steps and just account for never-halted positions below.

        # Leftover probability mass for positions that never crossed the
        # threshold: commit current h with the remaining `running` weight.
        output = output + running.to(dtype).unsqueeze(-1) * h

        denom = (
            attn_mask.to(device=e0.device, dtype=torch.float32).sum()
            if attn_mask is not None
            else torch.tensor(bsz * seq_len, device=e0.device, dtype=torch.float32)
        ).clamp(min=1.0)

        ponder_cost = updates.sum() / denom
        steps_used = updates.sum().detach() / denom.detach()

        return output, {
            "steps_used": float(steps_used),
            "ponder_cost": ponder_cost,
        }

    def _check_inputs(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        if e0.ndim != 3:
            raise ValueError("e0 must have shape [B, L, d_model]")

        bsz, seq_len, dim = e0.shape

        if dim != self.cfg.d_model:
            raise ValueError(f"expected d_model={self.cfg.d_model}, got {dim}")

        if word_pos is not None and word_pos.shape != (bsz, seq_len):
            raise ValueError("word_pos must have shape [B, L]")

        if morph_depth is not None and morph_depth.shape != (bsz, seq_len):
            raise ValueError("morph_depth must have shape [B, L]")

        if attn_mask is not None and attn_mask.shape != (bsz, seq_len):
            raise ValueError("attn_mask must have shape [B, L]")


def _grad_norm(x: torch.Tensor) -> float:
    if x.grad is None:
        return 0.0
    return x.grad.norm().item()


def _check() -> None:
    from Model.config import RDTConfig, tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    core = RecurrentCore(cfg)
    core.eval()

    bsz, seq_len = 2, 16
    e0 = torch.randn(bsz, seq_len, cfg.d_model)
    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)

    h, info = core(e0, word_pos=word_pos, morph_depth=morph_depth)

    print("RecurrentCore")
    print(f"  fixed_steps: {cfg.recurrent_steps}")
    print(f"  shape: {tuple(e0.shape)} -> {tuple(h.shape)}")
    print(f"  steps_used: {info['steps_used']}")

    e0g = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    h, _info = core(e0g, word_pos=word_pos, morph_depth=morph_depth)
    h.sum().backward()

    print(f"  grad_norm: {_grad_norm(e0g):.6f}")

    for steps in [2, 4, 8, 16]:
        with torch.no_grad():
            h, _info = core(
                e0,
                word_pos=word_pos,
                morph_depth=morph_depth,
                steps=steps,
            )
        print(f"  steps={steps}, out_norm={h.norm().item():.6f}")

    e0b = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    h, _info = core(
        e0b,
        word_pos=word_pos,
        morph_depth=morph_depth,
        bptt_window=2,
    )
    h.sum().backward()

    print(f"  bptt_window=2, grad_norm={_grad_norm(e0b):.6f}")

    cfg_act = RDTConfig(
        d_model=512,
        n_heads=8,
        head_dim=64,
        kv_lora_rank=128,
        ffn_hidden=1536,
        n_prelude=2,
        n_coda=2,
        mamba_per_block=5,
        attn_per_block=1,
        recurrent_steps=4,
        max_seq_len=2048,
        use_official_mamba=False,
        use_act=True,
        act_max_steps=16,
    )

    core_act = RecurrentCore(cfg_act)
    core_act.eval()

    h, info = core_act(e0, word_pos=word_pos, morph_depth=morph_depth)

    print("ACT")
    print(f"  steps_used: {info['steps_used']:.6f}")
    print(f"  ponder_cost: {float(info['ponder_cost'].detach()):.6f}")

    e0a = torch.randn(bsz, seq_len, cfg_act.d_model, requires_grad=True)
    h, info = core_act(e0a, word_pos=word_pos, morph_depth=morph_depth)
    (h.sum() + cfg_act.act_ponder_cost * info["ponder_cost"]).backward()

    print(f"  act_grad_norm: {_grad_norm(e0a):.6f}")


if __name__ == "__main__":
    _check()
