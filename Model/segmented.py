# -*- coding: utf-8 -*-

"""Segmented (Block-Transformer-style) recurrent core for the Daffodils RDT.

This core realises the user's target architecture: slice the context into
fixed-length blocks; encode every token with a **causal (forward-only) Mamba**;
take the causal hidden state at each block boundary as that block's *summary*;
run **block-causal attention + shared-weight RDT recurrent depth** over the
``n_seg`` summaries; finally scatter the *previous* block's refined summary back
to token resolution and decode the next block with a small **causal local
decoder**. Because ``n_seg << n_tok`` the (quadratic) attention / RDT cost drops
sharply relative to running it on every token.

Causality (zero future leakage), end-to-end:

* The block summary for block ``s`` is the causal Mamba state at token index
  ``min((s+1)*L_B - 1, L-1)`` — it only ever saw tokens ``<= boundary``.
* Block-causal attention lets summary ``i`` attend to summaries ``<= i`` only.
* Token ``t`` in block ``s`` is decoded from the **refined summary of block
  ``s-1``** (which depends only on tokens ``< s*L_B <= t``) plus a causal local
  decoder over the token embeddings ``e0[<= t]``. Block ``0`` uses a learned
  start context. A block summary therefore only ever conditions the *next*
  block, never its own tokens.

Literature: Block Transformer (arXiv:2406.02657), Block-State Transformer
(2306.09539), NSA (2502.11089) for the block-summary attention; Huginn
(2502.05171) for shared-weight RDT with random-r + truncated BPTT; causal SSM
basis Mamba (2312.00752). ``SegmentedCore`` is a drop-in replacement for
``RecurrentCore`` / ``TwoStageCore``: identical forward signature and return
contract (equal-length ``hidden`` + ``info`` dict).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from Model.blocks import AttnSubLayer, MambaSubLayer
from Model.layers.rmsnorm import RMSNorm
from Model.two_stage import MHCAttnSubLayer


class SegmentedCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.use_act:
            raise ValueError("SegmentedCore does not support use_act=True")
        if cfg.recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive")
        if cfg.segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if cfg.segmented_local_layers <= 0:
            raise ValueError("segmented_local_layers must be positive")

        self.segment_len = int(cfg.segment_len)
        self.drift_mode = cfg.recurrent_drift_mode
        self.n_streams = cfg.mhc_n_streams
        self.inject_scale = cfg.inject_scale
        self.inject_decay = cfg.recurrent_inject_decay
        self.random_r = bool(cfg.recurrent_random_r)
        self.r_min = int(cfg.recurrent_r_min)
        self.r_max = int(cfg.recurrent_r_max)

        self.grad_ckpt_stage1 = bool(getattr(cfg, "grad_ckpt_blocks", False))
        self.grad_ckpt_stage2 = bool(getattr(cfg, "grad_ckpt_recurrent", False))
        self.grad_ckpt_local = bool(getattr(cfg, "grad_ckpt_blocks", False))

        # Stage 1: causal Mamba token encoder (forward-only).
        self.stage1 = nn.ModuleList(
            MambaSubLayer(cfg, layer_idx=i) for i in range(cfg.stage1_mamba_layers)
        )

        # Stage 2: block-level refinement layers (shared across RDT steps).
        if self.drift_mode == "mhc":
            self.stage2 = nn.ModuleList(
                MHCAttnSubLayer(cfg, layer_idx=i)
                for i in range(cfg.stage2_attn_layers)
            )
        else:
            self.stage2 = nn.ModuleList(
                AttnSubLayer(cfg, layer_idx=i) for i in range(cfg.stage2_attn_layers)
            )

        if self.drift_mode in {"norm", "both"}:
            self.boundary_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        else:
            self.boundary_norm = None

        # Stage 3: local causal token decoder + previous-block context plumbing.
        self.ctx_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.start_ctx = nn.Parameter(torch.zeros(cfg.d_model))
        self.local = nn.ModuleList(
            MambaSubLayer(cfg, layer_idx=i)
            for i in range(cfg.segmented_local_layers)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        steps: int | None = None,
        bptt_window: int | None = None,
        cache=None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        self._check_inputs(e0, word_pos, morph_depth, attn_mask)

        if cache is not None:
            return self._forward_cached(
                e0,
                word_pos=word_pos,
                morph_depth=morph_depth,
                total_steps=self._resolve_steps(steps),
                cache=cache,
                pos_offset=pos_offset,
            )

        total_steps = self._resolve_steps(steps)

        # --- Stage 1: causal token encoding -------------------------------
        backbone = self._run_stage1(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

        bsz, seq_len, _ = e0.shape
        boundary_idx = self._boundary_indices(seq_len, e0.device)  # [n_seg]
        n_seg = boundary_idx.shape[0]

        # Block summaries = causal state at each block boundary (zero leakage).
        summaries = backbone.index_select(1, boundary_idx)  # [B, n_seg, d]

        # --- Stage 2: block-causal refinement (RDT) -----------------------
        seg_word_pos = torch.arange(n_seg, device=e0.device).unsqueeze(0).expand(
            bsz, n_seg
        )
        seg_morph = torch.zeros(bsz, n_seg, dtype=torch.long, device=e0.device)

        refined = self._refine(
            summaries,
            word_pos=seg_word_pos,
            morph_depth=seg_morph,
            total_steps=total_steps,
            bptt_window=bptt_window,
        )  # [B, n_seg, d]

        # --- Stage 3: scatter previous block's context, local causal decode
        ctx_tokens = self._scatter_prev_context(refined, seq_len)  # [B, L, d]
        h = e0 + self.ctx_proj(ctx_tokens)
        h = self._run_local(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

        info = {
            "steps_used": total_steps,
            "ponder_cost": e0.new_tensor(0.0),
            "global_semantic": self._global_semantic(backbone, attn_mask),
            "n_segments": n_seg,
        }
        return h, info

    # ------------------------------------------------------------------
    # Incremental decode (full-forward equivalent when kv_share_budget=0)
    # ------------------------------------------------------------------
    def _forward_cached(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        total_steps: int,
        cache,
        pos_offset: int,
    ) -> tuple[torch.Tensor, dict]:
        """Process the new chunk ``e0`` (``[B, m, d]``) one block-causally.

        Stage-1 Mamba and the local decoder carry constant-size recurrent state;
        the block-level RDT keeps a causal MLA cache per ``(step, layer)``.
        A block summary is
        finalized only when its block completes, and a token only ever reads its
        *previous* block's refined summary, so the result is identical to a fresh
        full forward over the growing prefix (zero future leakage preserved).
        """

        if int(self.cfg.kv_share_budget) > 0:
            raise NotImplementedError(
                "segmented cached decode does not support kv_share_budget > 0. "
                "A shared append-only MLA cache corrupts recurrent-step history; "
                "keep kv_share_budget=0 for cache-equivalent decode."
            )

        bsz, m, dim = e0.shape
        device = e0.device
        Lb = self.segment_len

        backbone = e0
        for i, layer in enumerate(self.stage1):
            backbone = layer(
                backbone,
                attn_mask=None,
                cache=cache.mamba_cache(f"seg.stage1.{i}"),
            )

        abs_pos = pos_offset + torch.arange(m, device=device)
        is_boundary = ((abs_pos + 1) % Lb) == 0
        boundary_local = torch.nonzero(is_boundary, as_tuple=False).flatten()

        prior = getattr(cache, "seg_refined", None)
        prior_segs = 0 if prior is None else prior.shape[1]

        if boundary_local.numel() > 0:
            new_summaries = backbone.index_select(1, boundary_local)  # [B, k, d]
            refined_new = self._refine_cached(
                new_summaries, total_steps, cache, seg_offset=prior_segs
            )
            if prior is None:
                cache.seg_refined = refined_new
            else:
                cache.seg_refined = torch.cat([prior, refined_new], dim=1)

        # Per-token previous-block context (refined[block-1] / start_ctx).
        block_of = torch.div(abs_pos, Lb, rounding_mode="floor")
        prev_block = block_of - 1
        start = self.start_ctx.to(dtype=e0.dtype).view(1, 1, dim).expand(bsz, m, dim)
        seg_refined = getattr(cache, "seg_refined", None)
        if seg_refined is not None:
            idx = prev_block.clamp(min=0).view(1, m, 1).expand(bsz, m, dim)
            gathered = torch.gather(seg_refined, 1, idx)
        else:
            gathered = start
        use_prev = (prev_block >= 0).view(1, m, 1)
        ctx = torch.where(use_prev, gathered, start)

        h = e0 + self.ctx_proj(ctx)
        for i, layer in enumerate(self.local):
            h = layer(
                h,
                attn_mask=None,
                cache=cache.mamba_cache(f"seg.local.{i}"),
            )

        info = {
            "steps_used": total_steps,
            "ponder_cost": e0.new_tensor(0.0),
            "global_semantic": backbone.mean(dim=1),
        }
        return h, info

    def _refine_cached(self, summaries, total_steps, cache, seg_offset):
        bsz, k, _ = summaries.shape
        device = summaries.device
        seg_wp = torch.arange(
            seg_offset, seg_offset + k, device=device
        ).unsqueeze(0).expand(bsz, k)
        seg_md = torch.zeros(bsz, k, dtype=torch.long, device=device)

        def key(step: int, li: int) -> str:
            return f"seg.rdt.s{step}.l{li}"

        if self.drift_mode == "mhc":
            streams = summaries.unsqueeze(-2).expand(
                -1, -1, self.n_streams, -1
            ).contiguous()
            for step in range(total_steps):
                for li, layer in enumerate(self.stage2):
                    streams = layer(
                        streams,
                        word_pos=seg_wp,
                        morph_depth=seg_md,
                        attn_mask=None,
                        causal=True,
                        cache=cache.mla_cache(key(step, li)),
                        pos_offset=seg_offset,
                    )
            return streams.mean(dim=-2)

        inject = self.drift_mode in {"decay", "both"}
        h = summaries
        for step in range(total_steps):
            if inject:
                scale = self.inject_scale * (self.inject_decay ** step)
                h = h + scale * summaries
            for li, layer in enumerate(self.stage2):
                h = layer(
                    h,
                    word_pos=seg_wp,
                    morph_depth=seg_md,
                    attn_mask=None,
                    causal=True,
                    cache=cache.mla_cache(key(step, li)),
                    pos_offset=seg_offset,
                )
            if self.boundary_norm is not None:
                h = self.boundary_norm(h)
        return h

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def _run_stage1(self, x, word_pos, morph_depth, attn_mask, causal):
        for layer in self.stage1:
            x = self._maybe_ckpt(
                layer,
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                enabled=self.grad_ckpt_stage1,
            )
        return x

    def _run_local(self, x, word_pos, morph_depth, attn_mask, causal):
        for layer in self.local:
            x = self._maybe_ckpt(
                layer,
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
                enabled=self.grad_ckpt_local,
            )
        return x

    def _refine(self, summaries, word_pos, morph_depth, total_steps, bptt_window):
        if bptt_window is not None:
            if bptt_window <= 0:
                raise ValueError("bptt_window must be positive")
            bptt_window = min(bptt_window, total_steps)

        if self.drift_mode == "mhc":
            streams = summaries.unsqueeze(-2).expand(
                -1, -1, self.n_streams, -1
            ).contiguous()
            for step in range(total_steps):
                if bptt_window is not None and step < total_steps - bptt_window:
                    streams = streams.detach()
                for layer in self.stage2:
                    streams = self._maybe_ckpt(
                        layer,
                        streams,
                        word_pos=word_pos,
                        morph_depth=morph_depth,
                        attn_mask=None,
                        causal=True,
                        enabled=self.grad_ckpt_stage2,
                    )
            return streams.mean(dim=-2)

        inject = self.drift_mode in {"decay", "both"}
        h = summaries
        for step in range(total_steps):
            if bptt_window is not None and step < total_steps - bptt_window:
                h = h.detach()
            if inject:
                scale = self.inject_scale * (self.inject_decay ** step)
                h = h + scale * summaries
            for layer in self.stage2:
                h = self._maybe_ckpt(
                    layer,
                    h,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=None,
                    causal=True,
                    enabled=self.grad_ckpt_stage2,
                )
            if self.boundary_norm is not None:
                h = self.boundary_norm(h)
        return h

    def _scatter_prev_context(self, refined: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Map each token to its *previous* block's refined summary.

        Token at position ``t`` (block ``s = t // L_B``) receives ``refined[s-1]``;
        block ``0`` receives the learned ``start_ctx``. ``refined[s-1]`` depends
        only on tokens ``< s*L_B <= t`` so this is strictly causal.
        """

        bsz, n_seg, dim = refined.shape
        device = refined.device

        seg_of_token = (
            torch.arange(seq_len, device=device) // self.segment_len
        )  # [L], in [0, n_seg-1]
        prev_seg = seg_of_token - 1  # -1 for block 0

        start = self.start_ctx.to(dtype=refined.dtype).view(1, 1, dim).expand(
            bsz, seq_len, dim
        )
        gather_idx = prev_seg.clamp(min=0).view(1, seq_len, 1).expand(bsz, seq_len, dim)
        gathered = torch.gather(refined, 1, gather_idx)  # [B, L, d]

        use_prev = (prev_seg >= 0).view(1, seq_len, 1)
        return torch.where(use_prev, gathered, start)

    def _boundary_indices(self, seq_len: int, device) -> torch.Tensor:
        if seq_len <= self.segment_len:
            return torch.tensor([seq_len - 1], device=device)
        last = torch.arange(
            self.segment_len - 1, seq_len, self.segment_len, device=device
        )
        if last.numel() == 0 or last[-1].item() != seq_len - 1:
            last = torch.cat(
                [last, torch.tensor([seq_len - 1], device=device)]
            )
        return last

    def _resolve_steps(self, steps: int | None) -> int:
        if steps is not None:
            total = int(steps)
        elif self.random_r and self.training:
            total = int(torch.randint(self.r_min, self.r_max + 1, (1,)).item())
        else:
            total = int(self.cfg.recurrent_steps)
        if total <= 0:
            raise ValueError("steps must be positive")
        return total

    @staticmethod
    def _global_semantic(backbone, attn_mask):
        if attn_mask is None:
            return backbone.mean(dim=1)
        mask = attn_mask.to(dtype=backbone.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (backbone * mask).sum(dim=1) / denom

    def _maybe_ckpt(self, layer, x, word_pos, morph_depth, attn_mask, causal, enabled):
        if enabled and self.training and x.requires_grad:
            def _fn(x_in):
                return layer(
                    x_in,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, x, use_reentrant=False)

        return layer(
            x,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

    def _check_inputs(self, e0, word_pos, morph_depth, attn_mask) -> None:
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
