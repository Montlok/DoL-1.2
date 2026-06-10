# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Model.blocks import StandardBlock
from Model.cache import RDTCache
from Model.config import RDTConfig
from Model.layers.rmsnorm import RMSNorm
from Model.recurrent import RecurrentCore
from Model.vision import VisionInjector


class RDTForCausalLM(nn.Module):
    def __init__(self, cfg: RDTConfig, patch_pixels: int = 14 * 14 * 3):
        super().__init__()

        self.cfg = cfg
        self.patch_pixels = patch_pixels

        self.embed = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_id,
        )

        self.vision = VisionInjector(cfg, patch_pixels)

        self.prelude = nn.ModuleList(
            StandardBlock(cfg, layer_idx=i) for i in range(cfg.n_prelude)
        )

        self.recurrent = RecurrentCore(cfg)

        self.coda = nn.ModuleList(
            StandardBlock(cfg, layer_idx=cfg.n_prelude + i) for i in range(cfg.n_coda)
        )

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.bidirectional = cfg.bidirectional

        if self.bidirectional:
            self.reverse_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.reverse_head = None

        self.apply(self._init_weights)
        self._scale_residual_projections()

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            if self.reverse_head is not None:
                self.reverse_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        steps: int | None = None,
        bptt_window: int | None = None,
        return_logits: bool = True,
        loss_chunk_size: int | None = None,
        cache=None,
    ) -> dict[str, torch.Tensor | dict | None]:
        past_len = 0 if cache is None else cache.seq_len
        self._check_inputs(input_ids, attention_mask, labels, past_len=past_len)
        if loss_chunk_size is None:
            loss_chunk_size = self.cfg.loss_chunk_size
        elif loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")

        bsz, seq_len = input_ids.shape

        if cache is not None:
            if attention_mask is not None and not bool(attention_mask.bool().all()):
                raise ValueError(
                    "cached decoding requires an all-ones attention_mask"
                )
            attention_mask = torch.ones_like(input_ids)
            if past_len > 0 and (word_pos is None or morph_depth is None):
                raise ValueError(
                    "incremental decoding requires explicit word_pos/morph_depth "
                    "(derive on the full sequence and slice the new positions)"
                )

        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_id).long()

        if word_pos is None or morph_depth is None:
            word_pos, morph_depth = self._default_morph_info(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        h = self.embed(input_ids)

        if pixel_values is not None:
            h = self.vision(h, input_ids, pixel_values)

        for i, block in enumerate(self.prelude):
            h = self._maybe_ckpt(
                block,
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
                cache_entry=cache.attn_entry(("prelude", i)) if cache is not None else None,
            )

        e0 = h

        h, rec_info = self.recurrent(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attention_mask,
            causal=True,
            steps=steps,
            bptt_window=bptt_window,
            cache=cache,
        )

        for i, block in enumerate(self.coda):
            h = self._maybe_ckpt(
                block,
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
                cache_entry=cache.attn_entry(("coda", i)) if cache is not None else None,
            )

        if cache is not None:
            cache.advance(seq_len)

        h = self.final_norm(h)
        logits = None

        loss = None
        loss_parts: dict[str, float] = {}

        if labels is not None:
            if not return_logits and loss_chunk_size is not None:
                loss, loss_parts = self._losses_chunked(
                    h,
                    labels,
                    rec_info,
                    loss_chunk_size,
                )
            else:
                logits = self.lm_head(h)
                loss, loss_parts = self._losses(h, logits, labels, rec_info)
        elif return_logits:
            logits = self.lm_head(h)

        return {
            "loss": loss,
            "logits": logits if return_logits else None,
            "loss_parts": loss_parts,
            "rec_info": rec_info,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Incremental greedy/sampled decoding with KV + SSM state caching.

        Prompts must be un-padded (all-ones ``attention_mask`` or ``None``);
        batch with equal-length prompts or run with bsz 1. ``temperature <= 0``
        is greedy. Finished rows are filled with EOS. ``steps`` fixes the
        recurrent depth for both prefill and decode; ACT is not supported.
        """

        if self.cfg.use_act:
            raise NotImplementedError(
                "generate() supports fixed recurrent steps only (use_act=False)"
            )
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty [B, L] tensor")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError("generate requires un-padded prompts (all-ones mask)")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        bsz, prompt_len = input_ids.shape
        if prompt_len + max_new_tokens > self.cfg.max_seq_len:
            raise ValueError("prompt_len + max_new_tokens exceeds max_seq_len")

        eos = self.cfg.eos_id if eos_id is None else eos_id
        was_training = self.training
        self.eval()

        try:
            cache = RDTCache()
            ids = input_ids
            ones = torch.ones_like(ids)
            word_pos, morph_depth = self._default_morph_info(ids, ones)

            out = self.forward(
                ids,
                attention_mask=ones,
                word_pos=word_pos,
                morph_depth=morph_depth,
                pixel_values=pixel_values,
                steps=steps,
                cache=cache,
            )

            finished = torch.zeros(bsz, dtype=torch.bool, device=ids.device)

            # Incremental morph-position state (per row) so each decode step
            # is O(B) instead of re-deriving (word_pos, morph_depth) over the
            # whole growing sequence (which would make generation O(L^2)).
            wb = int(self.cfg.word_boundary_id)
            mb = int(self.cfg.morpheme_boundary_id)
            is_wb0 = ids.eq(wb)
            is_content0 = ~((ids >= 0) & (ids < 256))
            wb_cum = is_wb0.long().sum(dim=1)
            any_wb = is_wb0.any(dim=1)
            any_content = is_content0.any(dim=1)
            # word_pos = clamp(wb_cum + shift); at the last prompt position
            # the clamp is inactive, so shift is recoverable by subtraction.
            shift = word_pos[:, -1] - wb_cum
            depth_last = morph_depth[:, -1]
            ones_col = torch.ones((bsz, 1), dtype=ones.dtype, device=ids.device)

            for _ in range(max_new_tokens):
                logits = out["logits"][:, -1].float()
                next_id = self._sample_token(logits, temperature, top_k)
                if eos is not None and eos >= 0:
                    next_id = torch.where(
                        finished, torch.full_like(next_id, eos), next_id
                    )
                ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
                if eos is not None and eos >= 0:
                    finished = finished | next_id.eq(eos)
                    if bool(finished.all()):
                        break

                n_wb = next_id.eq(wb)
                n_mb = next_id.eq(mb)
                n_special = (next_id >= 0) & (next_id < 256)
                n_other = n_special & ~n_wb & ~n_mb
                # The first wb/content token fixes the per-row shift: a wb
                # before any content anchors word 0 (shift -1).
                undecided = ~any_wb & ~any_content
                shift = torch.where(undecided & n_wb, shift - 1, shift)
                any_wb = any_wb | n_wb
                any_content = any_content | ~n_special
                wb_cum = wb_cum + n_wb.long()
                wp_new = (wb_cum + shift).clamp(min=0)
                depth_last = torch.where(
                    n_wb | n_other,
                    torch.zeros_like(depth_last),
                    depth_last + n_mb.long(),
                )
                if self.cfg.max_morph_depth > 0:
                    depth_last = depth_last.clamp(max=self.cfg.max_morph_depth - 1)

                out = self.forward(
                    ids[:, -1:],
                    attention_mask=ones_col,
                    word_pos=wp_new.unsqueeze(1),
                    morph_depth=depth_last.unsqueeze(1),
                    steps=steps,
                    cache=cache,
                )
        finally:
            if was_training:
                self.train()

        return ids

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _losses(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rec_info: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward = self._causal_loss(logits, labels)
        loss = forward
        parts = {"forward": float(forward.detach())}

        if self.reverse_head is not None:
            rev_logits = self.reverse_head(h)
            reverse = self._reverse_loss(rev_logits, labels)
            loss = loss + self.cfg.reverse_loss_weight * reverse
            parts["reverse"] = float(reverse.detach())

        ponder = rec_info.get("ponder_cost")
        if self.cfg.use_act and isinstance(ponder, torch.Tensor):
            loss = loss + self.cfg.act_ponder_cost * ponder
            parts["ponder"] = float(ponder.detach())

        return loss, parts

    def _losses_chunked(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        rec_info: dict,
        chunk_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward = self._chunked_causal_loss(h, labels, self.lm_head, chunk_size)
        loss = forward
        parts = {"forward": float(forward.detach())}

        if self.reverse_head is not None:
            reverse = self._chunked_reverse_loss(
                h,
                labels,
                self.reverse_head,
                chunk_size,
            )
            loss = loss + self.cfg.reverse_loss_weight * reverse
            parts["reverse"] = float(reverse.detach())

        ponder = rec_info.get("ponder_cost")
        if self.cfg.use_act and isinstance(ponder, torch.Tensor):
            loss = loss + self.cfg.act_ponder_cost * ponder
            parts["ponder"] = float(ponder.detach())

        return loss, parts

    def _causal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._token_loss(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

    def _reverse_loss(
        self,
        rev_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self._token_loss(
            rev_logits[:, 1:].reshape(-1, rev_logits.size(-1)),
            labels[:, :-1].reshape(-1),
        )

    def _token_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.cfg.ignore_index
        if logits.numel() == 0 or not bool(valid.any()):
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.cfg.ignore_index,
        )

    def _chunked_causal_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        return self._chunked_token_loss(
            h[:, :-1].reshape(-1, h.size(-1)),
            labels[:, 1:].reshape(-1),
            head,
            chunk_size,
        )

    def _chunked_reverse_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        return self._chunked_token_loss(
            h[:, 1:].reshape(-1, h.size(-1)),
            labels[:, :-1].reshape(-1),
            head,
            chunk_size,
        )

    def _chunked_token_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        valid = targets != self.cfg.ignore_index
        if hidden.numel() == 0 or not bool(valid.any()):
            return hidden.sum() * 0.0

        hidden = hidden[valid]
        targets = targets[valid]
        loss_sum = hidden.new_zeros(())

        for start in range(0, hidden.size(0), chunk_size):
            end = min(start + chunk_size, hidden.size(0))
            logits = F.linear(hidden[start:end], head.weight, head.bias)
            loss_sum = loss_sum + F.cross_entropy(
                logits,
                targets[start:end],
                reduction="sum",
            )

        return loss_sum / targets.numel()

    def _maybe_ckpt(
        self,
        block: nn.Module,
        h: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        cache_entry=None,
    ) -> torch.Tensor:
        if (
            getattr(self.cfg, "grad_ckpt_prelude_coda", False)
            and self.training
            and h.requires_grad
            and cache_entry is None
        ):
            def _fn(x):
                return block(
                    x,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, h, use_reentrant=False)
        return block(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
            cache=cache_entry,
        )

    def _default_morph_info(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive ``(word_pos, morph_depth)`` from boundary token IDs.

        This is a vectorized counterpart of
        :func:`Tokenizer.pretraining.derive_morph_info_from_boundary_ids`.
        Callers that already supply ``word_pos`` / ``morph_depth`` skip this.

        Semantics (see tokenizer-side reference for details):

        * ``word_boundary_id`` opens a new word at depth 0.
        * ``morpheme_boundary_id`` keeps the current word and bumps depth.
        * Other special tokens (ids in ``[0, 256)``) reset depth to 0 and
          stay on the current ``word_pos``; the next non-special content
          token will inherit ``word_pos`` until a new ``word_boundary``.
        * Padding positions (``attention_mask == 0``) keep their derived
          ``word_pos`` so downstream RoPE never sees ``-1``; the mask
          itself is what zeros out padded contributions.
        """

        device = input_ids.device
        bsz, seq_len = input_ids.shape
        cfg = self.cfg

        wb = int(cfg.word_boundary_id)
        mb = int(cfg.morpheme_boundary_id)
        special_hi = 256

        is_wb = input_ids == wb
        is_mb = input_ids == mb
        is_special = (input_ids >= 0) & (input_ids < special_hi)
        is_content = ~is_special
        is_other_special = is_special & ~is_wb & ~is_mb

        # word_pos: cumulative count of word_boundary occurrences with a
        # per-row shift. If the very first word_boundary appears before
        # any content token, the first wb anchors word 0 (shift = -1).
        # Otherwise content tokens implicitly open word 0 and the first
        # wb opens word 1 (shift = 0).
        wb_cum = is_wb.long().cumsum(dim=1)

        inf = seq_len + 1
        any_wb = is_wb.any(dim=1)
        any_content = is_content.any(dim=1)

        first_wb = torch.where(
            any_wb,
            is_wb.long().argmax(dim=1),
            torch.full((bsz,), inf, device=device, dtype=torch.long),
        )
        first_content = torch.where(
            any_content,
            is_content.long().argmax(dim=1),
            torch.full((bsz,), inf, device=device, dtype=torch.long),
        )
        shift = torch.where(
            first_wb < first_content,
            torch.full((bsz,), -1, device=device, dtype=torch.long),
            torch.zeros(bsz, device=device, dtype=torch.long),
        )

        word_pos = (wb_cum + shift.unsqueeze(-1)).clamp(min=0)

        # morph_depth: cumulative morpheme_boundary count since the most
        # recent reset (word_boundary or other special). The reset
        # position itself reads depth 0.
        cum_inc = is_mb.long().cumsum(dim=1)
        reset_positions = is_wb | is_other_special
        reset_value = torch.where(
            reset_positions, cum_inc, torch.full_like(cum_inc, -1)
        )
        last_reset, _ = reset_value.cummax(dim=1)
        depth = cum_inc - last_reset.clamp(min=0)
        depth = torch.where(reset_positions, torch.zeros_like(depth), depth)

        if cfg.max_morph_depth > 0:
            depth = depth.clamp(max=cfg.max_morph_depth - 1)

        return word_pos.to(torch.long), depth.to(torch.long)


    def _check_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        past_len: int = 0,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")

        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")

        # Dtype + shape checks only; value-range checks are skipped here
        # to avoid host syncs on the hot path. The embedding lookup will
        # raise an out-of-range error if vocab bounds are violated, and
        # we rely on the data pipeline to keep ids well-formed.
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [B, L]")

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have shape [B, L]")

        if past_len + input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # Last linear of every residual write: attention output, SwiGLU down
    # projection, and the SSM output projection (both NaiveSSM and the
    # official Mamba3 expose it as ``out_proj``).
    _RESIDUAL_PROJ_SUFFIXES = ("attn.o_proj", "ffn.w_down", "mamba.out_proj")

    @torch.no_grad()
    def _scale_residual_projections(self) -> None:
        """GPT-2-style depth-scaled init, extended to the unrolled loop.

        Every sublayer writes ``x + f(x)`` into the residual stream. With a
        weight-shared recurrent core the stream sees ``effective_depth``
        layers (prelude + block_layers x recurrent_steps + coda), so leaving
        the output projections at ``init_std`` makes the hidden-state norm
        grow linearly across recurrent steps (and costs bf16 mantissa
        precision late in the loop). Scaling them by
        ``1/sqrt(2 * effective_depth)`` keeps the residual variance roughly
        depth-invariant at init.
        """

        scale = (2.0 * self.cfg.effective_depth) ** -0.5

        for container in (self.prelude, self.recurrent, self.coda):
            for name, module in container.named_modules():
                if isinstance(module, nn.Linear) and name.endswith(
                    self._RESIDUAL_PROJ_SUFFIXES
                ):
                    module.weight.mul_(scale)

    @torch.no_grad()
    def count_params(self, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        total = 0

        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue

            ptr = param.data_ptr()
            if ptr in seen:
                continue

            seen.add(ptr)
            total += param.numel()

        return total


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    model = RDTForCausalLM(cfg)

    print("RDTForCausalLM")
    print(f"  params: {model.count_params():,}")
    print(f"  actual_layers: {cfg.actual_layers}")
    print(f"  effective_depth: {cfg.effective_depth}")

    bsz, seq_len = 2, 32
    input_ids = torch.randint(256, 24576, (bsz, seq_len))
    input_ids[:, 0] = cfg.bos_id

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    print("Text")
    print(f"  logits: {tuple(out['logits'].shape)}")
    print(f"  loss: {out['loss'].item():.6f}")
    print(f"  loss_parts: {out['loss_parts']}")
    print(f"  steps_used: {out['rec_info'].get('steps_used')}")

    out["loss"].backward()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += p.grad.norm().item() ** 2

    print(f"  grad_norm: {grad_sq**0.5:.6f}")

    ids2 = torch.tensor(
        [
            [
                cfg.bos_id,
                300,
                cfg.image_start_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_end_id,
                301,
                cfg.eos_id,
            ]
        ]
    )

    labels2 = ids2.clone()
    labels2[ids2 == cfg.image_start_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_patch_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_end_id] = cfg.ignore_index

    pixel_values = torch.randn(2, model.patch_pixels)

    out2 = model(
        input_ids=ids2,
        labels=labels2,
        pixel_values=pixel_values,
    )

    print("ImageText")
    print(f"  loss: {out2['loss'].item():.6f}")

    model.eval()
    with torch.no_grad():
        for steps in [2, 8]:
            out_step = model(input_ids=input_ids, steps=steps)
            print(
                f"  steps={steps}, logits_norm={out_step['logits'].norm().item():.6f}"
            )


if __name__ == "__main__":
    _check()
