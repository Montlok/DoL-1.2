# -*- coding: utf-8 -*-

"""Inference-time state caches for incremental decoding.

A weight-shared recurrent core means the *same* attention/SSM module runs
once per recurrent step with different inputs, so cache slots are keyed by
``(stage, step, layer)`` — one slot per unrolled position, not per module:

* ``("prelude", i)`` / ``("coda", i)`` — the fixed outer blocks.
* ``("rec", step_idx, layer_idx)`` — the recurrent block, one slot per
  (loop step x sublayer).

Attention slots store the post-RoPE key/value tensors. Recomputing keys
from the MLA latent with absorbed projections (the memory-optimal DeepSeek
decode path) is a follow-up; the full-tensor cache below is exact and is
what the equivalence tests pin down.

SSM slots store backend-specific state: the NaiveSSM fallback keeps a
``(conv_buf, ssm_state)`` tuple, the official mamba backend keeps an
``InferenceParams`` object plus its running ``seqlen_offset``.
"""

from __future__ import annotations

from typing import Any, Hashable

import torch


class AttnCacheEntry:
    """Rolling key/value tensors for one attention slot."""

    __slots__ = ("k", "v")

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def past_len(self) -> int:
        return 0 if self.k is None else int(self.k.shape[-2])

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full tensors."""

        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            if self.k.shape[0] != k_new.shape[0]:
                raise ValueError("cache batch size changed between calls")
            self.k = torch.cat([self.k, k_new], dim=-2)
            self.v = torch.cat([self.v, v_new], dim=-2)
        return self.k, self.v


class MambaCacheEntry:
    """Backend-specific SSM state for one mamba slot."""

    __slots__ = ("state", "inference_params", "seqlen_offset")

    def __init__(self) -> None:
        self.state: Any = None
        self.inference_params: Any = None
        self.seqlen_offset: int = 0


class RDTCache:
    """Per-generation container of all attention/SSM cache slots."""

    def __init__(self) -> None:
        self._attn: dict[Hashable, AttnCacheEntry] = {}
        self._mamba: dict[Hashable, MambaCacheEntry] = {}
        self.seq_len = 0

    def attn_entry(self, key: Hashable) -> AttnCacheEntry:
        entry = self._attn.get(key)
        if entry is None:
            entry = self._attn[key] = AttnCacheEntry()
        return entry

    def mamba_entry(self, key: Hashable) -> MambaCacheEntry:
        entry = self._mamba.get(key)
        if entry is None:
            entry = self._mamba[key] = MambaCacheEntry()
        return entry

    def advance(self, n_tokens: int) -> None:
        if n_tokens < 0:
            raise ValueError("n_tokens must be non-negative")
        self.seq_len += n_tokens


__all__ = ["AttnCacheEntry", "MambaCacheEntry", "RDTCache"]
