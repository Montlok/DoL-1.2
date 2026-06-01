# -*- coding: utf-8 -*-
"""Unified tokenizer id-space registry and vocabulary builders.

Two content tracks share one fixed 65536 id space:

* ``mongolian`` — MorphBPE (morphology-aware traditional Mongolian).
* ``general``   — one multilingual byte-level BPE covering Chinese, English,
  Japanese and Cyrillic Mongolian. Byte-level coverage means no ``<unk>`` for
  any non-Mongolian text.
"""

from __future__ import annotations

try:
    from ..multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS
except ImportError:  # pragma: no cover
    from Tokenizer.multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS


BASE_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<img>": 4,
    "\u2581": 17,
    "\u25c8": 18,
}

SPECIAL_TOKENS = {
    **BASE_SPECIAL_TOKENS,
    **MULTIMODAL_SPECIAL_TOKENS,
}

SEGMENT = {
    "special": (0, 256),
    "mongolian": (256, 24576),
    "general": (24576, 65536),
}


def make_byte_tokens() -> list[str]:
    return [f"<0x{i:02X}>" for i in range(256)]


def build_unified_vocab(
    morphbpe_vocab: dict[str, int],
    general_vocab: dict[str, int],
) -> dict[str, int]:
    """Lay out MorphBPE + general byte-level BPE in the fixed 65536 id space.

    MorphBPE tokens fill the ``mongolian`` segment; general byte-level BPE
    tokens fill the ``general`` segment. Tokens are assigned in ascending
    local-id order so the layout is deterministic and reproducible. Tokens
    that collide with reserved special tokens are skipped to keep special ids
    stable.
    """

    unified: dict[str, int] = dict(SPECIAL_TOKENS)

    mn_lo, mn_hi = SEGMENT["mongolian"]
    next_id = mn_lo
    for token, _local_id in sorted(morphbpe_vocab.items(), key=lambda x: x[1]):
        if token in SPECIAL_TOKENS:
            continue
        if next_id >= mn_hi:
            break
        if token in unified:
            continue
        unified[token] = next_id
        next_id += 1

    gen_lo, gen_hi = SEGMENT["general"]
    next_id = gen_lo
    for token, _local_id in sorted(general_vocab.items(), key=lambda x: x[1]):
        if token in SPECIAL_TOKENS:
            continue
        if next_id >= gen_hi:
            break
        if token in unified:
            continue
        unified[token] = next_id
        next_id += 1

    return unified
