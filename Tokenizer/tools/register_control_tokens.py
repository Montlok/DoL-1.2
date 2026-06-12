# -*- coding: utf-8 -*-

"""Register MVS/FVS/NIRUGU/NNBSP as vocab tokens in an existing MorphBPE model.

Upgrades a v2-era ``morphbpe.json`` (controls folded at encode) to the
control-preserving contract without retraining: the BPE merges are untouched,
the seven control characters are appended at the next free local ids, and
``MorphBPETokenizer`` switches to preserving mode purely on their presence.
Re-assemble the unified bundle afterwards (``build_unified_tokenizer``) so
the new ids reach the global vocab.

Usage::

    python3 -m Tokenizer.tools.register_control_tokens \
        --input  .../tok_build_v2/tokenizer/morphbpe.json \
        --output .../tok_build_v3/tokenizer/morphbpe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from Tokenizer.traditional_mongolian.unicode_norm import CONTROL_CHARS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="existing morphbpe.json")
    p.add_argument("--output", required=True, help="upgraded morphbpe.json")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vocab = payload.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        print("register_control_tokens: payload has no vocab", file=sys.stderr)
        return 2

    next_id = max(int(v) for v in vocab.values()) + 1
    added = []
    for name, ch in sorted(CONTROL_CHARS.items(), key=lambda kv: kv[1]):
        if ch in vocab:
            continue
        vocab[ch] = next_id
        added.append((name, next_id))
        next_id += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    if added:
        print(
            "register_control_tokens: added "
            + ", ".join(f"{name}={tid}" for name, tid in added)
        )
    else:
        print("register_control_tokens: all control tokens already present")
    print(f"register_control_tokens: wrote {args.output} (vocab={len(vocab)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
