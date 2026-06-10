# -*- coding: utf-8 -*-

"""Build masked SFT training examples from chat conversations.

The builder is tokenizer-agnostic: it takes an ``encode`` callable so the
masking logic can be unit-tested deterministically and any tokenizer (the
real :class:`TokenizerBundle` via ``bundle.encode``) can be plugged in.

Output rows match the contract consumed by
:class:`Model.training.data.PretrainingCollator`: ``input_ids``,
``attention_mask`` and ``labels`` (with ``ignore_index`` on masked positions),
so SFT reuses the existing training loop unchanged.

Masking rules (rigorous, see tests):
- Only assistant content + its terminating EOS carry labels.
- System/user/tool messages and every role header are masked.
- ``labels`` are NOT pre-shifted; the model shifts internally (HF convention),
  so ``labels[i] == input_ids[i]`` on supervised positions and ``ignore_index``
  elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from torch.utils.data import Dataset

from Model.config import IGNORE_INDEX

from .chat_template import ROLE_SENTINELS, TURN_SUFFIX, render_message

Encode = Callable[[str], list[int]]


def build_sft_example(
    messages: Sequence[dict[str, str]],
    encode: Encode,
    eos_id: int,
    bos_id: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    max_seq_len: int | None = None,
) -> dict[str, list[int]]:
    """Render + tokenize one conversation into a masked SFT example.

    Args:
        messages: ``[{"role": ..., "content": ...}, ...]``.
        encode: maps a text span to token ids (no special tokens added).
        eos_id: appended (and supervised) after each assistant turn.
        bos_id: if given, prepended (masked) at sequence start.
        ignore_index: label value for masked positions.
        max_seq_len: optional right-truncation length.

    Returns:
        ``{"input_ids", "attention_mask", "labels"}`` (equal-length lists).
    """
    input_ids: list[int] = []
    labels: list[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(ignore_index)

    for msg in messages:
        role = msg["role"]
        for span in render_message(role, msg["content"]):
            ids = encode(span.text)
            input_ids.extend(ids)
            labels.extend(ids if span.supervised else [ignore_index] * len(ids))
        if role == "assistant":
            # Teach the model to terminate the turn.
            input_ids.append(eos_id)
            labels.append(eos_id)

    if max_seq_len is not None and len(input_ids) > max_seq_len:
        input_ids, labels = truncate_preserving_supervision(
            input_ids,
            labels,
            max_seq_len=max_seq_len,
            ignore_index=ignore_index,
        )

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def truncate_preserving_supervision(
    input_ids: list[int],
    labels: list[int],
    *,
    max_seq_len: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Truncate from the left while preserving supervised completion tokens.

    SFT/DPO rows often have long prompts and short assistant completions. Plain
    right-truncation can drop the assistant turn entirely, yielding an all-masked
    row that trains no objective. Keep the suffix ending at the last supervised
    token instead, so at least the completion/EOS remains supervised.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must align")
    if len(input_ids) <= max_seq_len:
        return list(input_ids), list(labels)

    supervised = [idx for idx, label in enumerate(labels) if label != ignore_index]
    if not supervised:
        return list(input_ids[-max_seq_len:]), list(labels[-max_seq_len:])

    last_supervised = supervised[-1]
    start = max(0, last_supervised + 1 - max_seq_len)
    end = start + max_seq_len
    return list(input_ids[start:end]), list(labels[start:end])


def has_supervised_tokens(example: dict[str, list[int]], ignore_index: int = IGNORE_INDEX) -> bool:
    return any(label != ignore_index for label in example.get("labels", []))


def generation_prompt_ids(
    messages: Sequence[dict[str, str]],
    encode: Encode,
    bos_id: int | None = None,
) -> list[int]:
    """Token ids for a prompt ending at ``<|assistant|>\\n`` (for inference)."""
    ids: list[int] = [] if bos_id is None else [bos_id]
    for msg in messages:
        for span in render_message(msg["role"], msg["content"]):
            ids.extend(encode(span.text))
    ids.extend(encode(ROLE_SENTINELS["assistant"]))
    return ids


def iter_chat_jsonl(path: str | Path) -> Iterator[list[dict[str, str]]]:
    """Yield ``messages`` lists from a chat JSONL file.

    Each line is an object with a ``messages`` key (OpenAI-style) or a bare
    list of messages.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["messages"] if isinstance(obj, dict) else obj


class SFTChatDataset(Dataset):
    """Map-style dataset of masked SFT examples built from chat JSONL."""

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        eos_id: int,
        bos_id: int | None = None,
        ignore_index: int = IGNORE_INDEX,
        max_seq_len: int | None = None,
    ) -> None:
        self._rows = []
        for messages in iter_chat_jsonl(path):
            row = build_sft_example(
                messages,
                encode,
                eos_id=eos_id,
                bos_id=bos_id,
                ignore_index=ignore_index,
                max_seq_len=max_seq_len,
            )
            if has_supervised_tokens(row, ignore_index):
                self._rows.append(row)
        if not self._rows:
            raise ValueError("SFTChatDataset contains no supervised tokens")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._rows[idx]


__all__ = [
    "build_sft_example",
    "generation_prompt_ids",
    "has_supervised_tokens",
    "iter_chat_jsonl",
    "SFTChatDataset",
    "TURN_SUFFIX",
    "truncate_preserving_supervision",
]
