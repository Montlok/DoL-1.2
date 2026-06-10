# -*- coding: utf-8 -*-

"""Preference / prompt datasets for DPO and GRPO training.

Reuses the SFT chat renderer/masking so completion masks are identical to the
ones validated by ``test_sft_data`` (assistant content + EOS supervised,
prompt / role headers / tool-results masked). This keeps offline preference
(DPO) and online RL (GRPO) scoring consistent with SFT.

JSONL formats
-------------
DPO (``PreferenceDataset``)::

    {"messages": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}
    # or a bare prompt string instead of messages:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

GRPO (``PromptDataset``)::

    {"messages": [...]} | {"prompt": "..."}            # required
    {"reference": "..."}                                # optional, for verifiable reward
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset

from Model.config import IGNORE_INDEX

from .sft_data import build_sft_example

Encode = Callable[[str], list[int]]


def _as_messages(obj: dict) -> list[dict[str, str]]:
    if "messages" in obj:
        return list(obj["messages"])
    if "prompt" in obj:
        return [{"role": "user", "content": obj["prompt"]}]
    raise KeyError("preference/prompt row needs 'messages' or 'prompt'")


def build_preference_example(
    context: list[dict[str, str]],
    chosen: str,
    rejected: str,
    encode: Encode,
    eos_id: int,
    bos_id: int | None = None,
    max_seq_len: int | None = None,
) -> dict[str, list[int]]:
    """Tokenize a (context, chosen, rejected) triple into two masked sequences.

    Both branches share the same context; the completion mask is 1 exactly on
    the appended assistant response tokens (+ EOS), reusing the SFT builder.
    """
    def _one(answer: str) -> tuple[list[int], list[int]]:
        msgs = [*context, {"role": "assistant", "content": answer}]
        ex = build_sft_example(
            msgs, encode, eos_id=eos_id, bos_id=bos_id,
            ignore_index=IGNORE_INDEX, max_seq_len=max_seq_len,
        )
        ids = ex["input_ids"]
        mask = [0 if label == IGNORE_INDEX else 1 for label in ex["labels"]]
        return ids, mask

    chosen_ids, chosen_mask = _one(chosen)
    rejected_ids, rejected_mask = _one(rejected)
    if not any(chosen_mask):
        raise ValueError("chosen response has no supervised completion tokens")
    if not any(rejected_mask):
        raise ValueError("rejected response has no supervised completion tokens")
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_completion_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_completion_mask": rejected_mask,
    }


def _iter_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class PreferenceDataset(Dataset):
    """Map-style dataset of DPO preference examples from JSONL."""

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        eos_id: int,
        bos_id: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self._rows = [
            build_preference_example(
                _as_messages(obj), obj["chosen"], obj["rejected"],
                encode, eos_id=eos_id, bos_id=bos_id, max_seq_len=max_seq_len,
            )
            for obj in _iter_jsonl(path)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._rows[idx]


def _pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max((len(s) for s in seqs), default=1)
    return torch.tensor(
        [s + [pad_value] * (width - len(s)) for s in seqs], dtype=torch.long
    )


def preference_collate(
    rows: list[dict[str, list[int]]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Pad a batch of preference examples (chosen/rejected padded independently).

    Returns ``chosen_input_ids``, ``chosen_completion_mask``,
    ``chosen_attention_mask`` and the ``rejected_*`` counterparts.
    """
    out: dict[str, torch.Tensor] = {}
    for side in ("chosen", "rejected"):
        ids = _pad([r[f"{side}_input_ids"] for r in rows], pad_id)
        comp = _pad([r[f"{side}_completion_mask"] for r in rows], 0).float()
        attn = (ids != pad_id).long()
        out[f"{side}_input_ids"] = ids
        out[f"{side}_completion_mask"] = comp
        out[f"{side}_attention_mask"] = attn
    return out


class PromptDataset(Dataset):
    """Map-style dataset of GRPO prompts (+ optional reference answer)."""

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        bos_id: int | None = None,
        max_prompt_len: int | None = None,
    ) -> None:
        from .sft_data import generation_prompt_ids

        self._rows: list[dict] = []
        for obj in _iter_jsonl(path):
            ids = generation_prompt_ids(_as_messages(obj), encode, bos_id=bos_id)
            if max_prompt_len is not None:
                ids = ids[-max_prompt_len:]
            self._rows.append(
                {"prompt_ids": ids, "reference": obj.get("reference")}
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


__all__ = [
    "PreferenceDataset",
    "PromptDataset",
    "build_preference_example",
    "preference_collate",
]
