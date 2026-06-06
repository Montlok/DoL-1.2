# -*- coding: utf-8 -*-
"""Build minimum JSONL pretraining data with a tokenizer bundle."""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from Tokenizer.pretraining import (
    IGNORE_INDEX,
    EncodedSample,
    PretrainingDataBuilder,
    encoded_sample_to_dict,
    pack_samples,
)
from Tokenizer.unified.bundle import TokenizerBundle


def _iter_samples(
    path: str, builder: PretrainingDataBuilder
) -> Iterable[EncodedSample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield builder.encode_jsonl_line(line)
            else:
                yield builder.encode_text(line, metadata={"type": "text"})


def _summary(samples: Iterable[dict], unk_id: int, skipped_empty: int) -> dict:
    rows = list(samples)
    lengths = [len(row["input_ids"]) for row in rows]
    total_tokens = sum(lengths)
    unk_count = sum(row["input_ids"].count(unk_id) for row in rows)
    supervised_tokens = sum(
        1 for row in rows for label in row["labels"] if int(label) != IGNORE_INDEX
    )
    max_morph_depth = max(
        (
            max((int(value) for value in row.get("morph_depth", [])), default=0)
            for row in rows
        ),
        default=0,
    )
    return {
        "num_samples": len(rows),
        "skipped_empty": skipped_empty,
        "avg_len": (total_tokens / len(rows)) if rows else 0.0,
        "max_len": max(lengths) if lengths else 0,
        "unk_rate": (unk_count / total_tokens) if total_tokens else 0.0,
        "supervised_tokens": supervised_tokens,
        "supervised_rate": (supervised_tokens / total_tokens) if total_tokens else 0.0,
        "max_morph_depth": max_morph_depth,
    }


def _new_summary_stats(skipped_empty: int = 0) -> dict:
    return {
        "num_samples": 0,
        "skipped_empty": skipped_empty,
        "total_tokens": 0,
        "max_len": 0,
        "unk_count": 0,
        "supervised_tokens": 0,
        "max_morph_depth": 0,
    }


def _update_summary_stats(stats: dict, row: dict, unk_id: int) -> None:
    length = len(row["input_ids"])
    stats["num_samples"] += 1
    stats["total_tokens"] += length
    stats["max_len"] = max(stats["max_len"], length)
    stats["unk_count"] += row["input_ids"].count(unk_id)
    stats["supervised_tokens"] += sum(
        1 for label in row["labels"] if int(label) != IGNORE_INDEX
    )
    stats["max_morph_depth"] = max(
        stats["max_morph_depth"],
        max((int(value) for value in row.get("morph_depth", [])), default=0),
    )


def _finalize_summary(stats: dict) -> dict:
    total_tokens = int(stats["total_tokens"])
    num_samples = int(stats["num_samples"])
    return {
        "num_samples": num_samples,
        "skipped_empty": int(stats["skipped_empty"]),
        "avg_len": (total_tokens / num_samples) if num_samples else 0.0,
        "max_len": int(stats["max_len"]),
        "unk_rate": (int(stats["unk_count"]) / total_tokens) if total_tokens else 0.0,
        "supervised_tokens": int(stats["supervised_tokens"]),
        "supervised_rate": (
            int(stats["supervised_tokens"]) / total_tokens
        ) if total_tokens else 0.0,
        "max_morph_depth": int(stats["max_morph_depth"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-bundle", required=True)
    parser.add_argument("--input", required=True, help=".txt or .jsonl")
    parser.add_argument("--output", required=True, help="output JSONL")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pack", action="store_true", help="pack text-only samples")
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="pad packed samples to the configured packed sequence length",
    )
    parser.add_argument(
        "--pack-max-length",
        type=int,
        default=None,
        help="packed sequence length; defaults to --max-length",
    )
    args = parser.parse_args()

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    builder = PretrainingDataBuilder(bundle, max_length=args.max_length)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not args.pack:
        stats = _new_summary_stats()
        with open(args.output, "w", encoding="utf-8") as f:
            for sample in _iter_samples(args.input, builder):
                if not sample.input_ids:
                    stats["skipped_empty"] += 1
                    continue
                row = encoded_sample_to_dict(sample)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                _update_summary_stats(stats, row, bundle.tokenizer.unk_id)
        print(json.dumps(_finalize_summary(stats), ensure_ascii=False))
        return

    samples = list(_iter_samples(args.input, builder))
    skipped_empty = sum(1 for sample in samples if not sample.input_ids)
    samples = [sample for sample in samples if sample.input_ids]
    samples = pack_samples(
        samples,
        max_length=args.pack_max_length or args.max_length,
        pad_id=bundle.tokenizer.vocab["<pad>"],
        eos_id=bundle.tokenizer.vocab["<eos>"],
        pad_to_max_length=args.pad_to_max_length,
    )
    rows = [encoded_sample_to_dict(sample) for sample in samples]
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            _summary(rows, bundle.tokenizer.unk_id, skipped_empty), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
