# -*- coding: utf-8 -*-

"""Regression tests for the token-weighted corpus mixer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from Tokenizer.tools.build_corpus_mix import (
    MixConfig,
    SourceSpec,
    _emit_counts,
    compute_plan,
    emit_mix,
    measure_source,
)


# Char-proxy encoder: ~1 token per 4 chars (matches the smoke fallback).
def _proxy_encode(text: str):
    return [0] * max(len(text) // 4, 1)


def _write_jsonl(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


class ManifestTest(unittest.TestCase):
    def test_weight_normalization_not_required(self) -> None:
        cfg = MixConfig.from_raw(
            {
                "sources": [
                    {"path": "a.jsonl", "lang": "mn", "weight": 5},
                    {"path": "b.jsonl", "lang": "en", "weight": 5},
                ]
            }
        )
        self.assertEqual(len(cfg.sources), 2)
        self.assertEqual(cfg.max_epochs, 4)

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            SourceSpec.from_raw({"lang": "mn"})

    def test_empty_sources_raises(self) -> None:
        with self.assertRaises(ValueError):
            MixConfig.from_raw({"sources": []})

    def test_format_inference(self) -> None:
        self.assertEqual(SourceSpec.from_raw({"path": "x.parquet"}).fmt, "parquet")
        self.assertEqual(SourceSpec.from_raw({"path": "x.jsonl"}).fmt, "jsonl")
        self.assertEqual(SourceSpec.from_raw({"path": "x.txt"}).fmt, "txt")


class EmitCountsTest(unittest.TestCase):
    def test_subsample_expectation(self) -> None:
        import random

        rng = random.Random(0)
        total = sum(_emit_counts(0.3, rng) for _ in range(10000))
        self.assertAlmostEqual(total / 10000, 0.3, delta=0.03)

    def test_upsample_expectation(self) -> None:
        import random

        rng = random.Random(0)
        total = sum(_emit_counts(2.5, rng) for _ in range(10000))
        self.assertAlmostEqual(total / 10000, 2.5, delta=0.05)

    def test_zero(self) -> None:
        import random

        self.assertEqual(_emit_counts(0.0, random.Random(0)), 0)


class PlanTest(unittest.TestCase):
    def _stats(self, tmp):
        mn = os.path.join(tmp, "mn.jsonl")
        en = os.path.join(tmp, "en.jsonl")
        # Mongolian: small (low-resource). English: large.
        _write_jsonl(mn, ["A" * 40 for _ in range(10)])  # 100 tokens total
        _write_jsonl(en, ["B" * 40 for _ in range(100)])  # 1000 tokens total
        specs = [
            SourceSpec.from_raw({"path": mn, "lang": "mn", "weight": 0.7}),
            SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.3}),
        ]
        return [measure_source(s, _proxy_encode) for s in specs]

    def test_low_resource_upsampled_with_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._stats(tmp)
            stats = compute_plan(stats, total_tokens=None, max_epochs=4)
            mn, en = stats
            # mn weight dominates so it should up-sample, capped at 4 epochs.
            self.assertGreater(mn.repeat_factor, 1.0)
            self.assertLessEqual(mn.repeat_factor, 4.0)
            # en is plentiful so it should sub-sample (<1).
            self.assertLess(en.repeat_factor, 1.0)

    def test_cap_blocks_runaway_upsampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._stats(tmp)
            # Huge budget would demand >>4 epochs of mn; cap must hold.
            stats = compute_plan(stats, total_tokens=10_000_000, max_epochs=4)
            self.assertLessEqual(stats[0].repeat_factor, 4.0)


class EndToEndTest(unittest.TestCase):
    def test_emit_produces_text_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mn = os.path.join(tmp, "mn.jsonl")
            en = os.path.join(tmp, "en.jsonl")
            _write_jsonl(mn, ["ᠮᠣᠩᠭᠣᠯ" * 8 for _ in range(20)])
            _write_jsonl(en, ["hello world " * 8 for _ in range(200)])
            specs = [
                SourceSpec.from_raw({"path": mn, "lang": "mn", "weight": 0.6}),
                SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.4}),
            ]
            stats = [measure_source(s, _proxy_encode) for s in specs]
            # Budget chosen so Mongolian's target stays under its 4-epoch cap,
            # letting the higher weight translate into a higher realized share.
            stats = compute_plan(stats, total_tokens=1500, max_epochs=4)
            out = os.path.join(tmp, "mix.jsonl")
            report = emit_mix(stats, out, seed=0)

            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertTrue(rows)
            # Every emitted row is a text-only record.
            for r in rows:
                self.assertEqual(list(r.keys()), ["text"])
                self.assertTrue(r["text"].strip())

            # Report aggregates a realized language mixture summing to ~1.
            self.assertIn("mn", report["by_lang"])
            self.assertIn("en", report["by_lang"])
            self.assertAlmostEqual(sum(report["by_lang"].values()), 1.0, delta=0.02)
            # Mongolian's realized share should track its higher weight.
            self.assertGreater(report["by_lang"]["mn"], report["by_lang"]["en"])

    def test_new_source_rebalances(self) -> None:
        # Dropping in a second Mongolian source must shift realized shares
        # without any manual reconfiguration (weights are re-solved each run).
        with tempfile.TemporaryDirectory() as tmp:
            mn1 = os.path.join(tmp, "mn1.jsonl")
            en = os.path.join(tmp, "en.jsonl")
            _write_jsonl(mn1, ["A" * 40 for _ in range(10)])
            _write_jsonl(en, ["B" * 40 for _ in range(100)])

            def shares(specs):
                stats = [measure_source(s, _proxy_encode) for s in specs]
                stats = compute_plan(stats, total_tokens=None, max_epochs=4)
                out = os.path.join(tmp, "o.jsonl")
                rep = emit_mix(stats, out, seed=1)
                return rep["by_lang"]

            base = shares(
                [
                    SourceSpec.from_raw({"path": mn1, "lang": "mn", "weight": 0.5}),
                    SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.5}),
                ]
            )
            mn2 = os.path.join(tmp, "mn2.jsonl")
            _write_jsonl(mn2, ["C" * 40 for _ in range(50)])
            grown = shares(
                [
                    SourceSpec.from_raw({"path": mn1, "lang": "mn", "weight": 0.25}),
                    SourceSpec.from_raw({"path": mn2, "lang": "mn", "weight": 0.25}),
                    SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.5}),
                ]
            )
            # Both runs target mn~=en, realized shares stay balanced.
            self.assertAlmostEqual(grown["mn"], base["mn"], delta=0.15)


if __name__ == "__main__":
    unittest.main()
