# -*- coding: utf-8 -*-

"""Tests for the Chinese science/reasoning seed extractor."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from Tokenizer.tools.extract_science_seed import classify, extract


def _write_journal(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


class ClassifyTest(unittest.TestCase):
    def test_domains(self) -> None:
        self.assertEqual(classify("这是关于哲学与逻辑的研究"), "philosophy")
        self.assertEqual(classify("微积分与概率论"), "math")
        self.assertEqual(classify("量子力学与相对论"), "physics")
        self.assertEqual(classify("有机化学反应机理"), "chemistry")
        self.assertEqual(classify("一般科学实验方法"), "science")

    def test_no_match(self) -> None:
        self.assertIsNone(classify("今天的新闻和娱乐八卦"))

    def test_specific_before_generic(self) -> None:
        # Contains both 科学 (science) and 数学 (math) -> math wins (more specific).
        self.assertEqual(classify("数学科学"), "math")


class ExtractTest(unittest.TestCase):
    def _make(self, tmp: str) -> str:
        d = os.path.join(tmp, "journal")
        os.makedirs(d)
        recs = [
            {
                "media_c": "物理学报",
                "title_c": "量子纠缠研究",
                "keyword_c": "量子 力学",
                "remark_c": "本文研究量子纠缠现象" + "详" * 100,
            },
            {
                "media_c": "数学学报",
                "title_c": "代数几何",
                "keyword_c": "代数 几何",
                "remark_c": "本文讨论代数几何的若干问题" + "细" * 100,
            },
            {
                "media_c": "娱乐周刊",
                "title_c": "明星八卦",
                "keyword_c": "娱乐",
                "remark_c": "本期介绍最新娱乐资讯" + "闲" * 100,
            },
            {
                "media_c": "短文",
                "title_c": "短",
                "keyword_c": "物理",
                "remark_c": "太短",  # below min-chars -> dropped
            },
        ]
        _write_journal(os.path.join(d, "j1.json"), recs)
        return d

    def test_filters_and_classifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make(tmp)
            rows = list(extract(d, per_domain=100, min_chars=80, max_chars=2000))
            domains = {r["domain"] for r in rows}
            # Physics + math kept; entertainment + too-short dropped.
            self.assertEqual(domains, {"physics", "math"})
            self.assertEqual(len(rows), 2)
            for r in rows:
                self.assertGreaterEqual(len(r["text"]), 80)
                self.assertIn("title", r)

    def test_per_domain_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "journal")
            os.makedirs(d)
            recs = [
                {
                    "media_c": "物理学报",
                    "title_c": f"题{i}",
                    "keyword_c": "物理",
                    "remark_c": f"独特物理研究内容编号{i}" + "字" * 100,
                }
                for i in range(10)
            ]
            _write_journal(os.path.join(d, "j.json"), recs)
            rows = list(extract(d, per_domain=3, min_chars=80, max_chars=2000))
            self.assertEqual(len(rows), 3)

    def test_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "journal")
            os.makedirs(d)
            same = "完全相同的物理摘要内容" + "同" * 100
            recs = [
                {"media_c": "物理", "title_c": "a", "keyword_c": "物理", "remark_c": same},
                {"media_c": "物理", "title_c": "b", "keyword_c": "物理", "remark_c": same},
            ]
            _write_journal(os.path.join(d, "j.json"), recs)
            rows = list(extract(d, per_domain=100, min_chars=80, max_chars=2000))
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
