# -*- coding: utf-8 -*-

"""Tests for DPO/GRPO datasets (Model/posttrain/preference_data.py)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from Model.config import EOS_ID
from Model.posttrain.preference_data import (
    PreferenceDataset,
    PromptDataset,
    build_preference_example,
    preference_collate,
)


def _encode(text: str) -> list[int]:
    return [(ord(c) % 5000) + 1000 for c in text]


CTX = [{"role": "user", "content": "2+2?"}]


class BuildPreferenceTest(unittest.TestCase):
    def test_masks_only_assistant_response(self):
        ex = build_preference_example(CTX, "4", "5", _encode, eos_id=EOS_ID)
        # Completion mask must be 1 only on assistant tokens + EOS, and the last
        # supervised token id must be EOS.
        ids = ex["chosen_input_ids"]
        mask = ex["chosen_completion_mask"]
        self.assertEqual(len(ids), len(mask))
        self.assertEqual(sum(mask), len(_encode("4")) + 1)  # "4" + EOS
        last_sup = [i for i, m in zip(ids, mask) if m][-1]
        self.assertEqual(last_sup, EOS_ID)

    def test_chosen_and_rejected_share_prompt_prefix(self):
        ex = build_preference_example(CTX, "yes", "no", _encode, eos_id=EOS_ID)
        c, r = ex["chosen_input_ids"], ex["rejected_input_ids"]
        prefix = sum(1 for _ in range(min(len(c), len(r))) if c[_] == r[_])
        self.assertGreater(prefix, 0)


class CollateTest(unittest.TestCase):
    def test_collate_pads_and_builds_attention(self):
        rows = [
            build_preference_example(CTX, "4", "55", _encode, eos_id=EOS_ID),
            build_preference_example(
                [{"role": "user", "content": "longer question here"}],
                "a", "bb", _encode, eos_id=EOS_ID,
            ),
        ]
        batch = preference_collate(rows, pad_id=0)
        self.assertEqual(batch["chosen_input_ids"].shape[0], 2)
        # attention mask is 0 exactly where padded.
        ids = batch["chosen_input_ids"]
        self.assertTrue(torch.equal(batch["chosen_attention_mask"], (ids != 0).long()))
        # completion mask never marks padding.
        self.assertTrue(
            torch.all(batch["chosen_completion_mask"][ids == 0] == 0)
        )


class DatasetTest(unittest.TestCase):
    def test_preference_dataset_reads_both_formats(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pref.jsonl"
            p.write_text(
                json.dumps({"prompt": "hi", "chosen": "a", "rejected": "b"}) + "\n"
                + json.dumps({
                    "messages": [{"role": "user", "content": "q"}],
                    "chosen": "c", "rejected": "d",
                }) + "\n",
                encoding="utf-8",
            )
            ds = PreferenceDataset(p, _encode, eos_id=EOS_ID)
            self.assertEqual(len(ds), 2)
            self.assertIn("chosen_input_ids", ds[0])

    def test_prompt_dataset_carries_reference(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompts.jsonl"
            p.write_text(
                json.dumps({"prompt": "2+2?", "reference": "4"}) + "\n",
                encoding="utf-8",
            )
            ds = PromptDataset(p, _encode)
            self.assertEqual(ds[0]["reference"], "4")
            self.assertGreater(len(ds[0]["prompt_ids"]), 0)


if __name__ == "__main__":
    unittest.main()
