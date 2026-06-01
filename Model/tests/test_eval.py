# -*- coding: utf-8 -*-

"""Tests for the alignment eval suite (Model/posttrain/eval.py)."""

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM
from Model.posttrain.eval import (
    decode_cache_consistency,
    format_compliance_rate,
    purity_report,
    reward_report,
)
from Model.posttrain.rewards import RewardConfig


def _cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8, rope_head_dim=4,
        nope_head_dim=4, ffn_hidden=64, ffn_multiple=32, n_prelude=2, n_coda=2,
        recurrent_steps=3, mamba_d_state=8, mamba_expand=2, mamba_headdim=8,
        use_official_mamba=False, max_seq_len=64, core_type="two_stage",
        stage1_mamba_layers=3, stage2_attn_layers=2, recurrent_drift_mode="mhc",
    )


class CorrectnessGateTest(unittest.TestCase):
    def test_decode_cache_consistency_is_tight(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(_cfg()).eval()
        ids = torch.randint(300, model.cfg.vocab_size, (2, 11))
        diff = decode_cache_consistency(model, ids)
        self.assertLess(diff, 1e-4)


class QualityReportTest(unittest.TestCase):
    def test_reward_report_stats(self):
        cfg = RewardConfig(exact_match_weight=1.0)
        rep = reward_report(["a", "b", "c"], ["a", "x", "c"], cfg)
        self.assertEqual(rep["n"], 3.0)
        self.assertAlmostEqual(rep["reward_max"], 1.0, places=5)
        self.assertAlmostEqual(rep["reward_min"], 0.0, places=5)
        self.assertAlmostEqual(rep["reward_mean"], 2.0 / 3.0, places=5)

    def test_purity_report(self):
        rep = purity_report(["Сайн байна уу", "hello world"], min_ratio=0.8)
        self.assertGreater(rep["purity_mean"], 0.0)
        self.assertAlmostEqual(rep["frac_below"], 0.5, places=5)

    def test_format_compliance_rate(self):
        rate = format_compliance_rate(
            ["<think>r</think> ans", "no think", "<think>x</think> y"]
        )
        self.assertAlmostEqual(rate, 2.0 / 3.0, places=5)

    def test_empty_inputs_safe(self):
        self.assertEqual(format_compliance_rate([]), 0.0)
        self.assertEqual(purity_report([])["purity_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
