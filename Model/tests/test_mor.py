# -*- coding: utf-8 -*-

"""Tests for step-aware MoR depth routing (ACT halting enhanced per step)."""

import unittest
from dataclasses import replace

import torch

from Model.config import mor_tiny_config, tiny_config
from Model.model import RDTForCausalLM


def _fast_mor_cfg():
    # Small ACT bound + short sequence so the fully-unrolled ACT loop is cheap.
    return replace(mor_tiny_config(), act_max_steps=4, max_seq_len=64)


class MoRConfigTest(unittest.TestCase):
    def test_mor_tiny_is_full_router(self):
        cfg = mor_tiny_config()
        self.assertTrue(cfg.use_mor)
        self.assertTrue(cfg.use_act)
        self.assertTrue(cfg.use_mol)
        self.assertEqual(cfg.core_type, "interleaved")
        self.assertFalse(cfg.use_official_mamba)

    def test_rejects_mor_without_act(self):
        with self.assertRaises(ValueError):
            replace(tiny_config(), use_mor=True)  # use_act defaults to False

    def test_rejects_mor_on_two_stage(self):
        from Model.config import two_stage_tiny_config
        # two_stage rejects use_act, and MoR requires use_act -> rejected.
        with self.assertRaises(ValueError):
            replace(two_stage_tiny_config(), use_act=True, use_mor=True)


class MoRModelTest(unittest.TestCase):
    def test_cold_start_reduces_to_plain_act(self):
        # halt_step_bias=0 at init -> MoR on/off is bit-identical (the bias adds
        # exactly zero to the halt logit).
        torch.manual_seed(0)
        model = RDTForCausalLM(_fast_mor_cfg()).eval()
        ids = torch.randint(300, 320, (2, 16))
        core = model.recurrent
        self.assertTrue(core.use_mor)
        with torch.no_grad():
            l_mor = model(input_ids=ids)["logits"]
            core.use_mor = False
            l_act = model(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(l_mor, l_act, atol=1e-6))

    def test_step_aware_halting_changes_output(self):
        # A nonzero per-step bias must change the depth-weighted output -- proves
        # the bias is wired into the halting math at the right step index.
        torch.manual_seed(0)
        model = RDTForCausalLM(_fast_mor_cfg()).eval()
        ids = torch.randint(300, 320, (2, 16))
        core = model.recurrent
        with torch.no_grad():
            l0 = model(input_ids=ids)["logits"]
            # Bias step 0 strongly toward halting -> shallower effective depth.
            core.halt_step_bias[:] = torch.tensor([8.0, 0.0, 0.0, 0.0])
            l1 = model(input_ids=ids)["logits"]
        self.assertFalse(torch.allclose(l0, l1))

    def test_forward_backward_and_halt_bias_grad(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(_fast_mor_cfg())
        with torch.no_grad():
            model.recurrent.halt_step_bias.normal_()
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))
        g = model.recurrent.halt_step_bias.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        other = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(t).all() for t in other))

    def test_halt_bias_exempt_from_weight_decay(self):
        model = RDTForCausalLM(_fast_mor_cfg())
        self.assertTrue(getattr(model.recurrent.halt_step_bias, "_no_weight_decay", False))


if __name__ == "__main__":
    unittest.main()
