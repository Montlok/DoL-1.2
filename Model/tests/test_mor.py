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


class ACTDepthKnobTest(unittest.TestCase):
    """ACT decides its own depth; the fixed-depth knobs must not be silently
    dropped. ``steps`` raises (it cannot override a learned halt); ``bptt_window``
    IS honoured (truncated BPTT through the unrolled ACT loop)."""

    def _model(self):
        torch.manual_seed(0)
        return RDTForCausalLM(_fast_mor_cfg()).eval()

    def _io(self):
        cfg = _fast_mor_cfg()
        e0 = torch.randn(2, 8, cfg.d_model)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        return e0, wp, md

    def test_steps_override_raises_under_act(self):
        core = self._model().recurrent
        e0, wp, md = self._io()
        with self.assertRaises(ValueError):
            core(e0, word_pos=wp, morph_depth=md, steps=3)

    def test_forward_steps_override_raises(self):
        model = self._model()
        ids = torch.randint(300, 320, (2, 16))
        with self.assertRaises(ValueError):
            model(input_ids=ids, steps=2)

    def test_generate_recurrent_steps_raises_under_act(self):
        # The 'think harder' knob is meaningless for ACT (the halt head owns the
        # depth) -- it used to be a silent no-op; now it raises.
        model = self._model()
        prompt = torch.tensor([[model.cfg.bos_id, 300, 301]])
        with self.assertRaises(ValueError):
            model.generate(prompt, max_new_tokens=2, greedy=True, recurrent_steps=2)

    def test_generate_without_override_still_works_under_act(self):
        # Plain generate() passes steps=None -> must not raise.
        model = self._model()
        prompt = torch.tensor([[model.cfg.bos_id, 300, 301]])
        out = model.generate(prompt, max_new_tokens=2, greedy=True)
        self.assertEqual(out.shape[1], prompt.shape[1] + 2)

    def test_bptt_window_honoured_under_act(self):
        core = self._model().recurrent
        cfg = _fast_mor_cfg()
        e0 = torch.randn(2, 8, cfg.d_model, requires_grad=True)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        out, _info = core(e0, word_pos=wp, morph_depth=md, bptt_window=2)
        out.sum().backward()
        self.assertIsNotNone(e0.grad)
        self.assertTrue(torch.isfinite(e0.grad).all())

    def test_bptt_window_truncates_act_graph(self):
        # Truncation severs the early-step recurrent/inject path, so the gradient
        # reaching e0 is strictly smaller than the full-BPTT gradient. If
        # bptt_window were silently ignored (the bug), these would be equal.
        cfg = replace(_fast_mor_cfg(), inject_embedding=True)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)

        torch.manual_seed(0)
        core = RDTForCausalLM(cfg).eval().recurrent
        e_full = torch.randn(2, 8, cfg.d_model, requires_grad=True)
        e_trunc = e_full.detach().clone().requires_grad_(True)

        core(e_full, word_pos=wp, morph_depth=md)[0].sum().backward()
        core(e_trunc, word_pos=wp, morph_depth=md, bptt_window=1)[0].sum().backward()

        self.assertLess(e_trunc.grad.norm().item(), e_full.grad.norm().item())

    def test_bptt_window_equals_max_is_full_bptt(self):
        # Boundary: window == act_max_steps must match no truncation exactly.
        cfg = _fast_mor_cfg()
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        e_a = torch.randn(2, 8, cfg.d_model, requires_grad=True)
        e_b = e_a.detach().clone().requires_grad_(True)

        torch.manual_seed(0)
        RDTForCausalLM(cfg).eval().recurrent(
            e_a, word_pos=wp, morph_depth=md
        )[0].sum().backward()
        torch.manual_seed(0)
        RDTForCausalLM(cfg).eval().recurrent(
            e_b, word_pos=wp, morph_depth=md, bptt_window=cfg.act_max_steps
        )[0].sum().backward()

        self.assertTrue(torch.allclose(e_a.grad, e_b.grad, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
