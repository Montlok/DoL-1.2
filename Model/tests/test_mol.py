# -*- coding: utf-8 -*-

"""Tests for step-aware Mixture-of-LoRAs in the recurrent FFN (MoL)."""

import unittest
import warnings
from dataclasses import replace

import torch

from Model.config import mol_tiny_config, tiny_config
from Model.layers.mixture_lora_ffn import MixtureLoRAFFN
from Model.layers.swiglu import SwiGLU
from Model.model import RDTForCausalLM
from Model.recurrent import RecurrentCore


def _mol_modules(model):
    return [m for m in model.modules() if isinstance(m, MixtureLoRAFFN)]


class MixtureLoRAFFNUnitTest(unittest.TestCase):
    def _ffn(self, step_aware=True, top_k=0):
        torch.manual_seed(0)
        return MixtureLoRAFFN(
            64, 128, n_experts=4, rank=8, step_table=4,
            step_aware=step_aware, top_k=top_k,
        )

    def test_cold_start_equals_base_swiglu(self):
        # lora_b=0 at init -> identical to a plain SwiGLU sharing w_in/w_down,
        # regardless of router_alpha / step_embed (the keystone guarantee).
        ffn = self._ffn()
        base = SwiGLU(64, 128)
        base.w_in.load_state_dict(ffn.w_in.state_dict())
        base.w_down.load_state_dict(ffn.w_down.state_dict())
        with torch.no_grad():
            ffn.router_alpha.fill_(5.0)
            ffn.step_embed.normal_()
        x = torch.randn(2, 8, 64)
        self.assertTrue(torch.allclose(ffn(x, step=2), base(x), atol=1e-6))

    def test_step_aware_routing_differs_across_steps(self):
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(1.0)
            ffn.step_embed.normal_()
            ffn.lora_b.normal_(std=0.02)
        x = torch.randn(2, 8, 64)
        self.assertFalse(torch.allclose(ffn(x, step=0), ffn(x, step=1)))

    def test_step_aware_off_ignores_step(self):
        ffn = self._ffn(step_aware=False)
        self.assertIsNone(ffn.step_embed)
        with torch.no_grad():
            ffn.router_alpha.fill_(1.0)
            ffn.lora_b.normal_(std=0.02)
        x = torch.randn(2, 8, 64)
        self.assertTrue(torch.allclose(ffn(x, step=0), ffn(x, step=3), atol=1e-6))

    def test_ffn_is_per_position(self):
        # No cross-position mixing: perturbing later positions must not change
        # earlier outputs. This is what keeps MoL causal-safe regardless of the
        # surrounding model's direction.
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(1.0)
            ffn.step_embed.normal_()
            ffn.lora_b.normal_(std=0.02)
        x = torch.randn(2, 8, 64)
        y = ffn(x, step=1)
        x2 = x.clone()
        x2[:, 4:] = torch.randn(2, 4, 64)
        y2 = ffn(x2, step=1)
        self.assertTrue(torch.allclose(y[:, :4], y2[:, :4], atol=1e-6))

    def test_router_is_valid_distribution(self):
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(2.0)
            ffn.router_proj.weight.normal_()
        x = torch.randn(2, 8, 64)
        ref = ffn.router_norm(x) + ffn.step_embed[1]
        logits = ffn.router_bias + ffn.router_alpha * torch.tanh(ffn.router_proj(ref))
        w = torch.softmax(logits / ffn.router_temp, dim=-1)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2, 8), atol=1e-5))
        self.assertTrue((w >= 0).all())

    def test_topk_keeps_k_experts(self):
        ffn = self._ffn(top_k=2)
        with torch.no_grad():
            ffn.router_alpha.fill_(2.0)
            ffn.router_proj.weight.normal_()
        x = torch.randn(2, 8, 64)
        logits = ffn.router_bias + ffn.router_alpha * torch.tanh(
            ffn.router_proj(ffn.router_norm(x))
        )
        w = torch.softmax(ffn._topk_mask(logits / ffn.router_temp, 2), dim=-1)
        self.assertTrue(((w > 0).sum(-1) == 2).all())
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2, 8), atol=1e-5))

    def test_clamps_step_beyond_table(self):
        ffn = self._ffn()  # step_table=4
        # A deeper step than the table must clamp, not index out of range.
        ffn(torch.randn(2, 8, 64), step=99)

    def test_backward_grads_finite(self):
        ffn = self._ffn()
        with torch.no_grad():
            ffn.lora_b.normal_(std=0.02)
            ffn.router_alpha.fill_(1.0)
        x = torch.randn(2, 8, 64, requires_grad=True)
        ffn(x, step=1).sum().backward()
        for p in (ffn.lora_a, ffn.lora_b, ffn.router_proj.weight,
                  ffn.router_alpha, ffn.step_embed):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(ffn.lora_a.grad.norm().item(), 0.0)


class MoLModelTest(unittest.TestCase):
    def test_mol_config_is_cpu_interleaved(self):
        cfg = mol_tiny_config()
        self.assertTrue(cfg.use_mol)
        self.assertEqual(cfg.core_type, "interleaved")
        self.assertFalse(cfg.use_official_mamba)

    def test_rejects_non_interleaved(self):
        from dataclasses import replace

        from Model.config import two_stage_tiny_config
        with self.assertRaises(ValueError):
            replace(two_stage_tiny_config(), use_mol=True)

    def test_cold_start_invariant_to_router_perturbation(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(mol_tiny_config()).eval()
        ids = torch.randint(300, 320, (2, 16))
        with torch.no_grad():
            l0 = model(input_ids=ids)["logits"]
            for m in _mol_modules(model):
                m.router_alpha.fill_(3.0)
                m.step_embed.normal_()
            l1 = model(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(l0, l1, atol=1e-6))

    def test_forward_backward_finite_on_cpu(self):
        torch.manual_seed(0)
        cfg = mol_tiny_config()
        self.assertFalse(cfg.use_official_mamba)  # NaiveSSM CPU path
        model = RDTForCausalLM(cfg)
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_step_threaded_to_ffn(self):
        # The same token routed at different recurrent steps yields different FFN
        # outputs -- fails if `step` is not threaded core -> block -> sublayer.
        torch.manual_seed(0)
        model = RDTForCausalLM(mol_tiny_config()).eval()
        ffn = _mol_modules(model)[0]
        with torch.no_grad():
            ffn.router_alpha.fill_(1.0)
            ffn.step_embed.normal_()
            ffn.lora_b.normal_(std=0.02)
        x = torch.randn(2, 8, model.cfg.d_model)
        self.assertFalse(torch.allclose(ffn(x, step=0), ffn(x, step=2)))


class MoLStepTableSizingTest(unittest.TestCase):
    """The MoL step table must cover the deepest loop the config can be known to
    run, so deep step indices get their own ``step_embed`` row instead of all
    clamping onto the last one (a silent breadth-signal loss)."""

    def test_table_covers_random_r_max(self):
        # tiny: recurrent_steps=4. With random-r up to 8, the loop can run 8
        # steps, so the table must be 8 (not 4).
        cfg = replace(
            tiny_config(),
            use_mol=True,
            mol_step_aware=True,
            recurrent_random_r=True,
            recurrent_r_min=1,
            recurrent_r_max=8,
        )
        self.assertEqual(cfg.recurrence_step_table, 8)
        ffn = next(
            m for m in RDTForCausalLM(cfg).modules() if isinstance(m, MixtureLoRAFFN)
        )
        self.assertEqual(ffn.step_embed.shape[0], 8)

    def test_table_is_recurrent_steps_when_no_random_r(self):
        cfg = replace(tiny_config(), use_mol=True, mol_step_aware=True)
        self.assertEqual(cfg.recurrence_step_table, cfg.recurrent_steps)

    def test_table_is_act_bound_under_act(self):
        from Model.config import mor_tiny_config

        cfg = mor_tiny_config()
        self.assertEqual(cfg.recurrence_step_table, cfg.act_max_steps)

    def test_warns_once_when_steps_exceeds_table(self):
        # A fixed-mode steps override deeper than the table (e.g. a Poisson depth
        # draw, whose bound lives in TrainingConfig and is invisible here) clamps
        # silently; the core must warn exactly once.
        cfg = replace(tiny_config(), use_mol=True, mol_step_aware=True)  # table=4
        core = RecurrentCore(cfg)
        e0 = torch.randn(2, 8, cfg.d_model)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core(e0, word_pos=wp, morph_depth=md, steps=8)
            core(e0, word_pos=wp, morph_depth=md, steps=8)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 1)

    def test_no_warn_when_steps_within_table(self):
        cfg = replace(tiny_config(), use_mol=True, mol_step_aware=True)  # table=4
        core = RecurrentCore(cfg)
        e0 = torch.randn(2, 8, cfg.d_model)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core(e0, word_pos=wp, morph_depth=md, steps=4)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 0)

    def test_no_warn_without_step_aware_mol(self):
        # Plain fixed-depth model (no MoL) must never emit the clamp warning.
        cfg = tiny_config()
        core = RecurrentCore(cfg)
        e0 = torch.randn(2, 8, cfg.d_model)
        wp = torch.zeros(2, 8, dtype=torch.long)
        md = torch.zeros(2, 8, dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core(e0, word_pos=wp, morph_depth=md, steps=99)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 0)


if __name__ == "__main__":
    unittest.main()
