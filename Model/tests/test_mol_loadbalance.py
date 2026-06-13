# -*- coding: utf-8 -*-

"""Tests for MoL v2 load balancing (Switch/GShard aux loss + router z-loss).

The aux loss ``n_experts * sum_i f_i * P_i`` has theoretical minimum 1.0 (uniform
``P``) and maximum ``n_experts`` (total collapse), so the bounds below are exact,
not heuristic. Cold start (``lora_b=0``) must keep the main logits bit-identical
to plain SwiGLU regardless of the aux weights -- the aux is a pure extra training
term.
"""

import unittest
from dataclasses import replace

import torch

from Model.config import mol_tiny_config, tiny_config
from Model.layers.mixture_lora_ffn import MixtureLoRAFFN
from Model.model import RDTForCausalLM


def _mol_modules(model):
    return [m for m in model.modules() if isinstance(m, MixtureLoRAFFN)]


class AuxLossUnitTest(unittest.TestCase):
    def _ffn(self, *, step_aware=False, top_k=0, n_experts=4):
        torch.manual_seed(0)
        return MixtureLoRAFFN(
            64, 128, n_experts=n_experts, rank=8, step_table=4,
            step_aware=step_aware, top_k=top_k,
        )

    def test_aux_nonnegative_and_at_least_one(self):
        # aux = K * sum_i f_i P_i; with sum_i f_i = 1 the minimum over P is 1.0
        # (uniform P), so aux is always >= 1 and certainly non-negative.
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(2.0)
            ffn.router_proj.weight.normal_()
        ffn.reset_router_stats()
        ffn(torch.randn(8, 16, 64))
        aux, z = ffn.router_losses()
        self.assertGreaterEqual(aux.item(), 1.0 - 1e-5)
        self.assertGreaterEqual(z.item(), 0.0)

    def test_uniform_routing_hits_minimum(self):
        # router_alpha=0 -> all logits equal -> uniform P -> aux == 1.0 exactly,
        # independent of how the (tie-broken) argmax distributes the load.
        ffn = self._ffn()  # cold start: router_alpha = 0
        ffn.reset_router_stats()
        ffn(torch.randn(8, 16, 64))
        aux, _z = ffn.router_losses()
        self.assertAlmostEqual(aux.item(), 1.0, places=4)

    def test_collapsed_routing_is_large(self):
        # Route every token to expert 0 with high probability -> P concentrates on
        # the same expert the load selects -> aux approaches n_experts (the max).
        ffn = self._ffn(n_experts=4)
        with torch.no_grad():
            ffn.router_alpha.fill_(10.0)
            ffn.router_bias[:] = torch.tensor([10.0, -10.0, -10.0, -10.0])
            ffn.router_proj.weight.zero_()  # ignore input -> identical routing
        ffn.reset_router_stats()
        ffn(torch.randn(8, 16, 64))
        aux, _z = ffn.router_losses()
        self.assertGreater(aux.item(), 3.5)
        self.assertLessEqual(aux.item(), 4.0 + 1e-4)

    def test_collapsed_aux_exceeds_uniform_aux(self):
        uniform = self._ffn()  # alpha=0 -> uniform
        uniform.reset_router_stats()
        uniform(torch.randn(8, 16, 64))
        a_uniform, _ = uniform.router_losses()

        collapsed = self._ffn()
        with torch.no_grad():
            collapsed.router_alpha.fill_(10.0)
            collapsed.router_bias[:] = torch.tensor([10.0, -10.0, -10.0, -10.0])
            collapsed.router_proj.weight.zero_()
        collapsed.reset_router_stats()
        collapsed(torch.randn(8, 16, 64))
        a_collapsed, _ = collapsed.router_losses()
        self.assertGreater(a_collapsed.item(), a_uniform.item() + 0.5)

    def test_aux_gradient_pushes_toward_balance(self):
        # Descending the aux loss must *reduce* it toward the uniform minimum and
        # the router logit spread must shrink -- the whole point of load balancing.
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(3.0)
            ffn.router_bias[:] = torch.tensor([2.0, 0.5, -0.5, -2.0])
            ffn.router_proj.weight.normal_(std=0.5)
        x = torch.randn(8, 16, 64)
        params = [ffn.router_bias, ffn.router_alpha, ffn.router_proj.weight]
        opt = torch.optim.SGD(params, lr=0.5)

        ffn.reset_router_stats()
        ffn(x)
        aux0, _ = ffn.router_losses()
        first = aux0.item()
        spread0 = float(ffn.router_bias.detach().std())

        for _ in range(40):
            ffn.reset_router_stats()
            ffn(x)
            aux, _z = ffn.router_losses()
            opt.zero_grad()
            aux.backward()
            opt.step()

        ffn.reset_router_stats()
        ffn(x)
        last, _ = ffn.router_losses()
        spread1 = float(ffn.router_bias.detach().std())

        self.assertLess(last.item(), first)
        self.assertLess(spread1, spread0)  # logits flattened toward uniform

    def test_aux_grad_reaches_router_params(self):
        ffn = self._ffn(step_aware=True)
        with torch.no_grad():
            ffn.router_alpha.fill_(1.0)
            ffn.step_embed.normal_()
            ffn.router_proj.weight.normal_()
        ffn.reset_router_stats()
        ffn(torch.randn(4, 8, 64), step=1)
        aux, z = ffn.router_losses()
        (aux + z).backward()
        for p in (ffn.router_proj.weight, ffn.router_bias, ffn.router_alpha,
                  ffn.step_embed):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(ffn.router_proj.weight.grad.norm().item(), 0.0)

    def test_zloss_tracks_logit_magnitude(self):
        # z-loss = mean logsumexp(logits)^2. Larger logits -> larger z.
        small = self._ffn()
        with torch.no_grad():
            small.router_alpha.fill_(0.1)
            small.router_proj.weight.normal_(std=0.1)
        small.reset_router_stats()
        small(torch.randn(8, 16, 64))
        _a, z_small = small.router_losses()

        big = self._ffn()
        with torch.no_grad():
            big.router_alpha.fill_(8.0)
            big.router_bias[:] = torch.tensor([8.0, -8.0, -8.0, -8.0])
            big.router_proj.weight.zero_()
        big.reset_router_stats()
        big(torch.randn(8, 16, 64))
        _a2, z_big = big.router_losses()

        self.assertGreaterEqual(z_small.item(), 0.0)
        self.assertGreater(z_big.item(), z_small.item())

    def test_topk_aux_in_bounds(self):
        ffn = self._ffn(top_k=2)
        with torch.no_grad():
            ffn.router_alpha.fill_(3.0)
            ffn.router_proj.weight.normal_()
        ffn.reset_router_stats()
        ffn(torch.randn(8, 16, 64))
        aux, _z = ffn.router_losses()
        self.assertGreaterEqual(aux.item(), 1.0 - 1e-4)
        self.assertLessEqual(aux.item(), ffn.n_experts + 1e-4)

    def test_accumulates_across_calls(self):
        # The recurrent loop calls the same FFN once per step; router_losses must
        # average across those calls, not just keep the last one. Two identical
        # calls -> same mean as one call.
        ffn = self._ffn()
        with torch.no_grad():
            ffn.router_alpha.fill_(2.0)
            ffn.router_proj.weight.normal_()
        x = torch.randn(8, 16, 64)

        ffn.reset_router_stats()
        ffn(x)
        single, _ = ffn.router_losses()

        ffn.reset_router_stats()
        ffn(x)
        ffn(x)
        self.assertEqual(ffn._router_calls, 2)
        double, _ = ffn.router_losses()
        self.assertAlmostEqual(single.item(), double.item(), places=5)

    def test_router_losses_none_when_not_collected(self):
        ffn = self._ffn()
        ffn.collect_router_stats = False
        ffn.reset_router_stats()
        ffn(torch.randn(4, 8, 64))
        self.assertIsNone(ffn.router_losses())

    def test_empty_batch_no_accumulation(self):
        ffn = self._ffn()
        ffn.reset_router_stats()
        ffn(torch.randn(0, 8, 64))
        self.assertIsNone(ffn.router_losses())


class ConfigValidationTest(unittest.TestCase):
    def test_defaults(self):
        cfg = mol_tiny_config()
        self.assertEqual(cfg.mol_aux_weight, 0.01)
        self.assertEqual(cfg.mol_z_weight, 0.0)

    def test_negative_aux_weight_rejected(self):
        with self.assertRaises(ValueError):
            replace(tiny_config(), use_mol=True, mol_aux_weight=-1.0)

    def test_negative_z_weight_rejected(self):
        with self.assertRaises(ValueError):
            replace(tiny_config(), use_mol=True, mol_z_weight=-0.5)

    def test_zero_weights_allowed(self):
        cfg = replace(tiny_config(), use_mol=True, mol_aux_weight=0.0, mol_z_weight=0.0)
        self.assertEqual(cfg.mol_aux_weight, 0.0)


class ModelIntegrationTest(unittest.TestCase):
    def _cfg(self, **kw):
        return replace(mol_tiny_config(), **kw)

    def test_loss_parts_carry_aux_and_z(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(self._cfg(mol_aux_weight=0.01, mol_z_weight=0.001))
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        self.assertIn("mol_aux", out["loss_parts"])
        self.assertIn("mol_z", out["loss_parts"])
        self.assertGreaterEqual(out["loss_parts"]["mol_aux"], 1.0 - 1e-4)
        self.assertGreaterEqual(out["loss_parts"]["mol_z"], 0.0)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_aux_added_to_total_loss(self):
        # With a large aux weight the total loss must exceed the plain forward CE
        # (aux >= 1 > 0). With weight 0 they match.
        torch.manual_seed(0)
        ids = torch.randint(300, 320, (2, 16))

        m_big = RDTForCausalLM(self._cfg(mol_aux_weight=10.0, mol_z_weight=0.0)).eval()
        with torch.no_grad():
            out_big = m_big(input_ids=ids, labels=ids)
        total = float(out_big["loss"])
        fwd = out_big["loss_parts"]["forward"]
        rev = out_big["loss_parts"].get("reverse", 0.0)
        aux = out_big["loss_parts"]["mol_aux"]
        expected = fwd + m_big.cfg.reverse_loss_weight * rev + 10.0 * aux
        self.assertAlmostEqual(total, expected, places=3)

    def test_cold_start_logits_bit_identical_to_aux_off(self):
        # The aux/z terms must NOT touch the main logits: aux-on and aux-off
        # models built from the same seed produce identical logits at cold start.
        ids = torch.randint(300, 320, (2, 16))

        torch.manual_seed(0)
        m_on = RDTForCausalLM(self._cfg(mol_aux_weight=0.5, mol_z_weight=0.5)).eval()
        torch.manual_seed(0)
        m_off = RDTForCausalLM(self._cfg(mol_aux_weight=0.0, mol_z_weight=0.0)).eval()
        with torch.no_grad():
            l_on = m_on(input_ids=ids, labels=ids, return_logits=True)["logits"]
            l_off = m_off(input_ids=ids, labels=ids, return_logits=True)["logits"]
        self.assertTrue(torch.allclose(l_on, l_off, atol=1e-6))

    def test_use_mol_false_has_no_aux(self):
        # No MoL -> no aux/z keys, loss is unchanged from the plain path.
        torch.manual_seed(0)
        model = RDTForCausalLM(tiny_config())
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        self.assertNotIn("mol_aux", out["loss_parts"])
        self.assertNotIn("mol_z", out["loss_parts"])

    def test_aux_weight_zero_skips_collection(self):
        # use_mol=True but both weights 0 -> aux inactive, no stats collected, no
        # keys in loss_parts (and the FFN collection flag stays off).
        torch.manual_seed(0)
        model = RDTForCausalLM(self._cfg(mol_aux_weight=0.0, mol_z_weight=0.0))
        self.assertFalse(model._mol_aux_active)
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        self.assertNotIn("mol_aux", out["loss_parts"])
        for ffn in _mol_modules(model):
            self.assertFalse(ffn.collect_router_stats)

    def test_inference_path_does_not_collect(self):
        # labels=None (generate / pure logits): no router stats collected.
        torch.manual_seed(0)
        model = RDTForCausalLM(self._cfg(mol_aux_weight=0.01)).eval()
        ids = torch.randint(300, 320, (2, 16))
        with torch.no_grad():
            model(input_ids=ids, return_logits=True)
        for ffn in _mol_modules(model):
            self.assertFalse(ffn.collect_router_stats)
            self.assertIsNone(ffn.router_losses())

    def test_aux_grad_flows_to_router_in_full_model(self):
        # End-to-end: aux loss must produce gradients on the router params inside
        # the model. Disable lora/forward contribution comparison by checking the
        # router_proj grad is finite and nonzero after backward.
        torch.manual_seed(0)
        model = RDTForCausalLM(self._cfg(mol_aux_weight=1.0, mol_z_weight=0.1))
        # Warm the router so the aux gradient is nonzero (alpha=0 -> uniform ->
        # zero aux gradient, which is correct but uninformative for this check).
        with torch.no_grad():
            for ffn in _mol_modules(model):
                ffn.router_alpha.fill_(1.0)
                ffn.router_proj.weight.normal_(std=0.5)
        ids = torch.randint(300, 320, (2, 16))
        out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        router_grads = [
            ffn.router_proj.weight.grad for ffn in _mol_modules(model)
        ]
        self.assertTrue(all(g is not None for g in router_grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in router_grads))
        self.assertGreater(sum(g.norm().item() for g in router_grads), 0.0)

    def test_grad_ckpt_aux_matches_no_ckpt(self):
        # Load balancing must be grad-checkpoint safe: with grad_ckpt_recurrent
        # the recurrent FFN is recomputed in backward, yet the aux value and the
        # router gradient must match the non-checkpointed run exactly (no double
        # counting; the cached forward-pass tensor is the one summed into loss).
        ids = torch.randint(300, 320, (2, 16))

        def run(grad_ckpt):
            torch.manual_seed(0)
            model = RDTForCausalLM(
                self._cfg(mol_aux_weight=1.0, mol_z_weight=0.1,
                          grad_ckpt_recurrent=grad_ckpt)
            )
            with torch.no_grad():
                for ffn in _mol_modules(model):
                    ffn.router_alpha.fill_(1.0)
                    ffn.router_proj.weight.normal_(std=0.5)
            out = model(input_ids=ids, labels=ids)
            out["loss"].backward()
            aux = out["loss_parts"]["mol_aux"]
            grad = next(
                ffn.router_proj.weight.grad.clone() for ffn in _mol_modules(model)
            )
            return aux, grad

        aux_plain, g_plain = run(False)
        aux_ckpt, g_ckpt = run(True)
        self.assertAlmostEqual(aux_plain, aux_ckpt, places=4)
        self.assertTrue(torch.allclose(g_plain, g_ckpt, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
