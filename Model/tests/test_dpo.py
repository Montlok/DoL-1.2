# -*- coding: utf-8 -*-

"""Tests for DPO loss + model-driven preference scoring."""

import copy
import math
import unittest

import torch

from Model.config import RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.posttrain.dpo import DPOConfig, dpo_loss, dpo_step


def _cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8, rope_head_dim=4,
        nope_head_dim=4, ffn_hidden=64, ffn_multiple=32, n_prelude=1, n_coda=1,
        mamba_per_block=1, attn_per_block=1, recurrent_steps=2, mamba_d_state=8,
        mamba_expand=2, mamba_headdim=16, use_official_mamba=False, max_seq_len=16,
    )


class DPOLossTest(unittest.TestCase):
    def test_policy_equals_reference_is_log2(self):
        c = torch.tensor([-2.0, -3.0])
        r = torch.tensor([-5.0, -4.0])
        loss, m = dpo_loss(c, r, c, r, beta=0.1)
        self.assertAlmostEqual(float(loss), math.log(2), places=5)
        self.assertAlmostEqual(m["chosen_reward"], 0.0, places=6)
        self.assertAlmostEqual(m["rejected_reward"], 0.0, places=6)

    def test_larger_margin_lowers_loss(self):
        ref_c = torch.tensor([-3.0])
        ref_r = torch.tensor([-3.0])
        small, _ = dpo_loss(torch.tensor([-2.9]), torch.tensor([-3.1]), ref_c, ref_r)
        large, _ = dpo_loss(torch.tensor([-1.0]), torch.tensor([-5.0]), ref_c, ref_r)
        self.assertLess(float(large), float(small))

    def test_accuracy_metric(self):
        # chosen strongly preferred over rejected relative to ref.
        c = torch.tensor([-1.0, -1.0])
        r = torch.tensor([-9.0, -9.0])
        ref = torch.tensor([-5.0, -5.0])
        _, m = dpo_loss(c, r, ref, ref)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertGreater(m["reward_margin"], 0.0)


class DPOStepTest(unittest.TestCase):
    def _models(self):
        torch.manual_seed(0)
        policy = RDTForCausalLM(_cfg())
        reference = copy.deepcopy(policy).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        return policy, reference

    def _batch(self):
        cfg = _cfg()
        chosen = torch.randint(300, cfg.vocab_size, (2, 7))
        rejected = torch.randint(300, cfg.vocab_size, (2, 7))
        cmask = torch.zeros_like(chosen)
        cmask[:, 4:] = 1  # last tokens are the response
        rmask = torch.zeros_like(rejected)
        rmask[:, 4:] = 1
        return chosen, cmask, rejected, rmask

    def test_step_backprops_and_zero_init_loss(self):
        policy, reference = self._models()
        chosen, cmask, rejected, rmask = self._batch()
        cfg = DPOConfig(beta=0.1)
        loss, m = dpo_step(policy, reference, chosen, cmask, rejected, rmask, cfg)
        # Policy starts equal to reference -> loss is exactly log(2).
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=4)
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_reference_has_no_grad(self):
        policy, reference = self._models()
        chosen, cmask, rejected, rmask = self._batch()
        loss, _ = dpo_step(
            policy, reference, chosen, cmask, rejected, rmask, DPOConfig()
        )
        loss.backward()
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_length_normalization_changes_value(self):
        policy, reference = self._models()
        chosen, cmask, rejected, rmask = self._batch()
        # Make reference differ from policy so the loss is not pinned at log2.
        torch.manual_seed(1)
        for p in reference.parameters():
            p.data.add_(torch.randn_like(p) * 0.05)
        plain, _ = dpo_step(
            policy, reference, chosen, cmask, rejected, rmask,
            DPOConfig(length_normalize=False),
        )
        normed, _ = dpo_step(
            policy, reference, chosen, cmask, rejected, rmask,
            DPOConfig(length_normalize=True),
        )
        self.assertFalse(torch.allclose(plain, normed))

    def test_training_reference_helper_freezes_independent_copy(self):
        from scripts.train_dpo import _build_reference_model

        policy = RDTForCausalLM(_cfg())
        reference = _build_reference_model(
            policy,
            _cfg(),
            TrainingConfig(parallel="single", max_steps=1, warmup_steps=0),
            local_rank=0,
            device=torch.device("cpu"),
        )

        self.assertFalse(reference.training)
        self.assertFalse(reference.reverse_loss_enabled)
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        first_policy = next(policy.parameters())
        first_reference = next(reference.parameters())
        self.assertIsNot(first_policy, first_reference)
        self.assertTrue(torch.equal(first_policy, first_reference))


if __name__ == "__main__":
    unittest.main()
