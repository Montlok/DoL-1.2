# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM


def _cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=16,
    )


class GenerateTest(unittest.TestCase):
    def _prompt(self, cfg: RDTConfig) -> torch.Tensor:
        return torch.tensor([[cfg.bos_id, 300, 301], [cfg.bos_id, 302, 303]])

    def test_generate_appends_requested_tokens(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)

        out = model.generate(prompt, max_new_tokens=5, greedy=True)

        self.assertEqual(out.shape, (2, prompt.shape[1] + 5))
        self.assertTrue(torch.equal(out[:, : prompt.shape[1]], prompt))

    def test_greedy_is_deterministic(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)

        a = model.generate(prompt, max_new_tokens=6, greedy=True)
        b = model.generate(prompt, max_new_tokens=6, greedy=True)
        self.assertTrue(torch.equal(a, b))

    def test_generate_keeps_model_in_eval_off_for_training(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        model.train()
        model.generate(self._prompt(cfg), max_new_tokens=2, greedy=True)
        self.assertTrue(model.training)

    def test_does_not_exceed_max_seq_len_window(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = torch.full((1, cfg.max_seq_len), 300, dtype=torch.long)
        prompt[0, 0] = cfg.bos_id

        out = model.generate(prompt, max_new_tokens=3, greedy=True)
        self.assertEqual(out.shape, (1, cfg.max_seq_len + 3))

    def test_eos_stops_and_pads(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)

        # Make forward deterministically favor EOS, independent of tied weights.
        def fake_forward(window, return_logits=True):
            bsz = window.shape[0]
            logits = torch.zeros(bsz, window.shape[1], cfg.vocab_size)
            logits[:, -1, cfg.eos_id] = 100.0
            return {"logits": logits}

        model.forward = fake_forward  # type: ignore[method-assign]

        out = model.generate(prompt, max_new_tokens=5, greedy=True)
        first_new = out[:, prompt.shape[1]]
        self.assertTrue(torch.all(first_new == cfg.eos_id))
        # Everything after the EOS must be pad.
        tail = out[:, prompt.shape[1] + 1:]
        self.assertTrue(torch.all(tail == cfg.pad_id))

    def test_sampling_respects_top_k_one_equals_greedy(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)

        greedy = model.generate(prompt, max_new_tokens=4, greedy=True)
        sampled = model.generate(
            prompt, max_new_tokens=4, top_k=1, temperature=1.0
        )
        self.assertTrue(torch.equal(greedy, sampled))

    def test_invalid_args(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)
        with self.assertRaises(ValueError):
            model.generate(prompt, temperature=0.0)
        with self.assertRaises(ValueError):
            model.generate(prompt, top_p=1.5)
        with self.assertRaises(ValueError):
            model.generate(prompt, min_p=1.5)
        with self.assertRaises(ValueError):
            model.generate(prompt.view(-1), max_new_tokens=1)

    def test_min_p_filter_masks_low_prob_tokens(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        # One dominant token (prob ~0.9) and a long tail; min_p=0.5 must keep
        # only tokens with prob >= 0.5 * p_max, i.e. just the dominant one.
        logits = torch.full((1, 8), -10.0)
        logits[0, 3] = 5.0
        filtered = model._filter_logits(logits, top_k=None, top_p=None, min_p=0.5)
        kept = (filtered[0] > float("-inf")).nonzero().flatten().tolist()
        self.assertEqual(kept, [3])

    def test_min_p_one_keeps_only_max(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        prompt = self._prompt(cfg)
        # min_p=1.0 keeps only the argmax token -> equivalent to greedy.
        greedy = model.generate(prompt, max_new_tokens=4, greedy=True)
        min_p = model.generate(prompt, max_new_tokens=4, min_p=1.0)
        self.assertTrue(torch.equal(greedy, min_p))


if __name__ == "__main__":
    unittest.main()
