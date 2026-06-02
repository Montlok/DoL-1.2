# -*- coding: utf-8 -*-

"""Guards for the segmented (Block-Transformer-style) causal core.

The defining contract of this core is **zero future leakage**: the logits at
position ``t`` must depend only on tokens ``<= t``. This file proves that the
full ``RDTForCausalLM`` forward through ``SegmentedCore`` is exactly causal
(perturbing any future token leaves earlier logits bit-identical), plus basic
shape / training smoke.
"""

from __future__ import annotations

import unittest

import torch

from Model.config import segmented_tiny_config
from Model.model import RDTForCausalLM
from Model.segmented import SegmentedCore


def _fixed_morph(seq_len: int, bsz: int):
    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len).contiguous()
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)
    return word_pos, morph_depth


class SegmentedSmokeTest(unittest.TestCase):
    def test_forward_backward_shapes(self) -> None:
        torch.manual_seed(0)
        cfg = segmented_tiny_config()
        model = RDTForCausalLM(cfg)

        ids = torch.randint(0, cfg.vocab_size, (2, 13))
        out = model(ids, labels=ids)

        self.assertEqual(tuple(out["logits"].shape), (2, 13, cfg.vocab_size))
        self.assertEqual(out["rec_info"]["n_segments"], 4)  # ceil(13/4)
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)

    def test_start_ctx_receives_gradient(self) -> None:
        torch.manual_seed(0)
        cfg = segmented_tiny_config()
        model = RDTForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        model(ids, labels=ids)["loss"].backward()
        self.assertIsNotNone(model.recurrent.start_ctx.grad)
        self.assertGreater(model.recurrent.start_ctx.grad.abs().sum().item(), 0.0)


class SegmentedCausalityTest(unittest.TestCase):
    """Zero-leak red line: future tokens must not affect earlier logits."""

    def _logits(self, model, ids, word_pos, morph_depth):
        with torch.no_grad():
            return model(
                ids,
                word_pos=word_pos,
                morph_depth=morph_depth,
                return_logits=True,
            )["logits"]

    def test_future_token_does_not_change_earlier_logits(self) -> None:
        torch.manual_seed(0)
        cfg = segmented_tiny_config()
        model = RDTForCausalLM(cfg).eval()

        bsz, seq_len = 1, 17  # spans >4 segments (segment_len=4)
        word_pos, morph_depth = _fixed_morph(seq_len, bsz)
        ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

        base = self._logits(model, ids, word_pos, morph_depth)

        # Perturb the LAST token: every earlier logit must be bit-identical.
        for k in (seq_len - 1, 10, 8, 5):
            mod = ids.clone()
            mod[:, k] = (mod[:, k] + 1) % cfg.vocab_size
            pert = self._logits(model, mod, word_pos, morph_depth)
            self.assertTrue(
                torch.equal(base[:, :k], pert[:, :k]),
                f"leak detected: changing token {k} altered logits < {k}",
            )

    def test_block_summary_only_feeds_next_block(self) -> None:
        """A token changed inside block ``s`` must not alter any logit whose
        target lies at or before the first token it could legitimately see."""

        torch.manual_seed(1)
        cfg = segmented_tiny_config()
        model = RDTForCausalLM(cfg).eval()

        bsz, seq_len = 2, 16
        word_pos, morph_depth = _fixed_morph(seq_len, bsz)
        ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
        base = self._logits(model, ids, word_pos, morph_depth)

        # Change a token in block 2 (positions 8..11); logits for positions
        # < 8 must be untouched (they can only see blocks 0..1).
        mod = ids.clone()
        mod[:, 9] = (mod[:, 9] + 3) % cfg.vocab_size
        pert = self._logits(model, mod, word_pos, morph_depth)
        self.assertTrue(torch.equal(base[:, :9], pert[:, :9]))


class SegmentedRandomRTest(unittest.TestCase):
    def test_random_r_samples_in_range(self) -> None:
        cfg = segmented_tiny_config()
        cfg.recurrent_random_r = True
        cfg.recurrent_r_min = 2
        cfg.recurrent_r_max = 5
        core = SegmentedCore(cfg).train()
        seen = set()
        torch.manual_seed(0)
        for _ in range(50):
            seen.add(core._resolve_steps(None))
        self.assertTrue(seen.issubset({2, 3, 4, 5}))
        self.assertGreater(len(seen), 1)

    def test_eval_uses_fixed_steps(self) -> None:
        cfg = segmented_tiny_config()
        cfg.recurrent_random_r = True
        core = SegmentedCore(cfg).eval()
        self.assertEqual(core._resolve_steps(None), cfg.recurrent_steps)


if __name__ == "__main__":
    unittest.main()
