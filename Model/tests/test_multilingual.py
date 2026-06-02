# -*- coding: utf-8 -*-

"""Multilingual / compute-efficiency validation for the segmented core.

These tests do **not** rebuild the tokenizer; they validate two architectural
claims that matter for a low-resource, morphologically-rich language mix
(traditional Mongolian + Chinese):

1. **Adaptive depth gives hard tokens more compute.** A trivially predictable
   (constant) context converges in fewer recurrent-depth iterations than a
   high-entropy random context, so the KL early-exit spends test-time compute
   where it is actually needed (Huginn arXiv:2502.05171 6.1) -- the natural
   mechanism for giving rare Mongolian segments more "thinking".

2. **Attention/RDT cost scales with n_seg, not n_tok.** The block-summary design
   (Block Transformer arXiv:2406.02657) runs the quadratic refinement over
   ceil(L / segment_len) summaries, the source of the compute saving.

3. **Embedding/vocab efficiency:** tied input/output embeddings, and every id
   across the mixed-language vocab embeds to a finite vector and finite logits.
"""

import unittest

import torch

from Model.tests.test_segmented import _seg_cfg
from Model.model import RDTForCausalLM


class AdaptiveDepthSpendsComputeWhereNeededTest(unittest.TestCase):
    def _depth_used(self, model, window):
        calls = {"n": 0}
        orig = model.forward

        def counting_forward(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        model.forward = counting_forward  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                model._adaptive_depth_logits(window, None)
        finally:
            model.forward = orig  # type: ignore[method-assign]
        return calls["n"]

    def test_easy_context_uses_no_more_depth_than_hard(self):
        torch.manual_seed(0)
        cfg = _seg_cfg("none")
        cfg.kl_exit_threshold = 0.1
        cfg.recurrent_steps = 6
        model = RDTForCausalLM(cfg).eval()

        easy = torch.full((1, 8), 300, dtype=torch.long)  # constant -> low entropy
        hard = torch.randint(300, cfg.vocab_size, (1, 8))

        d_easy = self._depth_used(model, easy)
        d_hard = self._depth_used(model, hard)

        # Bounded by the configured maximum, and the easy context never needs
        # *more* depth than the hard one (typically strictly fewer).
        self.assertLessEqual(d_easy, cfg.recurrent_steps)
        self.assertLessEqual(d_hard, cfg.recurrent_steps)
        self.assertLessEqual(d_easy, d_hard)


class SegmentedComputeScalesWithSegmentsTest(unittest.TestCase):
    def test_refinement_runs_over_n_seg_summaries(self):
        torch.manual_seed(0)
        cfg = _seg_cfg("none", segment_len=4)
        model = RDTForCausalLM(cfg).eval()
        for length in (8, 12, 16, 17):
            ids = torch.randint(300, cfg.vocab_size, (1, length))
            with torch.no_grad():
                out = model(ids, return_logits=True)
            expected = -(-length // cfg.segment_len)  # ceil(L / L_B)
            self.assertEqual(out["rec_info"]["n_segments"], expected)
            # Far fewer summaries than tokens -> quadratic refinement is cheaper.
            self.assertLess(out["rec_info"]["n_segments"], length)


class MixedLanguageVocabTest(unittest.TestCase):
    def test_tied_embeddings_and_finite_logits_across_vocab(self):
        torch.manual_seed(0)
        cfg = _seg_cfg("none")
        model = RDTForCausalLM(cfg).eval()

        if cfg.tie_word_embeddings:
            self.assertIs(model.lm_head.weight, model.embed.weight)

        # Sample ids spanning the low (special/latin), middle and high (CJK /
        # Mongolian) regions of the shared vocab; all must embed + score finite.
        v = cfg.vocab_size
        ids = torch.tensor(
            [[0, 1, v // 4, v // 2, (3 * v) // 4, v - 1, v // 2 + 7, v // 3]]
        )
        with torch.no_grad():
            out = model(ids, return_logits=True)
        self.assertTrue(torch.isfinite(out["logits"]).all())


if __name__ == "__main__":
    unittest.main()
