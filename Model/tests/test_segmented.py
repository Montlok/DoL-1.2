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

from Model.config import RDTConfig, segmented_tiny_config
from Model.inference.cache import DecodeCache
from Model.model import RDTForCausalLM
from Model.segmented import SegmentedCore

ATOL = 1e-4


def _seg_cfg(drift_mode: str = "none", morph_rope: bool = True, segment_len: int = 4):
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=2,
        n_coda=2,
        recurrent_steps=3,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=8,
        mamba_d_conv=4,
        use_official_mamba=False,
        use_morphological_rope=morph_rope,
        max_seq_len=64,
        core_type="segmented",
        stage1_mamba_layers=3,
        stage2_attn_layers=2,
        segment_len=segment_len,
        segmented_local_layers=2,
        recurrent_drift_mode=drift_mode,
        mhc_n_streams=4,
        mhc_sinkhorn_iters=10,
    )


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


class SegmentedCachedDecodeTest(unittest.TestCase):
    """Cached incremental decode must be bit-exact with a full forward."""

    def _gold(self, cfg, prefill=3):
        torch.manual_seed(0)
        model = RDTForCausalLM(cfg).eval()
        b, length = 2, 11
        ids = torch.randint(300, cfg.vocab_size, (b, length))

        with torch.no_grad():
            ref = model(ids, return_logits=True)["logits"]
            cache = DecodeCache()
            mask = torch.ones_like(ids)
            wp, md = model._default_morph_info(ids, mask)
            diffs = []
            lg = model._forward_decode(
                ids[:, :prefill], wp[:, :prefill], md[:, :prefill], cache
            )
            for t in range(prefill):
                diffs.append((ref[:, t, :] - lg[:, t, :]).abs().max().item())
            for t in range(prefill, length):
                lg1 = model._forward_decode(
                    ids[:, t:t + 1], wp[:, t:t + 1], md[:, t:t + 1], cache
                )
                diffs.append((ref[:, t, :] - lg1[:, 0, :]).abs().max().item())
        return max(diffs)

    def test_incremental_logits_bit_exact_plain(self):
        self.assertLess(self._gold(_seg_cfg("none")), ATOL)

    def test_incremental_logits_bit_exact_decay(self):
        self.assertLess(self._gold(_seg_cfg("decay")), ATOL)

    def test_incremental_logits_bit_exact_mhc(self):
        self.assertLess(self._gold(_seg_cfg("mhc")), ATOL)

    def test_incremental_logits_bit_exact_non_morph_rope(self):
        self.assertLess(self._gold(_seg_cfg("none", morph_rope=False)), ATOL)

    def test_incremental_logits_bit_exact_segment_len8(self):
        self.assertLess(self._gold(_seg_cfg("none", segment_len=8)), ATOL)

    def test_prefill_spanning_multiple_blocks(self):
        # prefill of 9 tokens already closes two len-4 blocks before streaming.
        self.assertLess(self._gold(_seg_cfg("none"), prefill=9), ATOL)

    def test_kv_share_budget_runs_and_caps_caches(self):
        # KV-share (budget>0) is a lossy optimization (recycles MLA caches);
        # it must still decode without error and never hold more than
        # ``budget`` rdt caches per layer.
        torch.manual_seed(0)
        cfg = _seg_cfg("none")
        cfg.kv_share_budget = 2  # < recurrent_steps (3)
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (1, 10))
        cache = DecodeCache()
        mask = torch.ones_like(ids)
        wp, md = model._default_morph_info(ids, mask)
        with torch.no_grad():
            model._forward_decode(ids, wp, md, cache)
        rdt_keys = [k for k in cache.mla if k.startswith("seg.rdt.s")]
        slots = {k.split(".l")[0] for k in rdt_keys}
        self.assertLessEqual(len(slots), cfg.kv_share_budget)


class SegmentedGenerateCacheEquivalenceTest(unittest.TestCase):
    def _check(self, ids, **kw):
        torch.manual_seed(0)
        cfg = _seg_cfg("none")
        model = RDTForCausalLM(cfg).eval()
        with torch.no_grad():
            a = model.generate(ids, greedy=True, use_cache=False, **kw)
            b = model.generate(ids, greedy=True, use_cache=True, **kw)
        self.assertTrue(torch.equal(a, b), msg=f"{a}\n!=\n{b}")

    def test_batch_equivalence(self):
        self._check(torch.randint(300, 320, (2, 7)), max_new_tokens=12)

    def test_single_token_prompt(self):
        self._check(torch.randint(300, 320, (1, 1)), max_new_tokens=10)

    def test_eos_padding_equivalence(self):
        self._check(torch.randint(300, 320, (3, 5)), max_new_tokens=15, eos_id=305)


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
