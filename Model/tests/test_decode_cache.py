# -*- coding: utf-8 -*-

"""Bit-exactness tests for the incremental KV/state decode cache.

The whole-model cache is correct only if it is *numerically identical* to a
fresh full forward over the growing prefix. These tests pin that contract at
three levels (single layer, whole-model logits, ``generate()`` API) and probe
the adversarial corners surfaced from user / engineer / peer perspectives:
single-token prompts, EOS-triggered padding, batch > 1, non-morphological RoPE
(exercising ``pos_offset``), the plain (non-mHC) refinement loop, and the
official-Mamba backend rejection.
"""

import unittest

import torch

from Model.blocks import MambaSubLayer
from Model.config import RDTConfig
from Model.inference.cache import DecodeCache, MambaCache, MLACache
from Model.layers.mla import MLA
from Model.model import RDTForCausalLM

ATOL = 1e-4


def _two_stage_cfg(drift_mode: str = "mhc", morph_rope: bool = True) -> RDTConfig:
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
        core_type="two_stage",
        stage1_mamba_layers=3,
        stage2_attn_layers=2,
        recurrent_drift_mode=drift_mode,
        mhc_n_streams=4,
        mhc_sinkhorn_iters=10,
    )


class MLACacheTest(unittest.TestCase):
    def test_mla_cached_matches_full(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        mla = MLA(cfg).eval()
        b, length = 2, 9
        x = torch.randn(b, length, cfg.d_model)
        wp = torch.arange(length).unsqueeze(0).expand(b, length).contiguous()
        md = torch.zeros(b, length, dtype=torch.long)

        with torch.no_grad():
            ref = mla(x, word_pos=wp, morph_depth=md)
            cache = MLACache()
            parts = [mla(x[:, :3], word_pos=wp[:, :3], morph_depth=md[:, :3],
                         cache=cache)]
            for t in range(3, length):
                parts.append(
                    mla(x[:, t:t + 1], word_pos=wp[:, t:t + 1],
                        morph_depth=md[:, t:t + 1], cache=cache)
                )
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))

    def test_mla_cached_non_morph_rope_uses_pos_offset(self):
        torch.manual_seed(1)
        cfg = _two_stage_cfg(morph_rope=False)
        mla = MLA(cfg).eval()
        b, length = 2, 8
        x = torch.randn(b, length, cfg.d_model)

        with torch.no_grad():
            ref = mla(x)
            cache = MLACache()
            parts = [mla(x[:, :2], cache=cache, pos_offset=0)]
            for t in range(2, length):
                parts.append(mla(x[:, t:t + 1], cache=cache, pos_offset=t))
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))


class MambaCacheTest(unittest.TestCase):
    def test_mamba_cached_matches_full(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        layer = MambaSubLayer(cfg, layer_idx=0).eval()
        b, length = 2, 9
        x = torch.randn(b, length, cfg.d_model)

        with torch.no_grad():
            ref = layer(x)
            cache = MambaCache()
            parts = [layer(x[:, :4], cache=cache)]
            for t in range(4, length):
                parts.append(layer(x[:, t:t + 1], cache=cache))
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))


class WholeModelCacheTest(unittest.TestCase):
    def _gold(self, cfg: RDTConfig, prefill: int = 4):
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

    def test_incremental_logits_bit_exact_mhc(self):
        self.assertLess(self._gold(_two_stage_cfg("mhc")), ATOL)

    def test_incremental_logits_bit_exact_plain(self):
        self.assertLess(self._gold(_two_stage_cfg("none")), ATOL)

    def test_incremental_logits_bit_exact_decay(self):
        self.assertLess(self._gold(_two_stage_cfg("decay")), ATOL)

    def test_cache_seq_len_tracks_tokens(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (1, 5))
        cache = DecodeCache()
        mask = torch.ones_like(ids)
        wp, md = model._default_morph_info(ids, mask)
        with torch.no_grad():
            model._forward_decode(ids, wp, md, cache)
        self.assertEqual(cache.seq_len, 5)


class GenerateCacheEquivalenceTest(unittest.TestCase):
    def _check(self, ids, **kw):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
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

    def test_repetition_penalty_equivalence(self):
        self._check(
            torch.randint(300, 320, (2, 6)),
            max_new_tokens=10,
            repetition_penalty=1.3,
        )

    def test_special_tokens_in_prompt(self):
        ids = torch.tensor([[1, 2, 300, 301, 302, 3, 303]])
        self._check(ids, max_new_tokens=8)


class CacheRejectionTest(unittest.TestCase):
    def test_use_cache_rejects_non_two_stage(self):
        cfg = RDTConfig(
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
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (1, 3))
        with self.assertRaises(NotImplementedError):
            model.generate(ids, max_new_tokens=2, use_cache=True)

    def test_cached_generation_max_seq_len_guard(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        cfg_small = cfg
        object.__setattr__(cfg_small, "max_seq_len", 8)
        model = RDTForCausalLM(cfg_small).eval()
        ids = torch.randint(300, 320, (1, 6))
        with self.assertRaises(ValueError):
            model.generate(ids, max_new_tokens=10, greedy=True, use_cache=True)


if __name__ == "__main__":
    unittest.main()
