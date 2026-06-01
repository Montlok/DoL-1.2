# -*- coding: utf-8 -*-

"""Tests for the HuggingFace compatibility adapter (Model/hf)."""

import tempfile
import unittest

import torch

from Model.config import RDTConfig
from Model.hf import RDTForCausalLMHF, RDTHFConfig


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


class HFConfigTest(unittest.TestCase):
    def test_roundtrip_config_payload(self):
        cfg = _cfg()
        hf_cfg = RDTHFConfig.from_rdt_config(cfg)
        rebuilt = hf_cfg.to_rdt_config()
        self.assertEqual(rebuilt, cfg)

    def test_mirrors_common_hf_attrs(self):
        hf_cfg = RDTHFConfig.from_rdt_config(_cfg())
        self.assertEqual(hf_cfg.vocab_size, _cfg().vocab_size)
        self.assertEqual(hf_cfg.hidden_size, _cfg().d_model)
        self.assertEqual(hf_cfg.pad_token_id, _cfg().pad_id)

    def test_config_json_roundtrip(self):
        hf_cfg = RDTHFConfig.from_rdt_config(_cfg())
        with tempfile.TemporaryDirectory() as d:
            hf_cfg.save_pretrained(d)
            reloaded = RDTHFConfig.from_pretrained(d)
        self.assertEqual(reloaded.to_rdt_config(), _cfg())


class HFModelTest(unittest.TestCase):
    def _build(self):
        torch.manual_seed(0)
        hf_cfg = RDTHFConfig.from_rdt_config(_cfg())
        model = RDTForCausalLMHF(hf_cfg).eval()
        return model

    def _prompt(self):
        cfg = _cfg()
        return torch.tensor([[cfg.bos_id, 300, 301], [cfg.bos_id, 302, 303]])

    def test_forward_matches_inner_logits(self):
        model = self._build()
        ids = self._prompt()
        with torch.no_grad():
            hf_out = model(ids).logits
            inner = model.rdt(ids, return_logits=True)["logits"]
        self.assertTrue(torch.allclose(hf_out, inner))

    def test_forward_computes_masked_loss(self):
        model = self._build()
        ids = self._prompt()
        labels = ids.clone()
        labels[:, :2] = _cfg().ignore_index
        out = model(ids, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertTrue(torch.isfinite(out.loss))

    def test_embeddings_accessors(self):
        model = self._build()
        self.assertIs(model.get_input_embeddings(), model.rdt.embed)
        self.assertIs(model.get_output_embeddings(), model.rdt.lm_head)

    def test_greedy_generate_matches_native(self):
        model = self._build()
        ids = self._prompt()
        cfg = _cfg()
        native = model.rdt.generate(ids, max_new_tokens=5, greedy=True)
        hf = model.generate(
            ids,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            pad_token_id=cfg.pad_id,
        )
        self.assertTrue(torch.equal(native, hf))

    def test_save_and_reload_preserves_logits(self):
        model = self._build()
        ids = self._prompt()
        with torch.no_grad():
            before = model(ids).logits
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            reloaded = RDTForCausalLMHF.from_pretrained(d).eval()
        with torch.no_grad():
            after = reloaded(ids).logits
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_resize_token_embeddings(self):
        model = self._build()
        new_vocab = _cfg().vocab_size + 8
        model.resize_token_embeddings(new_vocab)
        self.assertEqual(model.get_input_embeddings().weight.shape[0], new_vocab)


if __name__ == "__main__":
    unittest.main()
