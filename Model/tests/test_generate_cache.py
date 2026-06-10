# -*- coding: utf-8 -*-

"""Cached incremental decoding must match full re-forward exactly."""

from __future__ import annotations

import pytest
import torch

from Model.cache import RDTCache
from Model.config import tiny_config
from Model.layers.mamba3_layer import NaiveSSM
from Model.model import RDTForCausalLM


def _cfg(**overrides):
    cfg = tiny_config()
    for key, value in overrides.items():
        object.__setattr__(cfg, key, value)
    return cfg


def _model(**overrides) -> RDTForCausalLM:
    torch.manual_seed(0)
    model = RDTForCausalLM(_cfg(**overrides))
    model.eval()
    return model


def _ids(bsz: int, length: int, vocab: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(260, vocab, (bsz, length), generator=gen)


class TestNaiveSSMIncremental:
    def test_chunked_matches_full(self):
        torch.manual_seed(0)
        ssm = NaiveSSM(d_model=64, d_state=8, expand=2, headdim=16, d_conv=4)
        ssm.eval()
        x = torch.randn(2, 12, 64)

        full = ssm(x)

        state = None
        outs = []
        for chunk in x.split([5, 1, 6], dim=1):
            y, state = ssm(chunk, state=state, return_state=True)
            outs.append(y)
        inc = torch.cat(outs, dim=1)

        assert torch.allclose(full, inc, atol=1e-5)

    def test_rejects_padded_mask(self):
        ssm = NaiveSSM(d_model=32, d_state=4, expand=2, headdim=8)
        x = torch.randn(1, 4, 32)
        mask = torch.tensor([[1, 1, 0, 0]])
        with pytest.raises(ValueError):
            ssm(x, attn_mask=mask, return_state=True)


class TestCachedForward:
    def test_prefill_matches_full_forward(self):
        model = _model()
        ids = _ids(2, 10, model.cfg.vocab_size)

        full = model(ids)["logits"]
        cached = model(ids, cache=RDTCache())["logits"]

        assert torch.allclose(full, cached, atol=1e-5)

    def test_decode_steps_match_full_forward(self):
        model = _model()
        ids = _ids(1, 9, model.cfg.vocab_size, seed=1)

        cache = RDTCache()
        prefill = model(ids[:, :6], cache=cache)["logits"]
        outs = [prefill]
        for t in range(6, 9):
            ones = torch.ones_like(ids[:, : t + 1])
            wp, md = model._default_morph_info(ids[:, : t + 1], ones)
            step = model(
                ids[:, t : t + 1],
                word_pos=wp[:, -1:],
                morph_depth=md[:, -1:],
                cache=cache,
            )["logits"]
            outs.append(step)
        inc = torch.cat(outs, dim=1)

        full = model(ids)["logits"]
        assert torch.allclose(full, inc, atol=1e-4)

    def test_decode_requires_explicit_positions(self):
        model = _model()
        ids = _ids(1, 5, model.cfg.vocab_size)
        cache = RDTCache()
        model(ids, cache=cache)
        with pytest.raises(ValueError, match="word_pos"):
            model(ids[:, -1:], cache=cache)

    def test_rejects_padded_mask(self):
        model = _model()
        ids = _ids(1, 5, model.cfg.vocab_size)
        mask = torch.ones_like(ids)
        mask[0, 0] = 0
        with pytest.raises(ValueError, match="all-ones"):
            model(ids, attention_mask=mask, cache=RDTCache())

    def test_max_seq_len_includes_past(self):
        model = _model()
        ids = _ids(1, model.cfg.max_seq_len, model.cfg.vocab_size)
        cache = RDTCache()
        model(ids, cache=cache)
        ones = torch.ones(1, 1, dtype=torch.long)
        with pytest.raises(ValueError, match="max_seq_len"):
            model(ids[:, -1:], word_pos=ones, morph_depth=ones, cache=cache)


class TestGenerate:
    def test_greedy_matches_full_reforward(self):
        model = _model()
        ids = _ids(2, 7, model.cfg.vocab_size, seed=2)

        fast = model.generate(ids, max_new_tokens=6, eos_id=-1)

        slow = ids
        for _ in range(6):
            logits = model(slow)["logits"][:, -1].float()
            slow = torch.cat([slow, logits.argmax(-1, keepdim=True)], dim=1)

        assert torch.equal(fast, slow)

    def test_boundary_tokens_match_full_reforward(self, monkeypatch):
        """Decode steps that emit word/morpheme-boundary or other special
        tokens must keep the incremental morph state identical to a full
        re-derivation over the whole sequence."""

        model = _model()
        cfg = model.cfg
        # prompt mixing boundaries: wb, content, mb, other-special, content
        ids = torch.tensor(
            [[cfg.word_boundary_id, 300, cfg.morpheme_boundary_id, 20, 301],
             [400, cfg.word_boundary_id, cfg.morpheme_boundary_id, 401, 20]]
        )
        forced = [
            cfg.word_boundary_id,
            cfg.morpheme_boundary_id,
            cfg.morpheme_boundary_id,
            20,  # other special: resets depth, keeps word_pos
            350,
            cfg.word_boundary_id,
        ]
        n_free = 4

        orig_sample = RDTForCausalLM._sample_token
        calls = {"n": 0}

        def forced_sample(logits, temperature, top_k):
            i = calls["n"]
            calls["n"] += 1
            if i < len(forced):
                return torch.full(
                    (logits.shape[0],), forced[i], dtype=torch.long
                )
            return orig_sample(logits, temperature, top_k)

        monkeypatch.setattr(
            RDTForCausalLM, "_sample_token", staticmethod(forced_sample)
        )
        fast = model.generate(ids, max_new_tokens=len(forced) + n_free, eos_id=-1)

        slow = ids
        for i in range(len(forced) + n_free):
            logits = model(slow)["logits"][:, -1].float()
            if i < len(forced):
                nxt = torch.full((slow.shape[0], 1), forced[i], dtype=torch.long)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            slow = torch.cat([slow, nxt], dim=1)

        assert torch.equal(fast, slow)

    def test_steps_override_matches_full_reforward(self):
        model = _model()
        ids = _ids(1, 5, model.cfg.vocab_size, seed=3)

        fast = model.generate(ids, max_new_tokens=4, steps=2, eos_id=-1)

        slow = ids
        for _ in range(4):
            logits = model(slow, steps=2)["logits"][:, -1].float()
            slow = torch.cat([slow, logits.argmax(-1, keepdim=True)], dim=1)

        assert torch.equal(fast, slow)

    def test_eos_stops_and_pads(self):
        model = _model()
        ids = _ids(2, 4, model.cfg.vocab_size, seed=4)
        first = model.generate(ids, max_new_tokens=8, eos_id=-1)[:, 4]

        eos = int(first[0])
        out = model.generate(ids, max_new_tokens=8, eos_id=eos)
        row = out[0, 4:]
        assert row[0] == eos
        assert (row[1:] == eos).all() or row.shape[0] == 1

    def test_rejects_act(self):
        model = _model(use_act=True, act_max_steps=4)
        ids = _ids(1, 4, model.cfg.vocab_size)
        with pytest.raises(NotImplementedError):
            model.generate(ids, max_new_tokens=2)

    def test_rejects_padded_prompt(self):
        model = _model()
        ids = _ids(1, 4, model.cfg.vocab_size)
        mask = torch.ones_like(ids)
        mask[0, 0] = 0
        with pytest.raises(ValueError):
            model.generate(ids, max_new_tokens=2, attention_mask=mask)

    def test_budget_check(self):
        model = _model()
        ids = _ids(1, model.cfg.max_seq_len - 1, model.cfg.vocab_size)
        with pytest.raises(ValueError, match="max_seq_len"):
            model.generate(ids, max_new_tokens=2)

    def test_sampling_path_runs(self):
        model = _model()
        ids = _ids(1, 5, model.cfg.vocab_size, seed=5)
        torch.manual_seed(0)
        out = model.generate(
            ids, max_new_tokens=3, temperature=0.8, top_k=10, eos_id=-1
        )
        assert out.shape == (1, 8)
