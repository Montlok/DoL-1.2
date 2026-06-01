# -*- coding: utf-8 -*-

"""Tests for the SFT chat template + masked example builder.

The central contract (test①): loss is computed on assistant content + EOS
only; system/user/tool messages and role headers are masked. A deterministic
character-level encoder makes the masking exactly checkable.
"""

import json
import tempfile
import unittest

import torch

from Model.config import IGNORE_INDEX, RDTConfig
from Model.model import RDTForCausalLM
from Model.posttrain.chat_template import (
    ROLE_SENTINELS,
    SFT_TEMPLATE_VERSION,
    render_text,
)
from Model.posttrain.sft_data import (
    SFTChatDataset,
    build_sft_example,
    generation_prompt_ids,
)

EOS = 2
BOS = 1


def _encode(text: str) -> list[int]:
    # Deterministic, invertible, and never collides with EOS/BOS sentinels.
    return [ord(c) + 1000 for c in text]


def _supervised_text(example: dict) -> str:
    chars = [
        chr(tok - 1000)
        for tok, lab in zip(example["input_ids"], example["labels"])
        if lab != IGNORE_INDEX and tok >= 1000
    ]
    return "".join(chars)


class TemplateTest(unittest.TestCase):
    def test_version_pinned(self):
        self.assertEqual(SFT_TEMPLATE_VERSION, "v1")

    def test_render_text_orders_roles(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        text = render_text(msgs)
        self.assertIn(ROLE_SENTINELS["user"], text)
        self.assertTrue(text.index("hi") < text.index("yo"))


class SFTMaskingTest(unittest.TestCase):
    def test_only_assistant_supervised(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "ANSWER"},
        ]
        ex = build_sft_example(msgs, _encode, eos_id=EOS, bos_id=BOS)
        self.assertEqual(len(ex["input_ids"]), len(ex["labels"]))
        self.assertEqual(len(ex["input_ids"]), len(ex["attention_mask"]))
        # Supervised text is exactly the assistant content (EOS is non-char).
        self.assertEqual(_supervised_text(ex), "ANSWER")
        # System/user content never appears in supervised positions.
        self.assertNotIn("SYS", _supervised_text(ex))
        self.assertNotIn("QUESTION", _supervised_text(ex))

    def test_eos_supervised_after_assistant(self):
        msgs = [{"role": "assistant", "content": "x"}]
        ex = build_sft_example(msgs, _encode, eos_id=EOS)
        self.assertEqual(ex["input_ids"][-1], EOS)
        self.assertEqual(ex["labels"][-1], EOS)

    def test_bos_is_masked(self):
        msgs = [{"role": "assistant", "content": "x"}]
        ex = build_sft_example(msgs, _encode, eos_id=EOS, bos_id=BOS)
        self.assertEqual(ex["input_ids"][0], BOS)
        self.assertEqual(ex["labels"][0], IGNORE_INDEX)

    def test_labels_not_preshifted(self):
        msgs = [{"role": "assistant", "content": "abc"}]
        ex = build_sft_example(msgs, _encode, eos_id=EOS)
        for tok, lab in zip(ex["input_ids"], ex["labels"]):
            if lab != IGNORE_INDEX:
                self.assertEqual(tok, lab)

    def test_tool_result_is_masked(self):
        msgs = [
            {"role": "user", "content": "compute"},
            {"role": "assistant", "content": "<tool_call>add</tool_call>"},
            {"role": "tool", "content": "TOOLRESULT"},
            {"role": "assistant", "content": "done"},
        ]
        ex = build_sft_example(msgs, _encode, eos_id=EOS)
        sup = _supervised_text(ex)
        self.assertNotIn("TOOLRESULT", sup)
        # The model-emitted tool call and final answer ARE supervised.
        self.assertIn("<tool_call>add</tool_call>", sup)
        self.assertIn("done", sup)

    def test_multiturn_masks_all_user_turns(self):
        msgs = [
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
        ]
        ex = build_sft_example(msgs, _encode, eos_id=EOS)
        sup = _supervised_text(ex)
        self.assertEqual(sup, "A1A2")

    def test_generation_prompt_ends_with_assistant_header(self):
        msgs = [{"role": "user", "content": "hi"}]
        ids = generation_prompt_ids(msgs, _encode, bos_id=BOS)
        tail = ids[-len(_encode(ROLE_SENTINELS["assistant"])):]
        self.assertEqual(tail, _encode(ROLE_SENTINELS["assistant"]))

    def test_dataset_reads_jsonl(self):
        rows = [
            {"messages": [{"role": "user", "content": "a"},
                          {"role": "assistant", "content": "b"}]},
            {"messages": [{"role": "user", "content": "c"},
                          {"role": "assistant", "content": "d"}]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            path = fh.name
        ds = SFTChatDataset(path, _encode, eos_id=EOS, bos_id=BOS)
        self.assertEqual(len(ds), 2)
        self.assertEqual(_supervised_text(ds[1]), "d")


class ReverseLossToggleTest(unittest.TestCase):
    def _cfg(self):
        return RDTConfig(
            d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8,
            rope_head_dim=4, nope_head_dim=4, ffn_hidden=64, ffn_multiple=32,
            n_prelude=1, n_coda=1, mamba_per_block=1, attn_per_block=1,
            recurrent_steps=2, mamba_d_state=8, mamba_expand=2, mamba_headdim=16,
            use_official_mamba=False, max_seq_len=16, bidirectional=True,
        )

    def test_disabling_reverse_loss_drops_term(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(self._cfg())
        ids = torch.randint(300, model.cfg.vocab_size, (2, 6))
        labels = ids.clone()

        on = model(ids, labels=labels)
        self.assertIn("reverse", on["loss_parts"])

        model.reverse_loss_enabled = False
        off = model(ids, labels=labels)
        self.assertNotIn("reverse", off["loss_parts"])
        # Forward (causal) term is unchanged; total loss differs.
        self.assertAlmostEqual(
            on["loss_parts"]["forward"], off["loss_parts"]["forward"], places=5
        )
        self.assertFalse(torch.allclose(on["loss"], off["loss"]))


if __name__ == "__main__":
    unittest.main()
