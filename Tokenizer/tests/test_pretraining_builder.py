# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import tempfile
import unittest

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.pretraining import IGNORE_INDEX, PretrainingDataBuilder, pack_samples
from Tokenizer.unified.bundle import TokenizerBundle


def build_smoke_bundle(tmp: str) -> TokenizerBundle:
    trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
    morphbpe = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
    morph_path = os.path.join(tmp, "morphbpe.json")
    morphbpe.save(morph_path)
    bundle = TokenizerBundle.from_files(morph_path)
    bundle_dir = os.path.join(tmp, "bundle")
    bundle.save_dir(bundle_dir)
    return TokenizerBundle.from_dir(bundle_dir)


class PretrainingBuilderTest(unittest.TestCase):
    def test_encode_pure_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ 文字 test")
        self.assertEqual(len(sample.input_ids), len(sample.attention_mask))
        self.assertEqual(sample.labels[0], IGNORE_INDEX)
        self.assertEqual(sample.labels[1:], sample.input_ids[1:])
        self.assertEqual(len(sample.token_offsets), len(sample.input_ids))
        self.assertEqual(len(sample.word_pos), len(sample.input_ids))
        self.assertEqual(len(sample.morph_depth), len(sample.input_ids))
        self.assertEqual(sample.modality_spans["image_token_spans"], [])

    def test_morph_features_reset_at_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ test")

        space_idx = sample.input_ids.index(bundle.tokenizer.vocab["▁"])
        self.assertEqual(sample.morph_depth[space_idx], 0)
        self.assertLess(sample.word_pos[space_idx - 1], sample.word_pos[space_idx + 1])

    def test_encode_image_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
        self.assertEqual(len(sample.modality_spans["image_token_spans"]), 1)
        self.assertEqual(sample.metadata["images"], ["x.jpg"])
        start, end = sample.modality_spans["image_token_spans"][0]
        self.assertTrue(
            all(label == IGNORE_INDEX for label in sample.labels[start:end])
        )
        self.assertEqual(sample.labels[0], IGNORE_INDEX)
        self.assertGreater(
            sum(1 for label in sample.labels if label != IGNORE_INDEX),
            0,
        )

    def test_ocr_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "ocr",
                    "text": "<image> text",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                    "ocr": [{"text": "hello", "bbox": [0, 0, 1, 1]}],
                }
            )
        self.assertEqual(sample.metadata["type"], "ocr")
        self.assertEqual(sample.metadata["ocr"][0]["text"], "hello")

    def test_flat_ocr_labels_are_normalized_for_single_image_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "ocr",
                    "text": "<image> text",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                    "ocr_labels": [1, 2, 3],
                    "reading_order": [0, 1, 2],
                }
            )
        self.assertEqual(sample.ocr_labels, [[1, 2, 3]])
        self.assertEqual(sample.reading_order, [[0, 1, 2]])

    def test_structural_special_labels_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_text("<ocr> hello")
        self.assertEqual(sample.labels[0], IGNORE_INDEX)

    def test_empty_label_ignore_tokens_disables_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            default_builder = PretrainingDataBuilder(bundle, max_length=128)
            default_sample = default_builder.encode_text("<ocr> hello")
            builder = PretrainingDataBuilder(
                bundle, max_length=128, label_ignore_tokens=set()
            )
            sample = builder.encode_text("<ocr> hello")
        # Default behaviour masks at least one structural token (e.g. <bos>,
        # <ocr>). With an empty ignore set, no token should be masked.
        self.assertIn(IGNORE_INDEX, default_sample.labels)
        self.assertNotIn(IGNORE_INDEX, sample.labels)

    def test_truncation_does_not_cut_image_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=3)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "<image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
        self.assertLessEqual(len(sample.input_ids), 3)
        self.assertEqual(sample.modality_spans["image_token_spans"], [])
        self.assertEqual(sample.images, [])
        self.assertEqual(sample.image_sizes, [])
        self.assertTrue(sample.metadata["truncated"])

    def test_pack_samples_for_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=64, add_bos=False, add_eos=False
            )
            first = builder.encode_text("hello")
            second = builder.encode_text("test")
            packed = pack_samples(
                [first, second],
                max_length=64,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].metadata["num_samples"], 2)
        self.assertLessEqual(len(packed[0].input_ids), 64)
        self.assertEqual(len(packed[0].labels), len(packed[0].attention_mask))
        self.assertEqual(len(packed[0].word_pos), len(packed[0].input_ids))
        self.assertGreater(max(packed[0].word_pos), max(first.word_pos))

    def test_pack_samples_can_pad_to_max_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=64, add_bos=False, add_eos=False
            )
            sample = builder.encode_text("hello")
            packed = pack_samples(
                [sample],
                max_length=8,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
                pad_to_max_length=True,
            )
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0].input_ids), 8)
        self.assertEqual(packed[0].attention_mask[-1], 0)
        self.assertEqual(packed[0].labels[-1], IGNORE_INDEX)
        self.assertEqual(packed[0].word_pos[-1], 0)
        self.assertEqual(packed[0].morph_depth[-1], 0)

    def test_pack_samples_trims_modality_spans_to_sequence_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            image_sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "<image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
            packed = pack_samples(
                [image_sample],
                max_length=3,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )

        self.assertEqual(packed, [])

    def test_pack_trim_does_not_leave_partial_image_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            image_sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "a <image> b",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
            packed = pack_samples(
                [image_sample],
                max_length=3,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )
        forbidden = {
            bundle.tokenizer.vocab["<image_start>"],
            bundle.tokenizer.vocab["<image_patch>"],
            bundle.tokenizer.vocab["<image_end>"],
        }
        self.assertEqual(len(packed), 1)
        self.assertTrue(forbidden.isdisjoint(packed[0].input_ids))
        self.assertEqual(packed[0].modality_spans["image_token_spans"], [])

    def test_build_pretraining_data_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            out = os.path.join(tmp, "out.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "image_text",
                            "text": "文字 <image>",
                            "images": ["x.jpg"],
                            "image_sizes": [[14, 14]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tokenizer.tools.build_pretraining_data",
                    "--tokenizer-bundle",
                    bundle_dir,
                    "--input",
                    inp,
                    "--output",
                    out,
                    "--max-length",
                    "128",
                    "--pack",
                    "--pack-max-length",
                    "16",
                    "--pad-to-max-length",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            with open(out, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
        self.assertEqual(summary["num_samples"], 1)
        self.assertGreater(summary["supervised_tokens"], 0)
        self.assertIn("input_ids", row)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertEqual(len(row["input_ids"]), len(row["word_pos"]))
        self.assertEqual(len(row["input_ids"]), len(row["morph_depth"]))
        self.assertEqual(len(row["input_ids"]), 16)
        self.assertEqual(row["labels"][-1], IGNORE_INDEX)


if __name__ == "__main__":
    unittest.main()
