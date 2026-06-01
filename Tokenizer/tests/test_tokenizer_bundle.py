# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.unified.bundle import TokenizerBundle


class TokenizerBundleTest(unittest.TestCase):
    def _train_tiny_morphbpe(self, tmp: str) -> str:
        trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
        tokenizer = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
        path = os.path.join(tmp, "tiny_morphbpe.json")
        tokenizer.save(path)
        return path

    def test_bundle_save_load_encode_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            morphbpe_path = self._train_tiny_morphbpe(tmp)
            bundle = TokenizerBundle.from_files(morphbpe_path)
            out_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(out_dir)

            self.assertTrue(os.path.exists(os.path.join(out_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "morphbpe.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "general.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "vocab.json")))

            loaded = TokenizerBundle.from_dir(out_dir)
            self.assertEqual(loaded.validate(), [])

            encoded = loaded.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 test", add_bos=True, add_eos=True)
            self.assertEqual(len(encoded.input_ids), len(encoded.tokens))
            self.assertGreater(len(encoded.input_ids), 4)
            self.assertEqual(encoded.tokens[0].token, "<bos>")
            self.assertEqual(encoded.tokens[-1].token, "<eos>")

            mm = loaded.encode_multimodal(
                "文字 <image> test",
                images=["img"],
                image_sizes=[(14, 14)],
            )
            self.assertEqual(len(mm.image_token_spans), 1)
            start, end = mm.image_token_spans[0]
            self.assertEqual([tok.token for tok in mm.tokens[start:end]], [
                "<image_start>",
                "<image_patch>",
                "<image_end>",
            ])
            self.assertEqual(len(mm.attention_mask), len(mm.input_ids))


if __name__ == "__main__":
    unittest.main()
