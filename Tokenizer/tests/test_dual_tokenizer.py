# -*- coding: utf-8 -*-

import unittest

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.multimodal import (
    IMAGE_END,
    IMAGE_PATCH,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    expand_image_placeholders,
)
from Tokenizer.traditional_mongolian import NNBSP
from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    SEGMENT,
    SPECIAL_TOKENS,
    build_unified_vocab,
    segment_by_language,
)


class FakeMorphBPE:
    vocab = {"ᠮᠣᠩᠭᠣᠯ": 0, "ᠪᠢᠴᠢᠭ": 1}

    def encode(self, text):
        return [self.vocab[text]]


def build_fake_tokenizer():
    general = GeneralBPEModel.minimal()
    vocab = build_unified_vocab(
        morphbpe_vocab=FakeMorphBPE.vocab,
        general_vocab=general.get_vocab(),
    )
    return DualTrackTokenizer(vocab, FakeMorphBPE(), general)


class DualTokenizerTest(unittest.TestCase):
    def test_segments_special_tokens_before_language_routing(self):
        spans = segment_by_language("这张图 " + IMAGE_PLACEHOLDER + " test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("general", "这张图"),
                ("space", " "),
                ("special", IMAGE_PLACEHOLDER),
                ("space", " "),
                ("general", "test"),
            ],
        )

    def test_fullwidth_latin_and_digits_route_to_general(self):
        spans = segment_by_language("Ａ３！test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [("general", "Ａ３！test")],
        )

    def test_mongolian_nnbsp_stays_inside_mongolian_span(self):
        text = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        spans = segment_by_language(text)
        self.assertEqual([(span.lang, span.text) for span in spans], [("mn", text)])

        latin = "hello" + NNBSP + "world"
        self.assertEqual(
            [(span.lang, span.text) for span in segment_by_language(latin)],
            [("general", "hello"), ("space", NNBSP), ("general", "world")],
        )

    def test_cjk_punctuation_routes_to_general(self):
        spans = segment_by_language("这。图")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [("general", "这。图")],
        )

    def test_mongolian_punctuation_routes_to_general(self):
        spans = segment_by_language("ᠰᠠᠢᠨ᠃")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("mn", "ᠰᠠᠢᠨ"),
                ("general", "᠃"),
            ],
        )

    def test_encode_routes_tracks_to_global_id_ranges(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans(
            "ᠮᠣᠩᠭᠣᠯ 这 test!", add_bos=True, add_eos=True
        )
        self.assertEqual(result.ids[0], SPECIAL_TOKENS["<bos>"])
        self.assertEqual(result.ids[-1], SPECIAL_TOKENS["<eos>"])
        mn_lo, mn_hi = SEGMENT["mongolian"]
        gen_lo, gen_hi = SEGMENT["general"]
        self.assertTrue(any(mn_lo <= i < mn_hi for i in result.ids))
        self.assertTrue(any(gen_lo <= i < gen_hi for i in result.ids))
        self.assertEqual(result.input_ids, result.ids)
        self.assertEqual(len(result.tokens), len(result.ids))
        self.assertEqual(result.tokens[0].start, -1)
        self.assertEqual(result.tokens[-1].end, -1)

    def test_multimodal_placeholder_and_patch_tokens_are_special(self):
        tokenizer = build_fake_tokenizer()
        expanded = expand_image_placeholders(IMAGE_PLACEHOLDER, patches_per_image=2)
        self.assertEqual(expanded, IMAGE_START + IMAGE_PATCH + IMAGE_PATCH + IMAGE_END)
        ids = tokenizer.encode(expanded)
        self.assertEqual(
            ids,
            [
                SPECIAL_TOKENS[IMAGE_START],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_END],
            ],
        )

    def test_emoji_round_trip(self):
        tokenizer = build_fake_tokenizer()
        ids = tokenizer.encode("🙂")
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids), "🙂")

    def test_mixed_script_round_trip(self):
        tokenizer = build_fake_tokenizer()
        text = "mixed 中 ᠮᠣᠩᠭᠣᠯ 🙂 日本語"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_token_level_offsets_cover_tracks(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("这 test 🙂")
        tracks = {tok.track for tok in result.tokens}
        self.assertIn("general", tracks)
        self.assertIn("space", tracks)
        # offsets are monotonic and within bounds
        for tok in result.tokens:
            self.assertLessEqual(tok.start, tok.end)

    def test_mongolian_punctuation_round_trips_without_unk(self):
        tokenizer = build_fake_tokenizer()
        ids = tokenizer.encode("᠃")
        self.assertNotIn(tokenizer.unk_id, ids)
        self.assertEqual(tokenizer.decode(ids), "᠃")

    def test_newline_and_tab_are_preserved_distinctly(self):
        # Newline/tab/CR must NOT collapse into the space token "▁". They route
        # to the general byte-level track so document structure survives.
        tokenizer = build_fake_tokenizer()
        for text in ("hello\nhello", "test\ttest", "a\r\nb", "x\n\ny"):
            result = tokenizer.encode_with_spans(text)
            self.assertEqual(tokenizer.decode(result.input_ids), text)
            self.assertFalse(
                any(tok.track == "space" for tok in result.tokens),
                f"structural whitespace wrongly folded to ▁ for {text!r}",
            )

    def test_plain_space_still_folds_to_space_token(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("hello  hello")
        space_toks = [t for t in result.tokens if t.track == "space"]
        self.assertEqual(len(space_toks), 2)
        self.assertEqual(tokenizer.decode(result.input_ids), "hello  hello")


class MongolianFallbackOffsetTest(unittest.TestCase):
    """The no-offset MorphBPE fallback must yield monotonic per-piece spans."""

    def test_multi_piece_fallback_offsets_are_monotonic(self):
        class MultiPieceMorphBPE:
            vocab = {"ᠮᠣᠩ": 0, "ᠭᠣᠯ": 1}

            def encode(self, text):
                return [0, 1]

        word = "ᠮᠣᠩᠭᠣᠯ"
        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=MultiPieceMorphBPE.vocab,
            general_vocab=general.get_vocab(),
        )
        tokenizer = DualTrackTokenizer(vocab, MultiPieceMorphBPE(), general)

        result = tokenizer.encode_with_spans(word)
        mn = [t for t in result.tokens if t.track == "mn"]
        self.assertEqual(len(mn), 2)
        self.assertEqual((mn[0].start, mn[0].end), (0, 3))
        self.assertEqual((mn[1].start, mn[1].end), (3, 6))


if __name__ == "__main__":
    unittest.main()
