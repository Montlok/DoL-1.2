# -*- coding: utf-8 -*-

"""Unit tests for OCR accuracy metrics (torch-free)."""

import unittest

from Model.ocr.metrics import (
    cer,
    edit_distance,
    nominal_normalize,
    ocr_report,
    wer,
)

# Free variation selector and Mongolian vowel separator — pure encoding
# variation that normalized CER must ignore.
FVS1 = "\u180b"
MVS = "\u180e"
NNBSP = "\u202f"
# A short traditional-Mongolian letter run (nominal code points).
MONG = "\u182d\u1820\u1837"  # GA A RA


class TestEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)

    def test_empty(self):
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", ""), 3)

    def test_substitution_insertion_deletion(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_symmetric(self):
        self.assertEqual(edit_distance("abcd", "abx"), edit_distance("abx", "abcd"))


class TestNominalNormalize(unittest.TestCase):
    def test_python_fallback_strips_fvs_mvs(self):
        folded = nominal_normalize([MONG + FVS1 + MONG + MVS], backend="python")
        self.assertEqual(folded, [MONG + MONG])

    def test_python_fallback_maps_nnbsp_to_space(self):
        folded = nominal_normalize([MONG + NNBSP + MONG], backend="python")
        self.assertEqual(folded, [MONG + " " + MONG])

    def test_auto_backend_returns_same_length(self):
        # auto must never change the number of items, regardless of backend.
        items = [MONG, MONG + FVS1, "", "abc"]
        self.assertEqual(len(nominal_normalize(items, backend="auto")), len(items))


class TestCER(unittest.TestCase):
    def test_fvs_difference_is_free_under_normalized_cer(self):
        # Prediction differs from reference only by an FVS -> normalized CER 0,
        # raw CER non-zero. This is the core Mongolian-OCR measurement point.
        pred = [MONG + FVS1]
        ref = [MONG]
        self.assertAlmostEqual(cer(pred, ref, backend="python"), 0.0)
        self.assertGreater(cer(pred, ref, normalize=False), 0.0)

    def test_corpus_micro_average(self):
        # One substitution over 3+3 reference chars -> 1/6.
        preds = ["abc", "xyz"]
        refs = ["abc", "xyq"]
        self.assertAlmostEqual(cer(preds, refs, normalize=False), 1 / 6)

    def test_perfect(self):
        self.assertEqual(cer(["abc"], ["abc"], normalize=False), 0.0)


class TestWER(unittest.TestCase):
    def test_word_error(self):
        preds = ["a b c d"]
        refs = ["a b x d"]
        self.assertAlmostEqual(wer(preds, refs, normalize=False), 1 / 4)


class TestOCRReport(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ocr_report(["a"], ["a", "b"], backend="python")

    def test_rejection_excludes_samples(self):
        preds = ["good", "WRONG"]
        refs = ["good", "right"]
        # Reject the second (wrong) sample -> scored set is perfect.
        rep = ocr_report(
            preds, refs, backend="python", rejected=[False, True]
        )
        self.assertEqual(rep.n, 1)
        self.assertAlmostEqual(rep.rejection_rate, 0.5)
        self.assertEqual(rep.norm_cer, 0.0)
        self.assertEqual(rep.line_exact, 1.0)

    def test_report_fields_consistent(self):
        preds = [MONG + FVS1, "abc"]
        refs = [MONG, "abd"]
        rep = ocr_report(preds, refs, backend="python")
        self.assertEqual(rep.n, 2)
        self.assertEqual(rep.backend, "python")
        # FVS-only diff on line 1 is free; line 2 has 1/3 char error -> over
        # ref length 3+3=6, total dist 1 -> 1/6.
        self.assertAlmostEqual(rep.norm_cer, 1 / 6)
        self.assertGreaterEqual(rep.raw_cer, rep.norm_cer)


if __name__ == "__main__":
    unittest.main()
