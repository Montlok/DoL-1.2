# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_vlm_align``.

Ensures bad CLI input fails cleanly (exit code 2 + stderr message) rather than
raising a raw traceback — in particular ``--image-size`` is validated before
``image_patch_count`` derives the default ``--n-image-tokens``.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from scripts import train_vlm_align


class TrainVlmAlignCliGuardsTest(unittest.TestCase):
    def test_non_positive_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "0", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())

    def test_non_multiple_of_four_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "6", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())

    def test_official_mamba_failure_returns_exit_code_two(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                train_vlm_align.torch.cuda,
                "is_available",
                return_value=False,
            ),
            contextlib.redirect_stderr(stderr),
        ):
            rc = train_vlm_align.main([
                "--mamba", "official",
                "--image-size", "4",
                "--n-image-tokens", "1",
                "--seq-len", "8",
            ])
        self.assertEqual(rc, 2)
        msg = stderr.getvalue()
        self.assertIn("--mamba official", msg)
        self.assertIn("target device is cpu", msg)


if __name__ == "__main__":
    unittest.main()
