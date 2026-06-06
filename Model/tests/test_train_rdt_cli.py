# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_rdt``.

These guards prevent the most dangerous foot-gun: kicking off what looks
like a real pretraining run but silently feeding the model random tokens
and emitting checkpoints with no useful signal.

Tests are hermetic: ``--output`` is redirected to a ``TemporaryDirectory``
so they never touch ``outputs/`` in the working tree, and we assert on
exit codes / stderr messages so a regression that exits for a different
reason (e.g. missing module) would not silently pass.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest

from scripts import train_rdt


class TrainRdtCliGuardsTest(unittest.TestCase):
    def test_requires_data_or_smoke(self) -> None:
        # No --data, no --smoke ⇒ must return non-zero exit code with a
        # diagnostic message before any expensive setup (model alloc,
        # output-dir creation, distributed init).
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main(["--config", "tiny", "--output", tmp])
            self.assertEqual(rc, 2)
            msg = stderr.getvalue()
            self.assertIn("--data is required", msg)
            # And no checkpoint dirs should have been created.
            import os
            self.assertEqual(os.listdir(tmp), [])

    def test_resume_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--smoke",
                    "--resume", "/definitely/does/not/exist",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--resume", stderr.getvalue())

    def test_tokenizer_bundle_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--smoke",
                    "--tokenizer-bundle", "/definitely/not/a/bundle",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--tokenizer-bundle", stderr.getvalue())

    def test_empty_shard_glob_fails_fast(self) -> None:
        # An empty glob (typo'd shard pattern) must abort with exit 2
        # *before* the model is allocated.
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", f"{tmp}/no_such_shards_*.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("zero shards", stderr.getvalue())

    def test_smoke_runs_without_data(self) -> None:
        # --smoke explicitly opts into the synthetic batch generator.
        with tempfile.TemporaryDirectory() as tmp:
            rc = train_rdt.main([
                "--config", "tiny", "--smoke", "--output", tmp,
            ])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
