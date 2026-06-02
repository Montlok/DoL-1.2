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
    def test_segmented_configs_are_cli_selectable(self) -> None:
        for name in ("segmented_tiny", "segmented_pretrain"):
            self.assertIn(name, train_rdt.CONFIG_CHOICES)
        cfg = train_rdt.CONFIG_CHOICES["segmented_pretrain"]()
        self.assertEqual(cfg.core_type, "segmented")

    def test_learning_framework_flags_reach_training_config(self) -> None:
        args = train_rdt.parse_args([
            "--config", "segmented_tiny",
            "--smoke",
            "--optimizer", "muon",
            "--adam-use-atan2",
            "--muon-momentum", "0.9",
            "--muon-ns-steps", "3",
            "--lr-schedule", "wsd",
            "--wsd-stable-ratio", "0.7",
            "--wsd-decay-shape", "linear",
            "--min-lr-ratio", "0.05",
            "--lr-decay-steps", "123",
            "--recurrent-steps-start", "2",
            "--recurrent-steps-ramp", "99",
        ])
        model_cfg = train_rdt._build_model_cfg(args)
        train_cfg = train_rdt._build_train_cfg(args, model_cfg)
        self.assertEqual(train_cfg.optimizer, "muon")
        self.assertTrue(train_cfg.adam_use_atan2)
        self.assertEqual(train_cfg.muon_momentum, 0.9)
        self.assertEqual(train_cfg.muon_ns_steps, 3)
        self.assertEqual(train_cfg.lr_schedule, "wsd")
        self.assertEqual(train_cfg.wsd_stable_ratio, 0.7)
        self.assertEqual(train_cfg.wsd_decay_shape, "linear")
        self.assertEqual(train_cfg.min_lr_ratio, 0.05)
        self.assertEqual(train_cfg.lr_decay_steps, 123)
        self.assertEqual(train_cfg.recurrent_steps_start, 2)
        self.assertEqual(train_cfg.recurrent_steps_ramp, 99)

    def test_random_r_not_overridden_unless_curriculum_active(self) -> None:
        model_cfg = train_rdt.CONFIG_CHOICES["segmented_pretrain"]()
        train_cfg = train_rdt.TrainingConfig(train_data="x", max_steps=10)
        self.assertIsNone(
            train_rdt._target_recurrent_steps_for_train(model_cfg, train_cfg)
        )
        ramp_cfg = train_rdt.TrainingConfig(
            train_data="x",
            max_steps=10,
            recurrent_steps_start=2,
            recurrent_steps_ramp=100,
        )
        self.assertEqual(
            train_rdt._target_recurrent_steps_for_train(model_cfg, ramp_cfg),
            model_cfg.recurrent_steps,
        )

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

    def test_missing_exact_data_file_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", f"{tmp}/missing.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("zero shards", stderr.getvalue())

    def test_empty_eval_shard_glob_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            data = os.path.join(tmp, "train.jsonl")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write(
                    '{"input_ids":[2,3],"attention_mask":[1,1],'
                    '"labels":[-100,3]}\n'
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", data,
                    "--eval-data", f"{tmp}/no_eval_shards_*.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--eval-data", stderr.getvalue())

    def test_smoke_runs_without_data(self) -> None:
        # --smoke explicitly opts into the synthetic batch generator.
        with tempfile.TemporaryDirectory() as tmp:
            rc = train_rdt.main([
                "--config", "tiny", "--smoke", "--output", tmp,
            ])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
