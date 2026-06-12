# -*- coding: utf-8 -*-

"""Tests for the rdt_monitor log-mode source (runs without a StatusReporter)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from scripts.rdt_monitor import (
    LogSource,
    _fmt_eta,
    downsample_history,
    parse_log_row,
)


class ParseLogRowTest(unittest.TestCase):
    def test_parses_step_and_numeric_metrics(self) -> None:
        row = parse_log_row("step=12 loss=3.1400 lr=2.500e-04\n")
        self.assertEqual(row, {"step": 12, "loss": 3.14, "lr": 2.5e-4})

    def test_non_step_lines_are_ignored(self) -> None:
        self.assertIsNone(parse_log_row("[init] loaded checkpoint\n"))
        self.assertIsNone(parse_log_row("VLM align real-data run OK in 3.0s\n"))

    def test_non_numeric_values_are_skipped(self) -> None:
        row = parse_log_row("step=3 loss=nan-ish state=running\n")
        self.assertEqual(row, {"step": 3})


class DownsampleHistoryTest(unittest.TestCase):
    def test_short_history_passes_through(self) -> None:
        rows = [{"step": i, "loss": float(i)} for i in range(5)]
        self.assertEqual(downsample_history(rows, 10), rows)

    def test_buckets_average_and_keep_last_step(self) -> None:
        rows = [{"step": i, "loss": float(i)} for i in range(1, 5)]
        out = downsample_history(rows, 2)
        self.assertEqual(len(out), 2)
        self.assertEqual([r["step"] for r in out], [2, 4])
        self.assertEqual([r["loss"] for r in out], [1.5, 3.5])

    def test_missing_keys_are_tolerated(self) -> None:
        rows = [{"step": 1, "loss": 1.0}, {"step": 2}, {"step": 3, "loss": 3.0}]
        out = downsample_history(rows, 1)
        self.assertEqual(out[0]["step"], 3)
        self.assertAlmostEqual(out[0]["loss"], 2.0)


class LogSourceTest(unittest.TestCase):
    def _write_log(self, lines: list[str]) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "run.log"
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def test_history_parses_only_metric_lines(self) -> None:
        path = self._write_log(
            ["starting up\n", "step=1 loss=4.0\n", "step=2 loss=3.5\n"]
        )
        src = LogSource(path)
        self.assertEqual(
            src.history(10),
            [{"step": 1, "loss": 4.0}, {"step": 2, "loss": 3.5}],
        )

    def test_status_running_then_stalled_then_finished(self) -> None:
        path = self._write_log(["step=5 loss=3.0\n"])
        src = LogSource(path, max_steps=10, stale_after=60.0)
        status = src.status()
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["step"], 5)
        self.assertEqual(status["metrics"], {"loss": 3.0})

        old = time.time() - 3600
        os.utime(path, (old, old))
        self.assertEqual(src.status()["state"], "stalled")

        path.write_text("step=10 loss=2.0\n", encoding="utf-8")
        os.utime(path, (old, old))
        self.assertEqual(src.status()["state"], "finished")

    def test_status_missing_file_is_none(self) -> None:
        src = LogSource("/nonexistent/run.log")
        self.assertIsNone(src.status())

    def test_rate_needs_a_wide_enough_window(self) -> None:
        path = self._write_log(["step=100 loss=3.0\n"])
        src = LogSource(path, max_steps=200)
        now = time.time()
        src._samples = [(now - 30.0, 40)]
        status = src.status()
        self.assertAlmostEqual(status["rate_steps_per_s"], 2.0, places=1)

    def test_eta_uses_measured_rate(self) -> None:
        eta = _fmt_eta(
            {"step": 100, "max_steps": 280, "rate_steps_per_s": 1.0}
        )
        self.assertEqual(eta, "00:03:00")


if __name__ == "__main__":
    unittest.main()
