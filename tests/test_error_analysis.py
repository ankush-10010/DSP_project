"""
test_error_analysis.py — Unit tests for Failure Mode & Error Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.error_analysis import (
    analyze_false_positives,
    analyze_false_negatives,
    evaluate_by_beat_type,
    plot_record_108_false_positives,
    plot_record_203_false_negatives,
    plot_beat_type_sensitivity,
)

class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.fs = 360.0
        self.t = np.arange(0, 10, 1.0 / self.fs)
        self.signal = np.sin(2 * np.pi * 1.5 * self.t)
        self.reference = np.array([360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240], dtype=np.int64)
        self.symbols = ["N", "N", "V", "N", "A", "N", "V", "N", "N"]
        self.tolerance = int(round(0.150 * self.fs))

    def test_analyze_false_positives_clean(self):
        detected = self.reference.copy()
        res = analyze_false_positives(detected, self.reference, self.signal, self.fs, self.tolerance)
        self.assertEqual(res["total_fps"], 0)
        self.assertEqual(len(res["fp_indices"]), 0)

    def test_analyze_false_positives_with_fp(self):
        # Insert a spurious peak at 360 + 90 (250 ms after first beat: T-wave region)
        detected = np.sort(np.append(self.reference, [450]))
        res = analyze_false_positives(detected, self.reference, self.signal, self.fs, self.tolerance)
        self.assertEqual(res["total_fps"], 1)
        self.assertEqual(res["fp_indices"][0], 450)
        self.assertEqual(len(res["fp_records"]), 1)
        self.assertEqual(res["fp_records"][0]["cause"], "T-wave oversensing")

    def test_analyze_false_negatives_clean(self):
        detected = self.reference.copy()
        res = analyze_false_negatives(
            detected, self.reference, self.signal, self.fs, self.tolerance, self.symbols
        )
        self.assertEqual(res["total_fns"], 0)
        self.assertEqual(len(res["fn_indices"]), 0)

    def test_analyze_false_negatives_with_fn(self):
        # Drop the beat at 1080 (which is 'V')
        detected = np.delete(self.reference, 2)
        res = analyze_false_negatives(
            detected, self.reference, self.signal, self.fs, self.tolerance, self.symbols
        )
        self.assertEqual(res["total_fns"], 1)
        self.assertEqual(res["fn_indices"][0], 1080)
        self.assertEqual(res["fn_symbols"][0], "V")
        self.assertIn("Ventricular", res["fn_records"][0]["cause"])

    def test_evaluate_by_beat_type(self):
        # Test on single clean record "100"
        df = evaluate_by_beat_type(["100"], fs=self.fs, tolerance_ms=150.0)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("beat_symbol", df.columns)
        self.assertIn("sensitivity_pct", df.columns)
        self.assertGreater(len(df), 0)
        # Record 100 has mostly 'N'
        n_row = df[df["beat_symbol"] == "N"]
        self.assertFalse(n_row.empty)
        self.assertGreater(n_row["sensitivity_pct"].iloc[0], 99.0)

    def test_plot_generation(self):
        tmp_dir = Path("/tmp/dsp_test_figures")
        tmp_dir.mkdir(exist_ok=True)
        p1 = tmp_dir / "test_108.png"
        p2 = tmp_dir / "test_203.png"
        p3 = tmp_dir / "test_chart.png"

        plot_record_108_false_positives(save_path=p1, t_start=1674.0, duration=2.0)
        self.assertTrue(p1.exists())
        self.assertGreater(p1.stat().st_size, 1000)

        plot_record_203_false_negatives(save_path=p2, t_start=8.0, duration=2.0)
        self.assertTrue(p2.exists())
        self.assertGreater(p2.stat().st_size, 1000)

        plot_beat_type_sensitivity(save_path=p3)
        self.assertTrue(p3.exists())
        self.assertGreater(p3.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
