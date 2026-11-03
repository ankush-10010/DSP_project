"""
Unit and integration tests for the DSP ECG pipeline.
"""

import unittest
import numpy as np

from src.data_loader import load_record, get_beat_annotations, list_records
from src.noise_analysis import compute_fft, compute_psd, identify_noise_components, compute_snr
from src.filters import (
    design_highpass_iir,
    design_notch_iir,
    design_lowpass_iir,
    design_highpass_fir,
    design_lowpass_fir,
    apply_filter,
    ecg_filter_pipeline,
)
from src.pan_tompkins import (
    bandpass_filter,
    derivative_filter,
    squaring,
    moving_window_integration,
    pan_tompkins_detect,
)
from src.feature_extraction import (
    compute_rr_intervals,
    compute_heart_rate,
    compute_sdnn,
    compute_rmssd,
    compute_pnn50,
    compute_hrv_frequency,
    compute_all_metrics,
)
from src.poincare import (
    compute_poincare_metrics,
    plot_poincare,
    plot_poincare_comparison,
)
from src.validation import match_peaks, compute_metrics, validate_record


class TestDataLoader(unittest.TestCase):
    def test_list_records(self):
        records = list_records()
        self.assertIn("100", records)
        self.assertGreaterEqual(len(records), 40)

    def test_load_record(self):
        sig, fs = load_record("100")
        self.assertEqual(fs, 360.0)
        self.assertGreater(len(sig), 10000)

    def test_get_beat_annotations(self):
        samples, symbols = get_beat_annotations("100")
        self.assertGreater(len(samples), 2000)
        self.assertEqual(len(samples), len(symbols))
        self.assertNotIn("+", symbols)


class TestNoiseAnalysis(unittest.TestCase):
    def setUp(self):
        self.fs = 360.0
        t = np.arange(0, 5, 1.0 / self.fs)
        self.sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    def test_compute_fft(self):
        freqs, mag = compute_fft(self.sig, self.fs)
        self.assertEqual(len(freqs), len(mag))
        self.assertGreaterEqual(freqs[0], 0)

    def test_compute_psd(self):
        freqs, psd = compute_psd(self.sig, self.fs)
        self.assertEqual(len(freqs), len(psd))
        self.assertTrue(np.all(psd >= 0))

    def test_compute_snr(self):
        clean = np.ones(100)
        noisy = clean + 0.1 * np.ones(100)
        snr = compute_snr(clean, noisy)
        self.assertAlmostEqual(snr, 20.0, places=1)


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.fs = 360.0
        self.sig = np.random.randn(1000)

    def test_iir_designs(self):
        b, a = design_highpass_iir(0.5, self.fs)
        filtered = apply_filter(self.sig, b, a)
        self.assertEqual(len(filtered), len(self.sig))

        b, a = design_notch_iir(50.0, self.fs)
        filtered = apply_filter(self.sig, b, a)
        self.assertEqual(len(filtered), len(self.sig))

        b, a = design_lowpass_iir(150.0, self.fs)
        filtered = apply_filter(self.sig, b, a)
        self.assertEqual(len(filtered), len(self.sig))

    def test_fir_designs(self):
        b = design_highpass_fir(0.5, self.fs, numtaps=51)
        filtered = apply_filter(self.sig, b)
        self.assertEqual(len(filtered), len(self.sig))

    def test_pipeline(self):
        cleaned = ecg_filter_pipeline(self.sig, self.fs)
        self.assertEqual(len(cleaned), len(self.sig))


class TestPanTompkins(unittest.TestCase):
    def test_stages(self):
        sig, fs = load_record("100")
        sub_sig = sig[:3600]  # 10s
        bp = bandpass_filter(sub_sig, fs)
        self.assertEqual(len(bp), len(sub_sig))

        deriv = derivative_filter(bp)
        self.assertEqual(len(deriv), len(sub_sig))

        sq = squaring(deriv)
        self.assertTrue(np.all(sq >= 0))

        mwi = moving_window_integration(sq, fs)
        self.assertEqual(len(mwi), len(sub_sig))

    def test_full_detection_record_100(self):
        sig, fs = load_record("100")
        sub_sig = sig[:3600 * 2]  # 20s
        res = pan_tompkins_detect(sub_sig, fs)
        self.assertIn("r_peaks", res)
        self.assertGreater(len(res["r_peaks"]), 10)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.fs = 360.0
        # 1 beat every 1 second (60 bpm)
        self.r_peaks = np.arange(360, 360 * 20, 360)

    def test_metrics(self):
        rr = compute_rr_intervals(self.r_peaks, self.fs)
        self.assertTrue(np.allclose(rr, 1.0))

        hr = compute_heart_rate(rr)
        self.assertTrue(np.allclose(hr, 60.0))

        sdnn = compute_sdnn(rr)
        self.assertAlmostEqual(sdnn, 0.0, places=3)

        all_m = compute_all_metrics(self.r_peaks, self.fs)
        self.assertAlmostEqual(all_m["mean_hr"], 60.0, places=1)


class TestValidation(unittest.TestCase):
    def test_match_peaks_exact(self):
        detected = np.array([100, 200, 300])
        ref = np.array([100, 200, 300])
        tp, fp, fn = match_peaks(detected, ref, tolerance_samples=10)
        self.assertEqual((tp, fp, fn), (3, 0, 0))

        metrics = compute_metrics(tp, fp, fn)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["positive_predictivity"], 1.0)

    def test_validate_record(self):
        sig, fs = load_record("100")
        sub_sig = sig[:3600 * 5]  # first 50s
        res = pan_tompkins_detect(sub_sig, fs)
        # Validate against full annotations (peaks outside sub_sig are reference-only)
        val = validate_record(res["r_peaks"], "100", fs=fs)
        self.assertGreater(val["tp"], 40)


class TestPoincare(unittest.TestCase):
    def setUp(self):
        self.fs = 360.0
        # 20 beats with slight physiological variation
        self.r_peaks = np.array([
            0, 360, 715, 1080, 1435, 1800, 2165, 2520, 2885, 3240,
            3605, 3960, 4325, 4680, 5045, 5400, 5765, 6120, 6485, 6840
        ])

    def test_constant_rr(self):
        rr = np.ones(10)
        res = compute_poincare_metrics(rr)
        self.assertAlmostEqual(res["sd1"], 0.0, places=6)
        self.assertAlmostEqual(res["sd2"], 0.0, places=6)
        self.assertAlmostEqual(res["sd_ratio"], 0.0, places=6)
        self.assertAlmostEqual(res["xc"], 1.0, places=6)
        self.assertAlmostEqual(res["yc"], 1.0, places=6)
        self.assertAlmostEqual(res["semi_major"], 0.0, places=6)
        self.assertAlmostEqual(res["semi_minor"], 0.0, places=6)
        self.assertAlmostEqual(res["angle"], 45.0, places=6)

    def test_analytical_known_values(self):
        rr = np.array([0.8, 0.85, 0.82, 0.88, 0.84, 0.90])
        res = compute_poincare_metrics(rr)

        delta_rr = np.diff(rr)
        expected_var_diff = float(np.var(delta_rr, ddof=1))
        expected_var_rr = float(np.var(rr, ddof=1))

        expected_sd1 = float(np.sqrt(0.5 * expected_var_diff))
        expected_sd2 = float(np.sqrt(max(0.0, 2.0 * expected_var_rr - 0.5 * expected_var_diff)))
        expected_ratio = expected_sd1 / expected_sd2

        self.assertAlmostEqual(res["sd1"], expected_sd1, places=6)
        self.assertAlmostEqual(res["sd2"], expected_sd2, places=6)
        self.assertAlmostEqual(res["sd_ratio"], expected_ratio, places=6)
        self.assertAlmostEqual(res["xc"], float(np.mean(rr)), places=6)
        self.assertAlmostEqual(res["yc"], float(np.mean(rr)), places=6)
        self.assertAlmostEqual(res["semi_major"], 2.0 * expected_sd2, places=6)
        self.assertAlmostEqual(res["semi_minor"], 2.0 * expected_sd1, places=6)

    def test_brennan_identity(self):
        # Brennan et al. (2001) theorem: SD1^2 + SD2^2 = 2 * Var(RR)
        np.random.seed(42)
        rr = 0.8 + 0.05 * np.random.randn(100)
        res = compute_poincare_metrics(rr)
        self.assertAlmostEqual(res["sd1"] ** 2 + res["sd2"] ** 2, 2.0 * res["var_rr"], places=6)

    def test_ellipse_parameters(self):
        rr = np.array([0.8, 0.85, 0.79, 0.82, 0.81, 0.84, 0.78, 0.83])
        res = compute_poincare_metrics(rr)

        self.assertIn("ellipse_center", res)
        self.assertEqual(res["ellipse_center"], (res["xc"], res["yc"]))
        self.assertEqual(res["ellipse_angle"], 45.0)
        self.assertAlmostEqual(res["ellipse_semi_major"], 2.0 * res["sd2"])
        self.assertAlmostEqual(res["ellipse_semi_minor"], 2.0 * res["sd1"])
        self.assertGreater(res["ellipse_area"], 0.0)
        self.assertGreater(res["confidence_area"], 0.0)

    def test_feature_extraction_integration(self):
        # compute_all_metrics includes poincare by default
        all_m = compute_all_metrics(self.r_peaks, self.fs, include_poincare=True)
        self.assertIn("poincare", all_m)
        self.assertIn("sd1", all_m)
        self.assertIn("sd2", all_m)
        self.assertIn("sd_ratio", all_m)
        self.assertGreater(all_m["sd1"], 0.0)
        self.assertGreater(all_m["sd2"], 0.0)

        # when include_poincare=False, poincare key is omitted
        all_m_no_p = compute_all_metrics(self.r_peaks, self.fs, include_poincare=False)
        self.assertNotIn("poincare", all_m_no_p)
        self.assertNotIn("sd1", all_m_no_p)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            compute_poincare_metrics(np.array([1.0, 1.0]))  # < 3 intervals

        with self.assertRaises(ValueError):
            compute_poincare_metrics(np.array([1.0, -0.5, 1.0]))  # non-positive interval

    def test_real_ecg_normal_vs_arrhythmia(self):
        sig_100, fs = load_record("100")
        sub_100 = sig_100[:360 * 60]  # 1 minute
        pt_100 = pan_tompkins_detect(sub_100, fs)
        rr_100 = compute_rr_intervals(pt_100["r_peaks"], fs)
        m_100 = compute_poincare_metrics(rr_100)

        self.assertGreater(m_100["sd1"], 0.0)
        self.assertGreater(m_100["sd2"], 0.0)

        sig_203, _ = load_record("203")
        sub_203 = sig_203[:360 * 60]  # 1 minute
        pt_203 = pan_tompkins_detect(sub_203, fs)
        rr_203 = compute_rr_intervals(pt_203["r_peaks"], fs)
        m_203 = compute_poincare_metrics(rr_203)

        # Arrhythmia has markedly higher dispersion than normal sinus (PVCs & pauses)
        self.assertGreater(m_203["sd1"], m_100["sd1"] * 2.0)
        self.assertGreater(m_203["sd2"], m_100["sd2"] * 2.0)

    def test_plotting_functions(self):
        rr = np.array([0.8, 0.85, 0.79, 0.82, 0.81, 0.84, 0.78, 0.83])
        fig1 = plot_poincare(rr, title="Test Poincaré", save_path=None)
        self.assertIsNotNone(fig1)

        fig2 = plot_poincare_comparison(rr, rr * 1.5, save_path=None)
        self.assertIsNotNone(fig2)


if __name__ == "__main__":
    unittest.main()
