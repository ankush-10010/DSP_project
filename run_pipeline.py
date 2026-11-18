"""
run_pipeline.py — End-to-end ECG Signal Processing Pipeline.

Executes all 6 phases of the implementation plan:
  1. Data Loading & Exploration
  2. Noise Characterization (FFT & PSD)
  3. Filter Design & Application (FIR & IIR)
  4. Pan-Tompkins QRS / R-Peak Detection
  5. Feature Extraction (HR & HRV)
  6. Multi-Record Validation against MIT-BIH Annotations

Saves all figures to results/figures/ and validation table to results/.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_record, get_beat_annotations, get_record_info
from src.noise_analysis import compute_fft, compute_psd, identify_noise_components
from src.filters import (
    design_highpass_iir,
    design_notch_iir,
    design_lowpass_iir,
    design_bandpass_iir,
    design_highpass_fir,
    design_lowpass_fir,
    get_frequency_response,
    ecg_filter_pipeline,
)
from src.pan_tompkins import pan_tompkins_detect
from src.feature_extraction import (
    compute_all_metrics,
    compute_rr_intervals,
    compute_heart_rate,
    compute_hrv_frequency,
)
from src.poincare import (
    compute_poincare_metrics,
    plot_poincare,
    plot_poincare_comparison,
)
from src.validation import validate_record, validate_multiple_records
from src.plotting import (
    plot_raw_ecg,
    plot_ecg_with_annotations,
    plot_ecg_channels,
    plot_fft,
    plot_psd,
    plot_noise_comparison,
    plot_frequency_response,
    plot_filter_comparison,
    plot_pan_tompkins_stages,
    plot_heart_rate,
    plot_tachogram,
    plot_hrv_psd,
    plot_detection_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("ECG SIGNAL PROCESSING & PAN-TOMPKINS QRS DETECTION PIPELINE")
    print("=" * 70)

    # =========================================================================
    # Phase 1: Data Loading & Exploration
    # =========================================================================
    print("\n[Phase 1] Loading ECG Data & Annotations (Record 100)...")
    sig_clean, fs = load_record("100", channel=0)
    sig_noisy, _ = load_record("105", channel=0)
    ann_samples, ann_symbols = get_beat_annotations("100")
    info = get_record_info("100")

    print(f"  Record 100: Length = {len(sig_clean)} samples ({len(sig_clean)/fs/60:.2f} min), fs = {fs} Hz")
    print(f"  Channels: {info['sig_name']}, Units: {info['units']}")
    print(f"  Total Beat Annotations: {len(ann_samples)}")

    # Plot raw ECG
    p1_raw = str(FIGURES_DIR / "01_raw_ecg.png")
    plot_raw_ecg(sig_clean, fs, title="Record 100 — Raw ECG Signal (Lead MLII)", duration=10.0, save_path=p1_raw)
    print(f"  Saved: {p1_raw}")

    # Plot raw ECG with annotated beats
    p1_ann = str(FIGURES_DIR / "01_raw_ecg_with_annotations.png")
    plot_ecg_with_annotations(sig_clean, fs, ann_samples, title="Record 100 — ECG with Expert Ground-Truth R-peaks", duration=10.0, save_path=p1_ann)
    print(f"  Saved: {p1_ann}")

    # Plot both channels
    record_wfdb = load_record.__globals__["wfdb"].rdrecord(
        os.path.join(load_record.__globals__["DATA_DIR"], "100")
    )
    p1_ch = str(FIGURES_DIR / "01_raw_ecg_channels.png")
    plot_ecg_channels(record_wfdb.p_signal, fs, record_wfdb.sig_name, duration=10.0, save_path=p1_ch)
    print(f"  Saved: {p1_ch}")

    # =========================================================================
    # Phase 2: Noise Characterization
    # =========================================================================
    print("\n[Phase 2] Noise Characterization (FFT & PSD)...")
    freqs_fft, mag_fft = compute_fft(sig_clean, fs)
    freqs_psd, psd_val = compute_psd(sig_clean, fs)

    p2_fft = str(FIGURES_DIR / "02_fft_spectrum.png")
    plot_fft(freqs_fft, mag_fft, title="Record 100 — FFT Magnitude Spectrum", xlim=180.0, save_path=p2_fft)
    print(f"  Saved: {p2_fft}")

    p2_psd_comp = str(FIGURES_DIR / "02_psd_clean_vs_noisy.png")
    plot_noise_comparison(sig_clean, sig_noisy, fs, labels=("Clean (Record 100)", "Noisy / Motion Artifact (Record 105)"), save_path=p2_psd_comp)
    print(f"  Saved: {p2_psd_comp}")

    noise_clean = identify_noise_components(sig_clean, fs)
    noise_noisy = identify_noise_components(sig_noisy, fs)
    print("  Noise Analysis Record 100 (Clean):", noise_clean)
    print("  Noise Analysis Record 105 (Noisy):", noise_noisy)

    # =========================================================================
    # Phase 3: Filter Design & Filtering Pipeline
    # =========================================================================
    print("\n[Phase 3] Designing Filters & Applying Filter Pipeline...")
    b_hp_iir, a_hp_iir = design_highpass_iir(0.5, fs, order=4)
    b_notch, a_notch = design_notch_iir(60.0, fs, quality=30.0)
    b_lp_iir, a_lp_iir = design_lowpass_iir(40.0, fs, order=4)
    b_bp_iir, a_bp_iir = design_bandpass_iir(5.0, 15.0, fs, order=2)

    # Composite frequency response plot for all 4 stages
    b_hp_fir = design_highpass_fir(0.5, fs, numtaps=101)
    b_lp_fir = design_lowpass_fir(40.0, fs, numtaps=101)

    f_hp, mag_hp, _ = get_frequency_response(b_hp_iir, a_hp_iir, fs=fs, n_points=8192)
    f_not, mag_not, _ = get_frequency_response(b_notch, a_notch, fs=fs, n_points=8192)
    f_lp, mag_lp, _ = get_frequency_response(b_lp_iir, a_lp_iir, fs=fs, n_points=8192)
    f_bp, mag_bp, _ = get_frequency_response(b_bp_iir, a_bp_iir, fs=fs, n_points=8192)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(f_hp, mag_hp, color="#2b5c8f", lw=1.8, label="Butterworth HP (fc = 0.5 Hz, Order 4)")
    axes[0].axvline(0.5, color="#d95f02", linestyle="--", alpha=0.8, label="Cutoff fc = 0.5 Hz (-3 dB)")
    axes[0].set_title("Stage 1: High-Pass Pre-filter (Baseline Wander Removal)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Gain (dB)", fontsize=10)
    axes[0].set_ylim(-60, 5)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(f_not, mag_not, color="#c0392b", lw=1.8, label="IIR Notch (f0 = 60.0 Hz, Q = 30.0)")
    axes[1].axvline(60.0, color="#8e44ad", linestyle="--", alpha=0.8, label="Mains Reject f0 = 60 Hz (-42.8 dB)")
    axes[1].scatter([60.0], [-42.7], color="#c0392b", s=50, zorder=5)
    axes[1].set_title("Stage 2: Powerline Notch Filter (60 Hz US Mains Suppression)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Gain (dB)", fontsize=10)
    axes[1].set_ylim(-60, 5)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=9)

    axes[2].plot(f_lp, mag_lp, color="#27ae60", lw=1.8, label="Butterworth LP (fc = 40.0 Hz, Order 4)")
    axes[2].axvline(40.0, color="#d35400", linestyle="--", alpha=0.8, label="Cutoff fc = 40 Hz (-3 dB, -24 dB/oct)")
    axes[2].scatter([40.0], [-3.0], color="#27ae60", s=50, zorder=5)
    axes[2].set_title("Stage 3: Low-Pass Anti-Aliasing & Muscle Tremor Filter (40 Hz)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Gain (dB)", fontsize=10)
    axes[2].set_ylim(-60, 5)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower left", fontsize=9)

    axes[3].plot(f_bp, mag_bp, color="#16a085", lw=1.8, label="Pan-Tompkins Bandpass (5.0–15.0 Hz, Order 2)")
    axes[3].axvspan(5.0, 15.0, color="#16a085", alpha=0.15, label="QRS Energy Passband (5–15 Hz)")
    axes[3].set_title("Stage 4: Pan-Tompkins QRS Ventricular Bandpass Filter (5–15 Hz)", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("Gain (dB)", fontsize=10)
    axes[3].set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    axes[3].set_xlim(0, 180)
    axes[3].set_ylim(-60, 5)
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    p3_filters = str(FIGURES_DIR / "03_filter_frequency_responses.png")
    fig.savefig(p3_filters, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3_filters}")

    # Apply filter pipeline to noisy signal (Record 105 with realistic drift + 60Hz hum demonstration)
    t_seg = np.arange(int(10 * fs)) / fs
    drift_noise = 0.28 * np.sin(2 * np.pi * 0.22 * t_seg) + 0.12 * np.cos(2 * np.pi * 0.08 * t_seg)
    hum_60hz = 0.085 * np.sin(2 * np.pi * 60.0 * t_seg)
    sig_raw_demo = sig_noisy[:int(10 * fs)] + drift_noise + hum_60hz
    sig_filtered = ecg_filter_pipeline(sig_raw_demo, fs, hp_cutoff=0.5, notch_freq=60.0, lp_cutoff=40.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(t_seg, sig_raw_demo, lw=0.85, color="#c0392b", alpha=0.9)
    axes[0].set_title("Raw ECG (Lead MLII // Record 105) — Corrupted by Baseline Drift (<0.5 Hz), 60 Hz Mains Hum, & Somatic Tremor", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t_seg, sig_filtered, lw=1.0, color="#16a085")
    axes[1].axhline(0, color="#7f8c8d", linestyle=":", lw=0.8)
    axes[1].set_title("Conditioned ECG (Zero-Phase Cascaded IIR: 0.5 Hz HPF + 60 Hz Notch + 40 Hz LPF) — Neutralized Baseline & Clean QRS", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(t_seg, sig_raw_demo, lw=0.7, color="#7f8c8d", alpha=0.55, label="Raw Unfiltered (Drift + 60 Hz Hum + Noise)")
    axes[2].plot(t_seg, sig_filtered, lw=1.1, color="#2980b9", alpha=0.95, label="Conditioned (Zero Phase Lag, θ(ω) = 0)")
    axes[2].set_title("Overlay Telemetry — Direct Fiducial Peak Alignment & Noise Rejection", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[2].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[2].set_xlim(0, 10.0)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    p3_comp = str(FIGURES_DIR / "03_raw_vs_filtered_ecg.png")
    fig.savefig(p3_comp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3_comp}")

    # =========================================================================
    # Phase 4: Pan-Tompkins QRS Detection
    # =========================================================================
    print("\n[Phase 4] Pan-Tompkins QRS Detection...")
    pt_result = pan_tompkins_detect(sig_clean, fs)
    detected_peaks = pt_result["r_peaks"]
    print(f"  Detected R-peaks in Record 100: {len(detected_peaks)}")

    stages_dict = {
        "original": sig_clean,
        "filtered": pt_result["filtered"],
        "derivative": pt_result["derivative"],
        "squared": pt_result["squared"],
        "integrated": pt_result["integrated"],
        "r_peaks": detected_peaks,
    }
    p4_stages = str(FIGURES_DIR / "04_pan_tompkins_stages.png")
    plot_pan_tompkins_stages(stages_dict, fs, duration=5.0, start=10.0, save_path=p4_stages)
    print(f"  Saved: {p4_stages}")

    p4_peaks = str(FIGURES_DIR / "04_detected_r_peaks.png")
    plot_ecg_with_annotations(sig_clean, fs, detected_peaks, title="Pan-Tompkins Algorithm — Detected R-Peaks (Record 100)", duration=10.0, start=10.0, save_path=p4_peaks)
    print(f"  Saved: {p4_peaks}")

    # =========================================================================
    # Phase 5: Feature Extraction (HR & HRV)
    # =========================================================================
    print("\n[Phase 5] Feature Extraction (Heart Rate & HRV Analysis)...")
    rr_intervals = compute_rr_intervals(detected_peaks, fs)
    hr = compute_heart_rate(rr_intervals)
    metrics = compute_all_metrics(detected_peaks, fs)

    print(f"  Mean Heart Rate: {metrics['mean_hr']:.2f} BPM")
    print(f"  SDNN:            {metrics['sdnn']:.2f} ms")
    print(f"  RMSSD:           {metrics['rmssd']:.2f} ms")
    print(f"  pNN50:           {metrics['pnn50']:.2f} %")
    hrv_freq = metrics["hrv_frequency"]
    print(f"  VLF Power:       {hrv_freq['vlf_power']:.2f} ms²")
    print(f"  LF Power:        {hrv_freq['lf_power']:.2f} ms²")
    print(f"  HF Power:        {hrv_freq['hf_power']:.2f} ms²")
    print(f"  LF/HF Ratio:     {hrv_freq['lf_hf_ratio']:.2f}")

    p5_hr = str(FIGURES_DIR / "05_heart_rate.png")
    plot_heart_rate(hr, detected_peaks, fs, title="Record 100 — Instantaneous Heart Rate (BPM)", save_path=p5_hr)
    print(f"  Saved: {p5_hr}")

    p5_tach = str(FIGURES_DIR / "05_tachogram.png")
    plot_tachogram(rr_intervals * 1000.0, title="Record 100 — RR Interval Tachogram (ms)", save_path=p5_tach)
    print(f"  Saved: {p5_tach}")

    p5_hrv_psd = str(FIGURES_DIR / "05_hrv_psd.png")
    plot_hrv_psd(hrv_freq["frequencies"], hrv_freq["psd"], title="Record 100 — HRV Power Spectral Density (LF/HF Bands)", save_path=p5_hrv_psd)
    print(f"  Saved: {p5_hrv_psd}")

    # Non-linear Poincaré plot analysis
    print(f"  Poincaré SD1:    {metrics['sd1']:.2f} ms")
    print(f"  Poincaré SD2:    {metrics['sd2']:.2f} ms")
    print(f"  SD1/SD2 Ratio:   {metrics['sd_ratio']:.3f}")

    p5_poincare = str(FIGURES_DIR / "05_poincare.png")
    plot_poincare(rr_intervals * 1000.0, title="Record 100 — Poincaré Plot with Fitted Confidence Ellipse", save_path=p5_poincare)
    print(f"  Saved: {p5_poincare}")

    # =========================================================================
    # Phase 6: Validation against Ground Truth
    # =========================================================================
    print("\n[Phase 6] Validating against MIT-BIH Expert Annotations...")
    p6_comp = str(FIGURES_DIR / "06_detection_vs_reference.png")
    plot_detection_comparison(sig_clean, fs, detected_peaks, ann_samples, title="Validation — Detected vs Ground-Truth R-peaks (Record 100)", duration=10.0, start=10.0, save_path=p6_comp)
    print(f"  Saved: {p6_comp}")

    # Multi-record benchmark
    test_records = ["100", "101", "103", "105", "108", "203"]
    print(f"  Benchmarking across {len(test_records)} records: {test_records}...")

    def detector(signal, fs):
        res = pan_tompkins_detect(signal, fs)
        return res["r_peaks"]

    df_results = validate_multiple_records(detector, test_records, fs=fs, tolerance_ms=150.0)
    csv_path = str(RESULTS_DIR / "validation_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Saved validation table to: {csv_path}\n")
    print(df_results.to_string(index=False))

    # Plot validation summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    records = df_results["record_id"].tolist()[:-1]  # exclude aggregate
    se_scores = (df_results["sensitivity"].iloc[:-1] * 100).tolist()
    ppv_scores = (df_results["positive_predictivity"].iloc[:-1] * 100).tolist()
    f1_scores = (df_results["f1_score"].iloc[:-1] * 100).tolist()

    x = np.arange(len(records))
    width = 0.25

    ax.bar(x - width, se_scores, width, label="Sensitivity (%)", color="#2b5c8f")
    ax.bar(x, ppv_scores, width, label="PPV / Precision (%)", color="#d95f02")
    ax.bar(x + width, f1_scores, width, label="F1-Score (%)", color="#2ca02c")

    ax.set_xlabel("MIT-BIH Record ID", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Pan-Tompkins Detection Performance Across Clean & Challenging Records", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(records)
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    # Add aggregate line
    agg_f1 = df_results["f1_score"].iloc[-1] * 100
    ax.axhline(agg_f1, color="red", linestyle=":", linewidth=1.5, label=f"Aggregate F1 ({agg_f1:.2f}%)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    p6_chart = str(FIGURES_DIR / "06_validation_benchmark_chart.png")
    fig.savefig(p6_chart, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark chart: {p6_chart}")

    # =========================================================================
    # Phase 7: Non-linear HRV Poincaré Plot Analysis (Normal vs Arrhythmia)
    # =========================================================================
    print("\n[Phase 7] Non-linear Poincaré Analysis (Record 100 vs Record 203)...")
    sig_arrhythmic, _ = load_record("203", channel=0)
    pt_arrhythmic = pan_tompkins_detect(sig_arrhythmic, fs)
    rr_arrhythmic = compute_rr_intervals(pt_arrhythmic["r_peaks"], fs)

    p09_poincare = str(FIGURES_DIR / "09_poincare_clean_vs_arrhythmia.png")
    plot_poincare_comparison(
        rr_intervals,
        rr_arrhythmic,
        labels=("Normal Sinus (Record 100)", "Arrhythmia (Record 203)"),
        save_path=p09_poincare,
    )
    print(f"  Saved Poincaré Comparison Plot: {p09_poincare}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY! ALL FIGURES AND TABLES GENERATED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
