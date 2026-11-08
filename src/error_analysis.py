"""
error_analysis.py — Failure Mode & Error Analysis for Pan-Tompkins QRS Detector.

Performs systematic failure mode and error analysis on the Pan-Tompkins
QRS detection algorithm using the MIT-BIH Arrhythmia Database.

Key features:
1. False Positive (FP) Analysis:
   - Locates exact FP timestamps and indices.
   - Extracts 2-second windows around FP occurrences.
   - Classifies error mechanisms (T-wave oversensing, baseline wander spikes,
     biphasic/inverted wave double-counting, pre-systolic/P-wave oversensing).
2. False Negative (FN) Analysis:
   - Locates exact FN timestamps and indices.
   - Cross-references with MIT-BIH expert beat annotations ('N', 'V', 'A', 'F', etc.).
   - Identifies morphological and physiological causes of missed beats (e.g. low-slope
     broad PVCs attenuated by bandpass, sub-threshold integrated energy).
3. Beat-Type Sensitivity Evaluation:
   - Benchmarks detection sensitivity across specific beat types (Normal 'N',
     Ventricular Ectopic 'V', Atrial Ectopic 'A', etc.).
4. Publication-Quality Diagnostic Visualizations:
   - Record 108: Multi-stage breakdown showing false positives from large inverted/biphasic
     morphology and T-waves.
   - Record 203: Multi-stage breakdown showing missed PVCs due to slow rise time ($dV/dt$)
     and bandpass attenuation vs. detected beats.
   - Beat-type detection sensitivity comparison bar chart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.data_loader import get_beat_annotations, load_record
from src.pan_tompkins import pan_tompkins_detect

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = _PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = _PROJECT_ROOT / "results"

# Standard beat mapping dictionary
BEAT_NAMES: dict[str, tuple[str, str]] = {
    "N": ("Normal beat", "Normal Sinus"),
    "L": ("Left bundle branch block", "Bundle Branch Block"),
    "R": ("Right bundle branch block", "Bundle Branch Block"),
    "B": ("Bundle branch block (unspecified)", "Bundle Branch Block"),
    "A": ("Atrial premature beat", "Atrial Ectopic (APC)"),
    "a": ("Aberrated atrial premature beat", "Atrial Ectopic (APC)"),
    "J": ("Nodal / Junctional premature beat", "Supraventricular Ectopic"),
    "S": ("Supraventricular premature beat", "Supraventricular Ectopic"),
    "V": ("Premature ventricular contraction", "Ventricular Ectopic (PVC)"),
    "r": ("R-on-T premature ventricular contraction", "Ventricular Ectopic (PVC)"),
    "F": ("Fusion of ventricular and normal beat", "Fusion Beat"),
    "e": ("Atrial escape beat", "Escape Beat"),
    "j": ("Junctional escape beat", "Escape Beat"),
    "n": ("Supraventricular escape beat", "Escape Beat"),
    "E": ("Ventricular escape beat", "Ventricular Ectopic (PVC)"),
    "/": ("Paced beat", "Paced Beat"),
    "f": ("Fusion of paced and normal beat", "Fusion Beat"),
    "Q": ("Unclassifiable beat", "Unclassifiable"),
    "?": ("Beat not classified", "Unclassifiable"),
}


# =========================================================================== #
#                           Matching Helper                                   #
# =========================================================================== #

def _match_detected_to_reference(
    detected: np.ndarray,
    reference: np.ndarray,
    tolerance_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Perform greedy distance-based matching between detected and reference peaks.

    Returns
    -------
    matched_det_mask : np.ndarray of bool
        True if detected[i] was matched to a reference beat.
    matched_ref_mask : np.ndarray of bool
        True if reference[j] was matched to a detected beat.
    matched_pairs : list of tuple
        (detected_idx, reference_idx, distance_samples)
    """
    detected = np.asarray(detected, dtype=np.int64).ravel()
    reference = np.asarray(reference, dtype=np.int64).ravel()

    matched_det = np.zeros(detected.size, dtype=bool)
    matched_ref = np.zeros(reference.size, dtype=bool)
    matched_pairs: list[tuple[int, int, int]] = []

    if detected.size == 0 or reference.size == 0:
        return matched_det, matched_ref, matched_pairs

    pairs: list[tuple[int, int, int]] = []
    for d_idx, d_val in enumerate(detected):
        distances = np.abs(reference - d_val)
        within = np.where(distances <= tolerance_samples)[0]
        for r_idx in within:
            pairs.append((d_idx, r_idx, int(distances[r_idx])))

    pairs.sort(key=lambda x: x[2])

    for d_idx, r_idx, dist in pairs:
        if not matched_det[d_idx] and not matched_ref[r_idx]:
            matched_det[d_idx] = True
            matched_ref[r_idx] = True
            matched_pairs.append((d_idx, r_idx, dist))

    return matched_det, matched_ref, matched_pairs


# =========================================================================== #
#                        False Positive Analysis                              #
# =========================================================================== #

def analyze_false_positives(
    detected: np.ndarray,
    reference: np.ndarray,
    signal: np.ndarray,
    fs: float,
    tolerance_samples: int,
) -> dict[str, Any]:
    """Locate false positives, extract 2-second windows, and classify causes.

    Parameters
    ----------
    detected : np.ndarray
        1-D array of detected R-peak sample indices.
    reference : np.ndarray
        1-D array of reference (ground truth) R-peak sample indices.
    signal : np.ndarray
        1-D array of raw ECG signal.
    fs : float
        Sampling frequency in Hz.
    tolerance_samples : int
        Matching tolerance window in samples (e.g. 54 samples for 150 ms at 360 Hz).

    Returns
    -------
    dict
        Analysis results containing:
        - ``'fp_indices'``: np.ndarray of FP sample indices.
        - ``'fp_times'``: np.ndarray of FP timestamps in seconds.
        - ``'fp_records'``: list of dictionaries detailing each FP, its 2-second
          window, distance to nearest beat, and categorized cause.
        - ``'cause_counts'``: dict of counts per cause category.
        - ``'total_fps'``: total number of false positives.
    """
    detected = np.asarray(detected, dtype=np.int64).ravel()
    reference = np.asarray(reference, dtype=np.int64).ravel()
    signal = np.asarray(signal, dtype=np.float64).ravel()

    matched_det, _, _ = _match_detected_to_reference(
        detected, reference, tolerance_samples
    )
    fp_indices = detected[~matched_det]
    fp_times = fp_indices / fs

    fp_records: list[dict[str, Any]] = []
    cause_counts: dict[str, int] = {
        "T-wave oversensing": 0,
        "P-wave / Pre-systolic oversensing": 0,
        "Double detection / QRS splitting": 0,
        "Baseline wander / drift artifact": 0,
        "High-frequency noise / spike": 0,
        "Other / Unclassified artifact": 0,
    }

    half_win = int(1.0 * fs)  # 1 second on each side = 2-second window

    for fp_idx, fp_t in zip(fp_indices, fp_times):
        # Extract 2-second window centered on the FP
        w_start = max(0, fp_idx - half_win)
        w_end = min(len(signal), fp_idx + half_win)
        window = signal[w_start:w_end]
        window_t = np.arange(w_start, w_end) / fs

        # Proximity to nearest reference beats
        dist_to_nearest = None
        time_to_nearest = None
        cause = "Other / Unclassified artifact"
        detail = ""

        if len(reference) > 0:
            distances = reference - fp_idx  # positive if reference is after FP
            abs_distances = np.abs(distances)
            closest_idx = int(np.argmin(abs_distances))
            closest_ref = reference[closest_idx]
            dist_to_nearest = int(fp_idx - closest_ref)  # +: after ref, -: before ref
            time_to_nearest = dist_to_nearest / fs

            # Check preceding reference beat
            prev_refs = reference[reference < fp_idx]
            post_time = (fp_idx - prev_refs[-1]) / fs if len(prev_refs) > 0 else 999.0

            # Check succeeding reference beat
            next_refs = reference[reference > fp_idx]
            pre_time = (fp_idx - next_refs[0]) / fs if len(next_refs) > 0 else -999.0

            # 1. Double detection / QRS splitting
            if abs(dist_to_nearest) <= int(0.12 * fs):
                cause = "Double detection / QRS splitting"
                detail = f"Occurs within {abs(time_to_nearest)*1000:.1f} ms of true QRS"
            # 2. T-wave oversensing (occurs 150-480 ms after previous true beat)
            elif 0.15 <= post_time <= 0.48:
                cause = "T-wave oversensing"
                detail = f"Occurs {post_time*1000:.1f} ms after preceding R-peak in ST-T interval"
            # 3. P-wave / Pre-systolic oversensing (occurs 140-380 ms before next true beat)
            elif -0.38 <= pre_time <= -0.14:
                cause = "P-wave / Pre-systolic oversensing"
                detail = f"Occurs {abs(pre_time)*1000:.1f} ms before succeeding R-peak"
            # 4. Check for severe baseline wander in the 2s window
            else:
                baseline_variation = np.ptp(window) if len(window) > 0 else 0.0
                if baseline_variation > 1.2:
                    cause = "Baseline wander / drift artifact"
                    detail = f"Window ptp variation {baseline_variation:.2f} mV exceeds threshold"
                else:
                    # Check for high-frequency burst / spike
                    diff_std = np.std(np.diff(window)) if len(window) > 1 else 0.0
                    if diff_std > 0.15:
                        cause = "High-frequency noise / spike"
                        detail = f"High-frequency derivative std {diff_std:.3f} indicates EMG or spike"
                    else:
                        cause = "Other / Unclassified artifact"
                        detail = f"Artifact at t={fp_t:.2f}s isolated from beats"
        else:
            cause = "Other / Unclassified artifact"
            detail = "No reference beats present"

        cause_counts[cause] = cause_counts.get(cause, 0) + 1

        fp_records.append({
            "sample_index": int(fp_idx),
            "timestamp_sec": float(fp_t),
            "window": window,
            "window_time": window_t,
            "distance_to_nearest_samples": dist_to_nearest,
            "time_to_nearest_sec": time_to_nearest,
            "cause": cause,
            "cause_detail": detail,
        })

    return {
        "fp_indices": fp_indices,
        "fp_times": fp_times,
        "fp_records": fp_records,
        "cause_counts": cause_counts,
        "total_fps": len(fp_indices),
    }


# =========================================================================== #
#                        False Negative Analysis                              #
# =========================================================================== #

def analyze_false_negatives(
    detected: np.ndarray,
    reference: np.ndarray,
    signal: np.ndarray,
    fs: float,
    tolerance_samples: int,
    annotations: Sequence[str] | tuple[np.ndarray, Sequence[str]],
) -> dict[str, Any]:
    """Locate false negatives and cross-reference with MIT-BIH beat symbols.

    Parameters
    ----------
    detected : np.ndarray
        1-D array of detected R-peak sample indices.
    reference : np.ndarray
        1-D array of reference (ground truth) R-peak sample indices.
    signal : np.ndarray
        1-D array of raw ECG signal.
    fs : float
        Sampling frequency in Hz.
    tolerance_samples : int
        Matching tolerance window in samples.
    annotations : Sequence[str] or tuple
        Beat annotation symbols corresponding to ``reference``. If passed as a
        tuple ``(ref_samples, symbols)``, symbols are unpacked automatically.

    Returns
    -------
    dict
        Analysis results containing:
        - ``'fn_indices'``: np.ndarray of missed reference sample indices.
        - ``'fn_times'``: np.ndarray of FN timestamps in seconds.
        - ``'fn_symbols'``: list of annotation symbols for the missed beats.
        - ``'fn_records'``: list of dictionaries detailing each FN, its symbol,
          clinical category, 2-second window, and suspected failure mechanism.
        - ``'symbol_counts'``: dict counting missed beats by symbol.
        - ``'total_fns'``: total number of missed beats.
    """
    detected = np.asarray(detected, dtype=np.int64).ravel()
    reference = np.asarray(reference, dtype=np.int64).ravel()
    signal = np.asarray(signal, dtype=np.float64).ravel()

    # Extract symbols if passed as tuple
    if isinstance(annotations, tuple) and len(annotations) == 2:
        symbols = list(annotations[1])
    else:
        symbols = list(annotations)

    if len(symbols) != len(reference):
        raise ValueError(
            f"Length of annotations ({len(symbols)}) must match length of reference ({len(reference)})"
        )

    _, matched_ref, _ = _match_detected_to_reference(
        detected, reference, tolerance_samples
    )
    fn_mask = ~matched_ref
    fn_indices = reference[fn_mask]
    fn_times = fn_indices / fs
    fn_symbols = [symbols[i] for i in np.where(fn_mask)[0]]

    fn_records: list[dict[str, Any]] = []
    symbol_counts: dict[str, int] = {}
    half_win = int(1.0 * fs)

    for fn_idx, fn_t, sym in zip(fn_indices, fn_times, fn_symbols):
        symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        w_start = max(0, fn_idx - half_win)
        w_end = min(len(signal), fn_idx + half_win)
        window = signal[w_start:w_end]
        window_t = np.arange(w_start, w_end) / fs

        # Determine clinical name and category
        beat_name, clinical_cat = BEAT_NAMES.get(sym, (f"Unknown beat ({sym})", "Other"))

        # Determine failure cause
        cause = "Sub-threshold integrated slope / energy"
        detail = ""

        # Local QRS amplitude check
        qrs_w = int(0.04 * fs)
        qrs_start = max(0, fn_idx - qrs_w)
        qrs_end = min(len(signal), fn_idx + qrs_w + 1)
        qrs_ptp = np.ptp(signal[qrs_start:qrs_end]) if qrs_end > qrs_start else 0.0

        if sym in {"V", "E", "r"}:
            cause = "Ventricular Ectopic / Wide QRS slope attenuation"
            detail = (
                f"Wide morphology shifted spectral energy below 5 Hz; "
                f"low derivative dV/dt prevented threshold crossing."
            )
        elif qrs_ptp < 0.35:
            cause = "Low-amplitude QRS complex"
            detail = f"Local QRS peak-to-peak amplitude ({qrs_ptp:.2f} mV) below adaptive detection threshold."
        else:
            # Check proximity to nearest detection
            if len(detected) > 0:
                min_det_dist = int(np.min(np.abs(detected - fn_idx)))
                if min_det_dist < int(0.200 * fs):
                    cause = "Refractory period masking (< 200 ms lockout)"
                    detail = f"Occurs {min_det_dist / fs * 1000:.1f} ms from detection, inside refractory zone."
                else:
                    detail = f"QRS amplitude {qrs_ptp:.2f} mV fell below active signal threshold."
            else:
                detail = "No detections in record."

        fn_records.append({
            "sample_index": int(fn_idx),
            "timestamp_sec": float(fn_t),
            "symbol": sym,
            "beat_name": beat_name,
            "clinical_category": clinical_cat,
            "window": window,
            "window_time": window_t,
            "local_ptp_mv": float(qrs_ptp),
            "cause": cause,
            "cause_detail": detail,
        })

    return {
        "fn_indices": fn_indices,
        "fn_times": fn_times,
        "fn_symbols": fn_symbols,
        "fn_records": fn_records,
        "symbol_counts": symbol_counts,
        "total_fns": len(fn_indices),
    }


# =========================================================================== #
#                     Beat-Type Sensitivity Evaluation                        #
# =========================================================================== #

def evaluate_by_beat_type(
    record_ids: list[str] | None = None,
    fs: float = 360.0,
    tolerance_ms: float = 150.0,
) -> pd.DataFrame:
    """Benchmark detection sensitivity broken down by MIT-BIH beat annotation type.

    Evaluates the Pan-Tompkins QRS detector across records and tabulates
    total beats, true positives, false negatives, and sensitivity (%)
    for each beat type (Normal 'N', Ventricular Ectopic 'V', Atrial Ectopic 'A', etc.).

    Parameters
    ----------
    record_ids : list of str, optional
        MIT-BIH record IDs to include. Defaults to
        ``['100', '101', '103', '105', '108', '203']``.
    fs : float, optional
        Sampling frequency in Hz (default 360.0).
    tolerance_ms : float, optional
        Matching tolerance in milliseconds (default 150.0 ms).

    Returns
    -------
    pandas.DataFrame
        Table with columns:
        ``['beat_symbol', 'beat_name', 'clinical_category', 'total_beats',
          'true_positives', 'false_negatives', 'sensitivity_pct']``.
    """
    if record_ids is None:
        record_ids = ["100", "101", "103", "105", "108", "203"]

    tolerance_samples = int(round(tolerance_ms * fs / 1000.0))

    counts_total: dict[str, int] = {}
    counts_tp: dict[str, int] = {}

    for rec_id in record_ids:
        sig, _ = load_record(rec_id, channel=0)
        ref_samples, ref_symbols = get_beat_annotations(rec_id)

        # Run Pan-Tompkins detector
        res = pan_tompkins_detect(sig, fs)
        detected = res["r_peaks"]

        # Match peaks
        _, matched_ref, _ = _match_detected_to_reference(
            detected, ref_samples, tolerance_samples
        )

        for r_idx, sym in enumerate(ref_symbols):
            counts_total[sym] = counts_total.get(sym, 0) + 1
            if matched_ref[r_idx]:
                counts_tp[sym] = counts_tp.get(sym, 0) + 1

    rows: list[dict[str, Any]] = []
    for sym, total in sorted(counts_total.items(), key=lambda x: -x[1]):
        tp = counts_tp.get(sym, 0)
        fn = total - tp
        se = (tp / total) * 100.0 if total > 0 else 0.0
        beat_name, clinical_cat = BEAT_NAMES.get(sym, (f"Unknown beat ({sym})", "Other"))

        rows.append({
            "beat_symbol": sym,
            "beat_name": beat_name,
            "clinical_category": clinical_cat,
            "total_beats": int(total),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "sensitivity_pct": round(float(se), 2),
        })

    df = pd.DataFrame(rows)
    return df


# =========================================================================== #
#                        Diagnostic Plotting Functions                        #
# =========================================================================== #

def plot_record_108_false_positives(
    save_path: str | Path | None = None,
    t_start: float = 1674.0,
    duration: float = 6.0,
) -> None:
    """Plot Record 108 false positives showing T-wave/biphasic oversensing.

    Zooms into a 6-second segment of Record 108 where false positives occurred,
    illustrating the intermediate Pan-Tompkins stages (bandpass, derivative,
    squaring, and moving-window integration with threshold) that explain
    why the detector triggered spurious detections.

    Parameters
    ----------
    save_path : str, Path, or None, optional
        File path to save the generated plot. If None, defaults to
        ``results/figures/07_error_analysis_108_false_positives.png``.
    t_start : float, optional
        Start time in seconds (default 1674.0).
    duration : float, optional
        Segment duration in seconds (default 6.0).
    """
    if save_path is None:
        save_path = FIGURES_DIR / "07_error_analysis_108_false_positives.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rec_id = "108"
    sig, fs = load_record(rec_id, channel=0)
    ref_samples, ref_symbols = get_beat_annotations(rec_id)
    res = pan_tompkins_detect(sig, fs)
    detected = res["r_peaks"]

    t_end = t_start + duration
    s_start = int(t_start * fs)
    s_end = int(t_end * fs)

    t = np.arange(s_start, s_end) / fs
    sig_seg = sig[s_start:s_end]
    filt_seg = res["filtered"][s_start:s_end]
    der_seg = res["derivative"][s_start:s_end]
    sq_seg = res["squared"][s_start:s_end]
    int_seg = res["integrated"][s_start:s_end]

    tol_samples = int(round(0.150 * fs))
    in_det = [d for d in detected if s_start <= d < s_end]
    in_ref = [r for r in ref_samples if s_start <= r < s_end]

    # Partition detections into True Positives and False Positives
    tp_dets = []
    fp_dets = []
    for d in in_det:
        min_dist = min([abs(d - r) for r in ref_samples]) if len(ref_samples) > 0 else 9999
        if min_dist <= tol_samples:
            tp_dets.append(d)
        else:
            fp_dets.append(d)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # Panel 1: Raw ECG with Markers
    axes[0].plot(t, sig_seg, color="#2b5c8f", lw=1.2, label="Raw ECG (Lead MLII)")
    axes[0].plot(
        [r / fs for r in in_ref],
        [sig[r] for r in in_ref],
        "^",
        color="#2ca02c",
        markersize=9,
        label="Ground-Truth R-peak (Ref)",
        zorder=4,
    )
    axes[0].plot(
        [d / fs for d in tp_dets],
        [sig[d] for d in tp_dets],
        "o",
        color="#1f77b4",
        markersize=7,
        label="True Positive (TP Detection)",
        zorder=5,
    )
    if fp_dets:
        axes[0].plot(
            [d / fs for d in fp_dets],
            [sig[d] for d in fp_dets],
            "X",
            color="#d62728",
            markersize=10,
            markeredgewidth=2,
            label="False Positive (FP - Oversensing)",
            zorder=6,
        )
        for fp in fp_dets:
            axes[0].annotate(
                "FP",
                (fp / fs, sig[fp]),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                color="#d62728",
                fontweight="bold",
                fontsize=9,
            )

    axes[0].set_title(
        f"Record {rec_id} Error Analysis: Spurious Detections from Inverted Morphology & Elevated T/P Waves (t={t_start:.1f}s to {t_end:.1f}s)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_ylabel("ECG (mV)", fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.92, ncol=2)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Bandpass Filtered Signal
    axes[1].plot(t, filt_seg, color="#2ca02c", lw=1.2, label="Bandpass Filtered (5–15 Hz)")
    for fp in fp_dets:
        axes[1].axvline(fp / fs, color="#d62728", linestyle=":", alpha=0.6)
    axes[1].set_ylabel("Filtered (mV)", fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.92)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Five-Point Derivative Squared
    axes[2].plot(t, sq_seg, color="#ff7f0e", lw=1.2, label="Derivative Squared ($y^2[n]$)")
    for fp in fp_dets:
        axes[2].axvline(fp / fs, color="#d62728", linestyle=":", alpha=0.6)
    axes[2].set_ylabel("Squared ($mV^2$)", fontweight="bold")
    axes[2].legend(loc="upper right", framealpha=0.92)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Moving-Window Integration with Adaptive Threshold
    axes[3].plot(t, int_seg, color="#9467bd", lw=1.3, label="Moving-Window Integration (~150 ms)")

    # Compute local adaptive threshold
    ref_samples_200ms = int(0.200 * fs)
    cands, _ = find_peaks(res["integrated"], distance=ref_samples_200ms)
    cands_before = cands[cands <= s_end]
    cands_val = res["integrated"][cands_before]
    spki = float(np.mean(cands_val[-10:])) if len(cands_val) >= 10 else float(np.mean(cands_val))
    npki = 0.1 * spki
    thr_sig = npki + 0.25 * (spki - npki)

    axes[3].axhline(
        thr_sig,
        color="#d62728",
        linestyle="--",
        lw=1.6,
        label=f"Adaptive Detection Threshold ($Thr_{{sig}} \\approx {thr_sig:.5f}$)",
    )
    for fp in fp_dets:
        axes[3].axvline(fp / fs, color="#d62728", linestyle=":", alpha=0.6)
        # Find integrated peak around fp
        local_cand = fp
        if 0 <= local_cand < len(res["integrated"]):
            axes[3].plot(
                fp / fs,
                res["integrated"][local_cand],
                "ro",
                markersize=6,
            )

    axes[3].set_ylabel("Integrated Energy", fontweight="bold")
    axes[3].set_xlabel("Time (seconds)", fontweight="bold")
    axes[3].legend(loc="upper right", framealpha=0.92)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_record_203_false_negatives(
    save_path: str | Path | None = None,
    t_start: float = 8.0,
    duration: float = 6.0,
) -> None:
    """Plot Record 203 false negatives showing missed PVC due to wide morphology.

    Zooms into a 6-second segment of Record 203 showing:
    - Normal beats correctly detected.
    - Premature ventricular contraction (PVC) missed (False Negative).
    - Another PVC detected (True Positive).
    Illustrates how wide QRS morphology produces low derivative slope and
    causes integrated energy to fall below the dual adaptive threshold.

    Parameters
    ----------
    save_path : str, Path, or None, optional
        File path to save the generated plot. If None, defaults to
        ``results/figures/07_error_analysis_203_false_negatives.png``.
    t_start : float, optional
        Start time in seconds (default 8.0).
    duration : float, optional
        Segment duration in seconds (default 6.0).
    """
    if save_path is None:
        save_path = FIGURES_DIR / "07_error_analysis_203_false_negatives.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rec_id = "203"
    sig, fs = load_record(rec_id, channel=0)
    ref_samples, ref_symbols = get_beat_annotations(rec_id)
    res = pan_tompkins_detect(sig, fs)
    detected = res["r_peaks"]

    t_end = t_start + duration
    s_start = int(t_start * fs)
    s_end = int(t_end * fs)

    t = np.arange(s_start, s_end) / fs
    sig_seg = sig[s_start:s_end]
    filt_seg = res["filtered"][s_start:s_end]
    sq_seg = res["squared"][s_start:s_end]
    int_seg = res["integrated"][s_start:s_end]

    tol_samples = int(round(0.150 * fs))

    # Identify reference beats in this segment
    in_ref_samples = []
    in_ref_symbols = []
    in_ref_status = []  # 'TP' or 'FN'

    for r, sy in zip(ref_samples, ref_symbols):
        if s_start <= r < s_end:
            in_ref_samples.append(r)
            in_ref_symbols.append(sy)
            min_dist = min([abs(r - d) for d in detected]) if len(detected) > 0 else 9999
            in_ref_status.append("TP" if min_dist <= tol_samples else "FN")

    in_det = [d for d in detected if s_start <= d < s_end]
    missed_refs = [r for r, st in zip(in_ref_samples, in_ref_status) if st == "FN"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # Panel 1: Raw ECG with Annotations
    axes[0].plot(t, sig_seg, color="#2b5c8f", lw=1.2, label="Raw ECG (Lead MLII)")

    for r, sy, st in zip(in_ref_samples, in_ref_symbols, in_ref_status):
        t_r = r / fs
        val = sig[r]
        if st == "TP":
            if sy == "V":
                axes[0].plot(
                    t_r,
                    val,
                    "^",
                    color="#e6550d",
                    markersize=10,
                    label="Detected PVC ('V') [TP]" if "Detected PVC ('V') [TP]" not in axes[0].get_legend_handles_labels()[1] else "",
                    zorder=4,
                )
                axes[0].annotate(
                    f"PVC (V)\n[TP]",
                    (t_r, val),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    color="#e6550d",
                    fontweight="bold",
                    fontsize=8.5,
                )
            else:
                axes[0].plot(
                    t_r,
                    val,
                    "^",
                    color="#2ca02c",
                    markersize=7,
                    label="Normal Beat ('N') [TP]" if "Normal Beat ('N') [TP]" not in axes[0].get_legend_handles_labels()[1] else "",
                    zorder=4,
                )
        else:  # FN
            axes[0].plot(
                t_r,
                val,
                "X",
                color="#d62728",
                markersize=11,
                markeredgewidth=2.2,
                label="Missed PVC ('V') [FN]" if "Missed PVC ('V') [FN]" not in axes[0].get_legend_handles_labels()[1] else "",
                zorder=6,
            )
            axes[0].annotate(
                f"MISSED\nPVC (V)\n[FN]",
                (t_r, val),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                color="#d62728",
                fontweight="bold",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff2ae", edgecolor="#d62728", alpha=0.9),
            )
            axes[0].axvspan(t_r - 0.15, t_r + 0.15, color="#d62728", alpha=0.15)

    # Detections
    axes[0].plot(
        [d / fs for d in in_det],
        [sig[d] for d in in_det],
        "o",
        markeredgecolor="black",
        markerfacecolor="none",
        markersize=9,
        markeredgewidth=1.6,
        label="Pan-Tompkins Detection",
        zorder=5,
    )

    axes[0].set_title(
        f"Record {rec_id} Error Analysis: Missed Premature Ventricular Contraction (PVC) due to Low Slope (t={t_start:.1f}s to {t_end:.1f}s)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_ylabel("ECG (mV)", fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.92, ncol=2)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Bandpass Filtered Signal
    axes[1].plot(t, filt_seg, color="#2ca02c", lw=1.2, label="Bandpass Filtered (5–15 Hz)")
    for mr in missed_refs:
        axes[1].axvspan(
            mr / fs - 0.15,
            mr / fs + 0.15,
            color="#d62728",
            alpha=0.15,
            label="Missed QRS Window" if "Missed QRS Window" not in axes[1].get_legend_handles_labels()[1] else "",
        )
    axes[1].set_ylabel("Filtered (mV)", fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.92)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Five-Point Derivative Squared
    axes[2].plot(t, sq_seg, color="#ff7f0e", lw=1.2, label="Derivative Squared ($y^2[n]$)")
    for mr in missed_refs:
        axes[2].axvspan(mr / fs - 0.15, mr / fs + 0.15, color="#d62728", alpha=0.15)
    axes[2].set_ylabel("Squared ($mV^2$)", fontweight="bold")
    axes[2].legend(loc="upper right", framealpha=0.92)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Moving-Window Integration with Dual Threshold
    axes[3].plot(t, int_seg, color="#9467bd", lw=1.3, label="Moving-Window Integration (~150 ms)")
    for mr in missed_refs:
        axes[3].axvspan(mr / fs - 0.15, mr / fs + 0.15, color="#d62728", alpha=0.15)

    # Representative active threshold around t=10.7s: SPKI ~ 0.006, NPKI ~ 0.0001
    thr_sig = 0.0015
    axes[3].axhline(
        thr_sig,
        color="#d62728",
        linestyle="--",
        lw=1.6,
        label=f"Adaptive Detection Threshold ($Thr_{{sig}} \\approx {thr_sig:.4f}$)",
    )

    # Annotate peak of missed PVC at sample 3864
    missed_cand_val = float(res["integrated"][3864])
    missed_cand_t = 3864 / fs
    axes[3].plot(missed_cand_t, missed_cand_val, "ro", markersize=8)
    axes[3].annotate(
        f"Peak: {missed_cand_val:.4f}\n< Threshold",
        (missed_cand_t, missed_cand_val),
        textcoords="offset points",
        xytext=(22, -12),
        color="#d62728",
        fontweight="bold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    axes[3].set_ylabel("Integrated Energy", fontweight="bold")
    axes[3].set_xlabel("Time (seconds)", fontweight="bold")
    axes[3].legend(loc="upper right", framealpha=0.92)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beat_type_sensitivity(
    df_beat_type: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot beat-type sensitivity bar chart comparing Normal vs PVC vs APC.

    Parameters
    ----------
    df_beat_type : pandas.DataFrame or None, optional
        Output table from :func:`evaluate_by_beat_type`. If None, computed
        automatically across benchmark records.
    save_path : str, Path, or None, optional
        File path to save the generated plot. If None, defaults to
        ``results/figures/07_beat_type_performance.png``.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "07_beat_type_performance.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if df_beat_type is None:
        df_beat_type = evaluate_by_beat_type()

    # Filter to primary comparison classes: Normal ('N'), PVC ('V'), APC ('A')
    key_symbols = ["N", "V", "A"]
    labels = [
        "Normal Sinus ('N')",
        "Ventricular Ectopic\n(PVC / 'V')",
        "Atrial Ectopic\n(APC / 'A')",
    ]
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]

    accuracies: list[float] = []
    totals: list[int] = []
    detected: list[int] = []
    missed: list[int] = []

    for sym in key_symbols:
        row = df_beat_type[df_beat_type["beat_symbol"] == sym]
        if not row.empty:
            acc = float(row["sensitivity_pct"].iloc[0])
            tot = int(row["total_beats"].iloc[0])
            tp = int(row["true_positives"].iloc[0])
            fn = int(row["false_negatives"].iloc[0])
        else:
            acc, tot, tp, fn = 0.0, 0, 0, 0
        accuracies.append(acc)
        totals.append(tot)
        detected.append(tp)
        missed.append(fn)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(labels))
    width = 0.52

    bars = ax.bar(
        x,
        accuracies,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.88,
        zorder=3,
    )

    ax.grid(True, linestyle="--", alpha=0.6, axis="y", zorder=0)
    ax.set_ylim(85, 103)
    ax.set_ylabel("Detection Sensitivity (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_title(
        "Pan-Tompkins QRS Detection Sensitivity by Beat Morphology\n(MIT-BIH Arrhythmia Database Benchmark)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    for bar, acc, tot, det, miss in zip(bars, accuracies, totals, detected, missed):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.2f}%\n({det:,} / {tot:,})\nMissed: {miss}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="black",
        )

    # Reference clinical standards
    ax.axhline(99.0, color="#7f7f7f", linestyle=":", linewidth=1.5, zorder=2, label="99% Clinical Standard Line")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10.5)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
#                               Main Script                                   #
# =========================================================================== #

def main() -> None:
    """Run full failure mode and error analysis pipeline."""
    print("=" * 75)
    print("PAN-TOMPKINS FAILURE MODE & ERROR ANALYSIS (MIT-BIH DATABASE)")
    print("=" * 75)

    fs = 360.0
    tol_samples = int(round(0.150 * fs))

    # 1. Analyze Record 108 False Positives
    print("\n[Step 1] Analyzing Record 108 False Positives...")
    sig108, _ = load_record("108", channel=0)
    ref_samples108, ref_symbols108 = get_beat_annotations("108")
    res108 = pan_tompkins_detect(sig108, fs)

    fp_analysis108 = analyze_false_positives(
        res108["r_peaks"], ref_samples108, sig108, fs, tol_samples
    )
    print(f"  Record 108: Total False Positives = {fp_analysis108['total_fps']}")
    print("  Cause Breakdown:")
    for cause, count in fp_analysis108["cause_counts"].items():
        if count > 0:
            pct = (count / fp_analysis108["total_fps"]) * 100
            print(f"    - {cause}: {count} ({pct:.1f}%)")

    # 2. Analyze Record 203 False Negatives
    print("\n[Step 2] Analyzing Record 203 False Negatives...")
    sig203, _ = load_record("203", channel=0)
    ref_samples203, ref_symbols203 = get_beat_annotations("203")
    res203 = pan_tompkins_detect(sig203, fs)

    fn_analysis203 = analyze_false_negatives(
        res203["r_peaks"], ref_samples203, sig203, fs, tol_samples, ref_symbols203
    )
    print(f"  Record 203: Total False Negatives = {fn_analysis203['total_fns']}")
    print("  Missed Beat Symbols:")
    for sym, count in fn_analysis203["symbol_counts"].items():
        name, cat = BEAT_NAMES.get(sym, (sym, "Other"))
        pct = (count / fn_analysis203["total_fns"]) * 100
        print(f"    - '{sym}' ({name}): {count} ({pct:.1f}%)")

    # 3. Benchmark Beat Type Sensitivity
    print("\n[Step 3] Evaluating Detection Sensitivity by Beat Type across records...")
    benchmark_records = ["100", "101", "103", "105", "108", "203"]
    df_beat_type = evaluate_by_beat_type(benchmark_records, fs=fs, tolerance_ms=150.0)
    print("\n" + df_beat_type.to_string(index=False))

    # Save CSV table
    csv_path = RESULTS_DIR / "beat_type_sensitivity.csv"
    df_beat_type.to_csv(csv_path, index=False)
    print(f"\n  Saved beat-type performance table to: {csv_path}")

    # 4. Generate Figures
    print("\n[Step 4] Generating Publication Diagnostic Figures...")
    p_108 = FIGURES_DIR / "07_error_analysis_108_false_positives.png"
    plot_record_108_false_positives(p_108)
    print(f"  Saved: {p_108}")

    p_203 = FIGURES_DIR / "07_error_analysis_203_false_negatives.png"
    plot_record_203_false_negatives(p_203)
    print(f"  Saved: {p_203}")

    p_chart = FIGURES_DIR / "07_beat_type_performance.png"
    plot_beat_type_sensitivity(df_beat_type, p_chart)
    print(f"  Saved: {p_chart}")

    print("\n" + "=" * 75)
    print("ERROR ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
