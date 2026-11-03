"""
validation.py — R-peak detection validation against MIT-BIH ground truth.

Compares detected R-peak locations against reference annotations from the
MIT-BIH Arrhythmia Database and computes standard detection performance
metrics (sensitivity, positive predictivity, F1, detection error rate).

Typical usage:
    >>> from validation import validate_record
    >>> results = validate_record(detected_peaks, '100', fs=360, tolerance_ms=150.0)
    >>> print(f"Se={results['sensitivity']:.4f}  PPV={results['positive_predictivity']:.4f}")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import wfdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the MIT-BIH Arrhythmia Database relative to the project root
DATA_DIR = Path(__file__).resolve().parent.parent / (
    "mitdb-1.0.0" / Path("mit-bih-arrhythmia-database-1.0.0")
)

# AAMI beat-annotation symbols (all annotations that mark actual heartbeats)
# Reference: https://www.physionet.org/physiobank/annotations.shtml
BEAT_ANNOTATION_SYMBOLS = {
    "N",  # Normal beat
    "L",  # Left bundle branch block beat
    "R",  # Right bundle branch block beat
    "B",  # Bundle branch block beat (unspecified)
    "A",  # Atrial premature beat
    "a",  # Aberrated atrial premature beat
    "J",  # Junctional premature beat
    "S",  # Supraventricular premature beat (or ectopic)
    "V",  # Premature ventricular contraction
    "r",  # R-on-T premature ventricular contraction
    "F",  # Fusion of ventricular and normal beat
    "e",  # Atrial escape beat
    "j",  # Junctional escape beat
    "n",  # Supraventricular escape beat
    "E",  # Ventricular escape beat
    "/",  # Paced beat
    "f",  # Fusion of paced and normal beat
    "Q",  # Unclassifiable beat
}


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------


def match_peaks(
    detected: np.ndarray,
    reference: np.ndarray,
    tolerance_samples: int,
) -> tuple[int, int, int]:
    """Match detected peaks to reference peaks within a tolerance window.

    Each reference peak can be matched to at most one detected peak (the
    closest one within the tolerance window).  Greedy matching is performed
    in order of increasing absolute distance so that the best matches are
    consumed first.

    Parameters
    ----------
    detected : np.ndarray
        1-D array of detected R-peak sample indices (need not be sorted).
    reference : np.ndarray
        1-D array of reference (ground-truth) R-peak sample indices.
    tolerance_samples : int
        Maximum allowed distance (in samples) between a detected peak and a
        reference peak for them to be considered a match.

    Returns
    -------
    tuple[int, int, int]
        ``(TP, FP, FN)`` — true positives, false positives, false negatives.
    """
    detected = np.asarray(detected, dtype=np.int64).ravel()
    reference = np.asarray(reference, dtype=np.int64).ravel()

    if detected.size == 0:
        return 0, 0, int(reference.size)
    if reference.size == 0:
        return 0, int(detected.size), 0

    # Track which detections / references have been matched
    matched_det = np.zeros(detected.size, dtype=bool)
    matched_ref = np.zeros(reference.size, dtype=bool)

    # Build all candidate pairs (det_idx, ref_idx, |distance|)
    # For large arrays this is O(N*M); for typical MIT-BIH records (~2000
    # beats) this is perfectly tractable.
    pairs: list[tuple[int, int, int]] = []
    for d_idx, d_val in enumerate(detected):
        distances = np.abs(reference - d_val)
        within = np.where(distances <= tolerance_samples)[0]
        for r_idx in within:
            pairs.append((d_idx, r_idx, int(distances[r_idx])))

    # Sort by distance so closest matches are consumed first (greedy)
    pairs.sort(key=lambda x: x[2])

    tp = 0
    for d_idx, r_idx, _ in pairs:
        if not matched_det[d_idx] and not matched_ref[r_idx]:
            matched_det[d_idx] = True
            matched_ref[r_idx] = True
            tp += 1

    fp = int(np.sum(~matched_det))
    fn = int(np.sum(~matched_ref))

    return tp, fp, fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    """Compute standard beat-detection performance metrics.

    Parameters
    ----------
    tp : int
        True positives — correctly detected beats.
    fp : int
        False positives — spurious detections.
    fn : int
        False negatives — missed beats.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'sensitivity'`` (Se, recall): TP / (TP + FN)
        - ``'positive_predictivity'`` (PPV, precision): TP / (TP + FP)
        - ``'f1_score'``: harmonic mean of Se and PPV
        - ``'detection_error_rate'`` (DER): (FP + FN) / (TP + FN)
        - ``'tp'``, ``'fp'``, ``'fn'``: raw counts
        - ``'total_ref_beats'``: TP + FN (total reference beats)
    """
    total_ref = tp + fn

    sensitivity = tp / total_ref if total_ref > 0 else 0.0
    positive_predictivity = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    se_ppv_sum = sensitivity + positive_predictivity
    f1_score = (
        2.0 * sensitivity * positive_predictivity / se_ppv_sum
        if se_ppv_sum > 0
        else 0.0
    )

    detection_error_rate = (fp + fn) / total_ref if total_ref > 0 else 0.0

    return {
        "sensitivity": sensitivity,
        "positive_predictivity": positive_predictivity,
        "f1_score": f1_score,
        "detection_error_rate": detection_error_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_ref_beats": total_ref,
    }


# ---------------------------------------------------------------------------
# Record-level validation
# ---------------------------------------------------------------------------


def _load_beat_annotations(record_id: str) -> np.ndarray:
    """Load beat-only annotation sample indices for a MIT-BIH record.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. ``'100'``.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of sample indices corresponding to beat annotations.
    """
    record_path = str(DATA_DIR / record_id)
    ann = wfdb.rdann(record_path, "atr")

    # Filter to beat annotations only (exclude rhythm changes, comments, etc.)
    beat_mask = np.array(
        [sym in BEAT_ANNOTATION_SYMBOLS for sym in ann.symbol], dtype=bool
    )
    return ann.sample[beat_mask]


def validate_record(
    detected_peaks: np.ndarray,
    record_id: str,
    fs: float = 360.0,
    tolerance_ms: float = 150.0,
) -> dict:
    """Validate detected R-peaks against MIT-BIH annotations for one record.

    Loads the ground-truth beat annotations for *record_id*, converts the
    tolerance from milliseconds to samples, and computes matching + metrics.

    Parameters
    ----------
    detected_peaks : np.ndarray
        1-D array of detected R-peak sample indices.
    record_id : str
        MIT-BIH record identifier (e.g. ``'100'``).
    fs : float, optional
        Sampling frequency in Hz (default 360).
    tolerance_ms : float, optional
        Tolerance window in milliseconds (default 150 ms, ≈ 54 samples at
        360 Hz).

    Returns
    -------
    dict
        Performance metrics dictionary (see :func:`compute_metrics`) plus
        an additional ``'record_id'`` key.
    """
    tolerance_samples = int(round(tolerance_ms * fs / 1000.0))
    reference_peaks = _load_beat_annotations(record_id)

    tp, fp, fn = match_peaks(
        detected=np.asarray(detected_peaks),
        reference=reference_peaks,
        tolerance_samples=tolerance_samples,
    )

    metrics = compute_metrics(tp, fp, fn)
    metrics["record_id"] = record_id
    return metrics


# ---------------------------------------------------------------------------
# Multi-record validation
# ---------------------------------------------------------------------------


def validate_multiple_records(
    detection_func: Callable[[np.ndarray, float], np.ndarray],
    record_ids: list[str],
    fs: float = 360.0,
    tolerance_ms: float = 150.0,
):
    """Run a detection function on multiple MIT-BIH records and validate.

    For each record the two-channel ECG signal is loaded, *detection_func*
    is called with ``(signal, fs)`` where *signal* is the full 2-D array
    (samples × channels), and the returned R-peak indices are validated
    against the ground-truth annotations.

    Parameters
    ----------
    detection_func : callable
        ``detection_func(signal, fs) -> np.ndarray`` — takes a 2-D signal
        array and sampling rate, returns a 1-D array of R-peak sample
        indices.
    record_ids : list[str]
        List of MIT-BIH record identifiers (e.g. ``['100', '101', …]``).
    fs : float, optional
        Sampling frequency in Hz (default 360).
    tolerance_ms : float, optional
        Tolerance window in milliseconds (default 150 ms).

    Returns
    -------
    pandas.DataFrame or dict[str, dict]
        If *pandas* is available, a DataFrame with one row per record plus
        an ``'AGGREGATE'`` summary row.  Otherwise a plain ``dict`` of
        dicts keyed by record id (and ``'AGGREGATE'``).
    """
    all_results: list[dict] = []

    for rec_id in record_ids:
        record_path = str(DATA_DIR / rec_id)
        record = wfdb.rdrecord(record_path)
        if record.p_signal.ndim > 1:
            signal = record.p_signal[:, 0]
        else:
            signal = record.p_signal

        detected_peaks = detection_func(signal, fs)
        metrics = validate_record(detected_peaks, rec_id, fs=fs, tolerance_ms=tolerance_ms)
        all_results.append(metrics)

    # Compute aggregate metrics across all records
    agg_tp = sum(r["tp"] for r in all_results)
    agg_fp = sum(r["fp"] for r in all_results)
    agg_fn = sum(r["fn"] for r in all_results)
    agg_metrics = compute_metrics(agg_tp, agg_fp, agg_fn)
    agg_metrics["record_id"] = "AGGREGATE"
    all_results.append(agg_metrics)

    # Try to return a pandas DataFrame; fall back to dict of dicts
    try:
        import pandas as pd

        df = pd.DataFrame(all_results)
        # Reorder columns for readability
        col_order = [
            "record_id",
            "sensitivity",
            "positive_predictivity",
            "f1_score",
            "detection_error_rate",
            "tp",
            "fp",
            "fn",
            "total_ref_beats",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        return df
    except ImportError:
        return {r["record_id"]: r for r in all_results}
