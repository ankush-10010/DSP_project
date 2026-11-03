"""
Feature Extraction Module for ECG Signal Processing Pipeline
=============================================================

Computes heart rate and heart rate variability (HRV) metrics from detected
R-peaks. Supports both time-domain and frequency-domain HRV analysis.

Designed for use with the MIT-BIH Arrhythmia Database (360 Hz sampling rate).

References:
    - Task Force of ESC/NASPE, "Heart rate variability: standards of
      measurement, physiological interpretation, and clinical use," 1996.
"""

import numpy as np
from scipy import interpolate, signal

from src.poincare import compute_poincare_metrics


def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Compute R-R intervals in seconds from detected R-peak sample indices.

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak sample indices (integer or float).
    fs : float
        Sampling frequency in Hz (e.g., 360 Hz for MIT-BIH).

    Returns
    -------
    np.ndarray
        R-R intervals in seconds.  Length is ``len(r_peaks) - 1``.

    Raises
    ------
    ValueError
        If fewer than 2 R-peaks are provided.
    """
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    if len(r_peaks) < 2:
        raise ValueError("At least 2 R-peaks are required to compute RR intervals.")
    return np.diff(r_peaks) / fs


def compute_heart_rate(rr_intervals: np.ndarray) -> np.ndarray:
    """Compute instantaneous heart rate from R-R intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    np.ndarray
        Instantaneous heart rate in beats per minute (BPM).
        ``HR = 60 / RR``
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    if np.any(rr_intervals <= 0):
        raise ValueError("All RR intervals must be positive.")
    return 60.0 / rr_intervals


def compute_mean_hr(rr_intervals: np.ndarray) -> float:
    """Compute mean heart rate from R-R intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    float
        Mean heart rate in BPM.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    return float(np.mean(compute_heart_rate(rr_intervals)))


def compute_sdnn(rr_intervals: np.ndarray) -> float:
    """Compute SDNN — standard deviation of NN (R-R) intervals.

    SDNN reflects all cyclic components responsible for variability in the
    recording period.

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    float
        SDNN in milliseconds.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    return float(np.std(rr_intervals, ddof=1) * 1000.0)


def compute_rmssd(rr_intervals: np.ndarray) -> float:
    """Compute RMSSD — root mean square of successive RR interval differences.

    RMSSD is a primary time-domain measure of short-term HRV, reflecting
    parasympathetic (vagal) cardiac regulation.

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    float
        RMSSD in milliseconds.

    Raises
    ------
    ValueError
        If fewer than 2 RR intervals are provided.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals are required for RMSSD.")
    successive_diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(successive_diffs ** 2)) * 1000.0)


def compute_pnn50(rr_intervals: np.ndarray) -> float:
    """Compute pNN50 — percentage of successive RR differences exceeding 50 ms.

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    float
        pNN50 as a percentage (0–100).

    Raises
    ------
    ValueError
        If fewer than 2 RR intervals are provided.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals are required for pNN50.")
    successive_diffs = np.abs(np.diff(rr_intervals))
    nn50 = np.sum(successive_diffs > 0.050)  # 50 ms threshold
    return float(nn50 / len(successive_diffs) * 100.0)


def compute_hrv_frequency(rr_intervals: np.ndarray) -> dict:
    """Compute frequency-domain HRV metrics via Welch's method.

    The RR interval series is first interpolated onto a uniformly-sampled
    4 Hz time grid using cubic spline interpolation, then the power spectral
    density (PSD) is estimated with Welch's method.

    Frequency bands (Task Force, 1996):
        - VLF : 0.003 – 0.04  Hz
        - LF  : 0.04  – 0.15  Hz
        - HF  : 0.15  – 0.40  Hz

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R intervals in seconds.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'vlf_power'``   : float — VLF band power (ms²)
        - ``'lf_power'``    : float — LF band power (ms²)
        - ``'hf_power'``    : float — HF band power (ms²)
        - ``'lf_hf_ratio'`` : float — LF / HF power ratio
        - ``'total_power'`` : float — Total power (VLF + LF + HF) (ms²)

    Raises
    ------
    ValueError
        If the RR interval series is too short for frequency analysis.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64)
    if len(rr_intervals) < 4:
        raise ValueError(
            "At least 4 RR intervals are required for frequency-domain analysis."
        )

    # --- Interpolate to uniform 4 Hz sampling ---
    fs_interp = 4.0  # Hz
    # Cumulative time axis for each RR interval (starting at 0)
    rr_cumsum = np.cumsum(rr_intervals)
    rr_times = np.concatenate(([0.0], rr_cumsum))
    # RR value at each time point: duplicate first value at t=0
    rr_values = np.concatenate(([rr_intervals[0]], rr_intervals))

    # Cubic spline interpolation
    spline = interpolate.CubicSpline(rr_times, rr_values)
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs_interp)
    rr_interp = spline(t_uniform)

    # Remove mean (detrend) before spectral estimation
    rr_interp = rr_interp - np.mean(rr_interp)

    # --- PSD via Welch's method ---
    # Use a 256-sample window or the signal length, whichever is smaller
    nperseg = min(256, len(rr_interp))
    freqs, psd = signal.welch(
        rr_interp,
        fs=fs_interp,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    # PSD is in s²/Hz; convert to ms²/Hz  (×1e6)
    psd_ms2 = psd * 1e6

    # --- Band power integration via trapezoidal rule ---
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    # np.trapezoid (NumPy ≥ 2.0) replaces the deprecated np.trapz
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    vlf_power = float(_trapz(psd_ms2[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    lf_power = float(_trapz(psd_ms2[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(_trapz(psd_ms2[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else float("inf")
    total_power = vlf_power + lf_power + hf_power

    return {
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "total_power": total_power,
        "frequencies": freqs,
        "psd": psd_ms2,
    }


def compute_all_metrics(
    r_peaks: np.ndarray,
    fs: float,
    include_poincare: bool = True,
) -> dict:
    """Compute all heart rate and HRV metrics from R-peak indices.

    This is a convenience function that calls every individual metric function
    and returns the results in a single dictionary.

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak sample indices.
    fs : float
        Sampling frequency in Hz.
    include_poincare : bool, optional
        Whether to compute non-linear Poincaré plot metrics (SD1, SD2, ratio,
        and ellipse parameters). Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - ``'rr_intervals'``       : np.ndarray — RR intervals (s)
        - ``'heart_rate'``         : np.ndarray — Instantaneous HR (BPM)
        - ``'mean_hr'``            : float — Mean heart rate (BPM)
        - ``'sdnn'``               : float — SDNN (ms)
        - ``'rmssd'``              : float — RMSSD (ms)
        - ``'pnn50'``              : float — pNN50 (%)
        - ``'hrv_frequency'``      : dict — Frequency-domain HRV metrics
        - ``'poincare'``           : dict (optional) — Poincaré HRV metrics
        - ``'sd1'``                : float (optional) — SD1 (ms)
        - ``'sd2'``                : float (optional) — SD2 (ms)
        - ``'sd_ratio'``           : float (optional) — SD1/SD2 ratio
    """
    rr_intervals = compute_rr_intervals(r_peaks, fs)

    results = {
        "rr_intervals": rr_intervals,
        "heart_rate": compute_heart_rate(rr_intervals),
        "mean_hr": compute_mean_hr(rr_intervals),
        "sdnn": compute_sdnn(rr_intervals),
        "rmssd": compute_rmssd(rr_intervals),
        "pnn50": compute_pnn50(rr_intervals),
        "hrv_frequency": compute_hrv_frequency(rr_intervals),
    }

    if include_poincare:
        poincare = compute_poincare_metrics(rr_intervals * 1000.0)
        results["poincare"] = poincare
        results["sd1"] = poincare["sd1"]
        results["sd2"] = poincare["sd2"]
        results["sd_ratio"] = poincare["sd_ratio"]

    return results
