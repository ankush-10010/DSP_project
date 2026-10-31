"""
Pan-Tompkins QRS Detection Algorithm
=====================================

Implementation of the Pan-Tompkins real-time QRS detection algorithm
as described in:

    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.

The algorithm consists of a linear filtering stage (bandpass → derivative →
squaring → moving-window integration) followed by adaptive thresholding
with search-back to reliably detect QRS complexes in ECG signals.

Designed for the MIT-BIH Arrhythmia Database (360 Hz, 2-channel ECG).
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, find_peaks


# ---------------------------------------------------------------------------
# Stage 1: Bandpass Filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 5.0,
    high: float = 15.0,
    causal: bool = False,
) -> np.ndarray:
    """Apply a bandpass filter to suppress noise and baseline wander.

    The Pan-Tompkins paper specifies a passband of 5–15 Hz to maximise
    the QRS energy while attenuating P/T waves and high-frequency noise.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1-D ECG signal.
    fs : float
        Sampling frequency in Hz.
    low : float, optional
        Lower cutoff frequency in Hz (default 5.0).
    high : float, optional
        Upper cutoff frequency in Hz (default 15.0).
    causal : bool, optional
        If True, use standard causal IIR filter (lfilter); if False,
        use forward-backward zero-phase filtering (filtfilt, default).

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal (same length as input).
    """
    nyq = 0.5 * fs
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal[:, 0]
    b, a = butter(N=2, Wn=[low / nyq, high / nyq], btype="band")
    if causal:
        return lfilter(b, a, signal).astype(np.float64)
    return filtfilt(b, a, signal).astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 2: Five-Point Derivative
# ---------------------------------------------------------------------------

def derivative_filter(signal: np.ndarray) -> np.ndarray:
    """Five-point derivative filter as specified by Pan & Tompkins.

    The transfer function approximates a derivative while providing
    some low-pass smoothing::

        y[n] = (1/8) * (-x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2])

    Parameters
    ----------
    signal : np.ndarray
        1-D signal (typically the bandpass-filtered ECG).

    Returns
    -------
    np.ndarray
        Derivative-filtered signal (same length as input).
        The first and last two samples are zero-padded to maintain length.
    """
    N = len(signal)
    y = np.zeros(N, dtype=np.float64)
    if N >= 5:
        y[2:N - 2] = (1.0 / 8.0) * (
            -signal[0:N - 4]
            - 2.0 * signal[1:N - 3]
            + 2.0 * signal[3:N - 1]
            + signal[4:N]
        )
    return y


# ---------------------------------------------------------------------------
# Stage 3: Squaring
# ---------------------------------------------------------------------------

def squaring(signal: np.ndarray) -> np.ndarray:
    """Element-wise squaring to emphasise large slopes (QRS complexes).

    Squaring makes all values positive and non-linearly amplifies the
    output of the derivative filter, accentuating higher-frequency
    components arising from QRS complexes.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal (typically the derivative output).

    Returns
    -------
    np.ndarray
        Squared signal, ``y = x ** 2``.
    """
    return signal.astype(np.float64) ** 2


# ---------------------------------------------------------------------------
# Stage 4: Moving-Window Integration
# ---------------------------------------------------------------------------

def moving_window_integration(
    signal: np.ndarray,
    fs: float,
    window_ms: float = 150.0,
) -> np.ndarray:
    """Moving-window (boxcar) integrator.

    Produces a waveform that captures the slope *and* width of the QRS
    complex.  The window width recommended by Pan & Tompkins is ~150 ms.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal (typically the squared output).
    fs : float
        Sampling frequency in Hz.
    window_ms : float, optional
        Integration window width in milliseconds (default 150.0).

    Returns
    -------
    np.ndarray
        Integrated signal (same length as input).
    """
    window_size = max(1, int(window_ms * fs / 1000.0))
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    # 'same' keeps the output length equal to the input length
    integrated = np.convolve(signal, kernel, mode="same")
    return integrated


# ---------------------------------------------------------------------------
# Main Detection Function
# ---------------------------------------------------------------------------

def pan_tompkins_detect(
    signal: np.ndarray,
    fs: float,
    window_ms: float = 150.0,
    filter_mode: str = "bandpass",
    causal: bool = False,
) -> dict:
    """Run the full Pan-Tompkins QRS detection pipeline.

    Processing chain
    ~~~~~~~~~~~~~~~~
    1. Bandpass filter (5–15 Hz)
    2. Five-point derivative
    3. Squaring
    4. Moving-window integration (~150 ms)
    5. Adaptive dual-threshold with search-back

    Detected R-peak indices refer to positions in the **original input
    signal**, not any of the intermediate filtered versions.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1-D ECG signal.
    fs : float
        Sampling frequency in Hz.
    window_ms : float, optional
        Moving window integration width in ms (default 150.0).
    filter_mode : str, optional
        Filtering strategy: 'bandpass' (standard 5-15 Hz, default),
        'raw' or 'none' (no pre-filtering), or 'full' (HP 0.5Hz +
        Notch 50Hz + LP 150Hz + Pan-Tompkins bandpass).
    causal : bool, optional
        If True, use standard causal IIR filtering (lfilter) instead
        of zero-phase forward-backward filtering (filtfilt, default).

    Returns
    -------
    dict
        Keys:

        - ``'r_peaks'`` : np.ndarray of int – R-peak sample indices
          in the original signal.
        - ``'filtered'`` : np.ndarray – bandpass-filtered signal.
        - ``'derivative'`` : np.ndarray – derivative output.
        - ``'squared'`` : np.ndarray – squared output.
        - ``'integrated'`` : np.ndarray – moving-window integration output.
        - ``'threshold_info'`` : dict with ``'signal_level'`` and
          ``'noise_level'`` histories (lists of floats).
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)

    # ---- linear filtering stages ----------------------------------------
    if filter_mode == "bandpass":
        filtered = bandpass_filter(signal, fs, causal=causal)
    elif filter_mode in ("none", "raw"):
        filtered = signal.copy()
    elif filter_mode == "full":
        from src.filters import ecg_filter_pipeline
        pre_filtered = ecg_filter_pipeline(signal, fs, hp_cutoff=0.5, notch_freq=50.0, lp_cutoff=150.0)
        filtered = bandpass_filter(pre_filtered, fs, causal=causal)
    else:
        raise ValueError(f"Unknown filter_mode '{filter_mode}'. Must be 'bandpass', 'raw', or 'full'.")

    derivative = derivative_filter(filtered)
    squared = squaring(derivative)
    integrated = moving_window_integration(squared, fs, window_ms=window_ms)

    # ---- find all local peaks in the integrated signal -------------------
    # Minimum distance between candidate peaks: 200 ms (refractory period)
    refractory_samples = int(0.200 * fs)

    candidate_indices, _ = find_peaks(integrated, distance=refractory_samples)

    if len(candidate_indices) == 0:
        return {
            "r_peaks": np.array([], dtype=int),
            "filtered": filtered,
            "derivative": derivative,
            "squared": squared,
            "integrated": integrated,
            "threshold_info": {"signal_level": [], "noise_level": []},
        }

    candidate_values = integrated[candidate_indices]

    # ---- initialise adaptive thresholds from the first 2 seconds ---------
    init_samples = int(2.0 * fs)
    init_mask = candidate_indices < init_samples

    if np.any(init_mask):
        spki = np.max(candidate_values[init_mask])  # signal peak estimate
        npki = np.mean(candidate_values[init_mask])  # noise peak estimate
    else:
        # Fallback: use the first few candidate peaks
        spki = np.max(candidate_values[:5]) if len(candidate_values) >= 5 else np.max(candidate_values)
        npki = np.mean(candidate_values[:5]) if len(candidate_values) >= 5 else np.mean(candidate_values)

    thr_sig = npki + 0.25 * (spki - npki)
    thr_noise = 0.5 * thr_sig

    # ---- adaptive thresholding pass --------------------------------------
    signal_peaks: list[int] = []          # indices into *integrated* signal
    rr_intervals: list[int] = []          # in samples
    signal_level_history: list[float] = []
    noise_level_history: list[float] = []

    for idx, val in zip(candidate_indices, candidate_values):
        # --- search-back logic -------------------------------------------
        if len(signal_peaks) >= 2:
            avg_rr = int(np.mean(rr_intervals[-8:]))  # recent RR average
            rr_limit = int(1.66 * avg_rr)

            if (idx - signal_peaks[-1]) > rr_limit:
                # Search back between last detected peak and current
                sb_start = signal_peaks[-1] + refractory_samples
                sb_end = idx

                sb_mask = (candidate_indices > sb_start) & (candidate_indices < sb_end)
                sb_indices = candidate_indices[sb_mask]
                sb_values = candidate_values[sb_mask]

                if len(sb_values) > 0:
                    thr_sb = 0.5 * thr_sig
                    sb_above = sb_values > thr_sb
                    if np.any(sb_above):
                        best = np.argmax(sb_values[sb_above])
                        sb_peak_idx = sb_indices[sb_above][best]
                        sb_peak_val = sb_values[sb_above][best]

                        signal_peaks.append(int(sb_peak_idx))
                        if len(signal_peaks) >= 2:
                            rr_intervals.append(
                                signal_peaks[-1] - signal_peaks[-2]
                            )

                        # Update SPKI with reduced gain for search-back peaks
                        spki = 0.25 * sb_peak_val + 0.75 * spki

        # --- normal classification ----------------------------------------
        if val > thr_sig:
            # Signal peak
            signal_peaks.append(int(idx))
            if len(signal_peaks) >= 2:
                rr_intervals.append(signal_peaks[-1] - signal_peaks[-2])
            spki = 0.125 * val + 0.875 * spki
        else:
            # Noise peak
            npki = 0.125 * val + 0.875 * npki

        # Recompute thresholds
        thr_sig = npki + 0.25 * (spki - npki)
        thr_noise = 0.5 * thr_sig

        signal_level_history.append(float(spki))
        noise_level_history.append(float(npki))

    # ---- enforce refractory period on final peak list --------------------
    if len(signal_peaks) > 1:
        cleaned: list[int] = [signal_peaks[0]]
        for pk in signal_peaks[1:]:
            if (pk - cleaned[-1]) >= refractory_samples:
                cleaned.append(pk)
        signal_peaks = cleaned

    # ---- map peaks from integrated signal to original signal R-peaks -----
    # The integrated signal is time-aligned (same length) with the original
    # thanks to 'same'-mode convolution and zero-padded derivative.  We
    # search a local window in the *bandpass-filtered* signal (which best
    # shows R-peak morphology) then report those indices — which correspond
    # directly to the original signal because filtfilt preserves length.

    half_qrs = int(0.075 * fs)  # ~75 ms half-window for R-peak search
    r_peaks: list[int] = []

    for pk in signal_peaks:
        lo = max(0, pk - half_qrs)
        hi = min(N, pk + half_qrs + 1)
        # Use the bandpass-filtered signal; R-peak is the local max
        local_max = int(lo + np.argmax(np.abs(filtered[lo:hi])))
        r_peaks.append(local_max)

    r_peaks_arr = np.unique(np.array(r_peaks, dtype=int))

    return {
        "r_peaks": r_peaks_arr,
        "filtered": filtered,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "threshold_info": {
            "signal_level": signal_level_history,
            "noise_level": noise_level_history,
        },
    }
