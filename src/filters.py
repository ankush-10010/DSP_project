"""
Filter design and application for ECG signal processing.

Provides IIR (Butterworth, notch) and FIR filter design functions,
zero-phase filter application, frequency response computation, and
a complete ECG filtering pipeline (high-pass → notch → low-pass).

Designed for use with MIT-BIH Arrhythmia Database signals (360 Hz).
"""

import numpy as np
from scipy.signal import (
    butter,
    iirnotch,
    firwin,
    filtfilt,
    freqz,
)


# ---------------------------------------------------------------------------
# IIR filter design
# ---------------------------------------------------------------------------

def design_highpass_iir(cutoff: float, fs: float, order: int = 4) -> tuple:
    """Design a Butterworth high-pass IIR filter.

    Parameters
    ----------
    cutoff : float
        Cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    b, a : np.ndarray
        Numerator and denominator coefficients.
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='high')
    return b, a


def design_lowpass_iir(cutoff: float, fs: float, order: int = 4) -> tuple:
    """Design a Butterworth low-pass IIR filter.

    Parameters
    ----------
    cutoff : float
        Cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    b, a : np.ndarray
        Numerator and denominator coefficients.
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='low')
    return b, a


def design_notch_iir(freq: float, fs: float, quality: float = 30.0) -> tuple:
    """Design an IIR notch (band-reject) filter.

    Parameters
    ----------
    freq : float
        Frequency to reject in Hz (e.g. 50 or 60 Hz mains).
    fs : float
        Sampling frequency in Hz.
    quality : float, optional
        Quality factor — ratio of centre frequency to 3-dB bandwidth
        (default 30.0).

    Returns
    -------
    b, a : np.ndarray
        Numerator and denominator coefficients.
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return b, a


def design_bandpass_iir(
    low: float, high: float, fs: float, order: int = 4
) -> tuple:
    """Design a Butterworth bandpass IIR filter.

    Parameters
    ----------
    low : float
        Lower cut-off frequency in Hz.
    high : float
        Upper cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    b, a : np.ndarray
        Numerator and denominator coefficients.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return b, a


# ---------------------------------------------------------------------------
# FIR filter design
# ---------------------------------------------------------------------------

def design_highpass_fir(
    cutoff: float, fs: float, numtaps: int = 101
) -> np.ndarray:
    """Design an FIR high-pass filter using a windowed sinc method.

    Parameters
    ----------
    cutoff : float
        Cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    numtaps : int, optional
        Number of filter taps; must be odd for a Type-I high-pass
        (default 101).

    Returns
    -------
    coeffs : np.ndarray
        FIR filter coefficients.
    """
    coeffs = firwin(numtaps, cutoff, pass_zero=False, fs=fs)
    return coeffs


def design_lowpass_fir(
    cutoff: float, fs: float, numtaps: int = 101
) -> np.ndarray:
    """Design an FIR low-pass filter using a windowed sinc method.

    Parameters
    ----------
    cutoff : float
        Cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    numtaps : int, optional
        Number of filter taps (default 101).

    Returns
    -------
    coeffs : np.ndarray
        FIR filter coefficients.
    """
    coeffs = firwin(numtaps, cutoff, pass_zero=True, fs=fs)
    return coeffs


def design_bandpass_fir(
    low: float, high: float, fs: float, numtaps: int = 101
) -> np.ndarray:
    """Design an FIR bandpass filter using a windowed sinc method.

    Parameters
    ----------
    low : float
        Lower cut-off frequency in Hz.
    high : float
        Upper cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    numtaps : int, optional
        Number of filter taps; must be odd for a Type-I bandpass
        (default 101).

    Returns
    -------
    coeffs : np.ndarray
        FIR filter coefficients.
    """
    coeffs = firwin(numtaps, [low, high], pass_zero=False, fs=fs)
    return coeffs


# ---------------------------------------------------------------------------
# Filter application & analysis
# ---------------------------------------------------------------------------

def apply_filter(
    signal: np.ndarray, b, a=None
) -> np.ndarray:
    """Apply a digital filter with zero-phase distortion.

    Uses ``scipy.signal.filtfilt`` for forward-backward filtering so the
    output has zero phase shift relative to the input.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1-D).
    b : array_like
        Numerator coefficients of the filter.
    a : array_like or None, optional
        Denominator coefficients. If *None* (FIR filter), ``a`` is set
        to 1.

    Returns
    -------
    filtered : np.ndarray
        Zero-phase filtered signal, same length as *signal*.
    """
    if a is None:
        a = 1
    filtered = filtfilt(b, a, signal)
    return filtered


def get_frequency_response(
    b,
    a=None,
    fs: float = 360,
    n_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the frequency response of a digital filter.

    Parameters
    ----------
    b : array_like
        Numerator coefficients.
    a : array_like or None, optional
        Denominator coefficients (default 1 for FIR).
    fs : float, optional
        Sampling frequency in Hz (default 360).
    n_points : int, optional
        Number of frequency points to evaluate (default 4096).

    Returns
    -------
    frequencies_hz : np.ndarray
        Frequency axis in Hz.
    magnitude_db : np.ndarray
        Magnitude response in dB.
    phase_degrees : np.ndarray
        Phase response in degrees.
    """
    if a is None:
        a = 1
    w, h = freqz(b, a, worN=n_points)
    frequencies_hz = w * fs / (2.0 * np.pi)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase_degrees = np.degrees(np.angle(h))
    return frequencies_hz, magnitude_db, phase_degrees


# ---------------------------------------------------------------------------
# ECG filtering pipeline
# ---------------------------------------------------------------------------

def ecg_filter_pipeline(
    signal: np.ndarray,
    fs: float,
    hp_cutoff: float = 0.5,
    notch_freq: float = 60.0,
    lp_cutoff: float = 40.0,
    use_fir: bool = False,
) -> np.ndarray:
    """Full ECG filtering chain: high-pass → notch → low-pass.

    The pipeline removes baseline wander (high-pass), suppresses
    power-line interference (notch), and attenuates high-frequency
    noise (low-pass).

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1-D).
    fs : float
        Sampling frequency in Hz.
    hp_cutoff : float, optional
        High-pass cut-off in Hz (default 0.5).
    notch_freq : float, optional
        Notch centre frequency in Hz (default 60.0 for US mains;
        use 50.0 for European mains).
    lp_cutoff : float, optional
        Low-pass cut-off in Hz (default 40.0).
    use_fir : bool, optional
        If *True*, use FIR filters for the high-pass and low-pass
        stages instead of IIR Butterworth filters (default *False*).

    Returns
    -------
    cleaned : np.ndarray
        Filtered ECG signal, same length as *signal*.
    """
    # --- 1. High-pass: remove baseline wander ---
    if use_fir:
        b_hp = design_highpass_fir(hp_cutoff, fs)
        cleaned = apply_filter(signal, b_hp)
    else:
        b_hp, a_hp = design_highpass_iir(hp_cutoff, fs)
        cleaned = apply_filter(signal, b_hp, a_hp)

    # --- 2. Notch: suppress power-line interference ---
    b_notch, a_notch = design_notch_iir(notch_freq, fs)
    cleaned = apply_filter(cleaned, b_notch, a_notch)

    # --- 3. Low-pass: attenuate high-frequency noise ---
    if use_fir:
        b_lp = design_lowpass_fir(lp_cutoff, fs)
        cleaned = apply_filter(cleaned, b_lp)
    else:
        b_lp, a_lp = design_lowpass_iir(lp_cutoff, fs)
        cleaned = apply_filter(cleaned, b_lp, a_lp)

    return cleaned
