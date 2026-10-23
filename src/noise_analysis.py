"""
Noise characterization of ECG signals using frequency-domain analysis.

This module provides tools for analyzing noise components in ECG signals
from the MIT-BIH Arrhythmia Database (360 Hz sampling rate, 2-channel ECG).
It uses FFT and Welch's PSD estimation to identify common noise sources
such as baseline wander, powerline interference, and high-frequency noise.
"""

import numpy as np
from scipy.signal import welch


def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided FFT magnitude spectrum of a real-valued signal.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal (1-D array).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : np.ndarray
        Positive frequency bins in Hz.
    magnitude_spectrum : np.ndarray
        Magnitude spectrum in dB (20 * log10 of the normalized amplitude).
        A small epsilon is added before the log to avoid -inf for zero bins.
    """
    n = len(signal)
    fft_coeffs = np.fft.rfft(signal)
    # Normalize by number of samples
    magnitude = np.abs(fft_coeffs) * (2.0 / n)
    # DC component should not be doubled
    magnitude[0] /= 2.0
    # If n is even, the Nyquist bin should not be doubled either
    if n % 2 == 0:
        magnitude[-1] /= 2.0

    # Convert to dB with a floor to avoid log10(0)
    eps = np.finfo(float).tiny
    magnitude_db = 20.0 * np.log10(magnitude + eps)

    frequencies = np.fft.rfftfreq(n, d=1.0 / fs)
    return frequencies, magnitude_db


def compute_psd(
    signal: np.ndarray, fs: float, nperseg: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal (1-D array).
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method (default: 4096).
        If the signal is shorter than ``nperseg``, it is clamped to the
        signal length.

    Returns
    -------
    frequencies : np.ndarray
        Frequency bins in Hz.
    psd : np.ndarray
        Power spectral density estimate (V²/Hz).
    """
    # Clamp nperseg to signal length to avoid scipy errors
    nperseg = min(nperseg, len(signal))
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)
    return frequencies, psd


def identify_noise_components(signal: np.ndarray, fs: float) -> dict:
    """Analyze an ECG signal and identify common noise components.

    The function examines the power spectral density to flag:

    * **Baseline wander** — significant energy below 0.5 Hz.
    * **50 Hz powerline interference** — spectral peak near 50 Hz.
    * **60 Hz powerline interference** — spectral peak near 60 Hz.
    * **High-frequency noise** — significant energy above 100 Hz.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal (1-D array).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'baseline_wander'`` (bool): True if significant energy < 0.5 Hz.
        - ``'powerline_50hz'`` (bool): True if a peak is detected near 50 Hz.
        - ``'powerline_60hz'`` (bool): True if a peak is detected near 60 Hz.
        - ``'high_freq_noise'`` (bool): True if significant energy > 100 Hz.
        - ``'noise_bands'`` (list[tuple]): List of
          ``(low_freq, high_freq, description)`` tuples for each detected
          noise band.
    """
    frequencies, psd = compute_psd(signal, fs)
    total_power = np.sum(psd)

    # Helper: fraction of total power in a frequency band
    def band_power(f_low: float, f_high: float) -> float:
        mask = (frequencies >= f_low) & (frequencies <= f_high)
        return np.sum(psd[mask]) / total_power if total_power > 0 else 0.0

    # Helper: detect a spectral peak near a target frequency
    def has_peak_near(target_freq: float, tolerance: float = 2.0,
                      prominence_factor: float = 5.0) -> bool:
        """Return True if the PSD has a prominent peak within ±tolerance of target_freq.

        A peak must satisfy two conditions:
        1. It must be at least ``prominence_factor`` times the median PSD in
           its ±20 Hz neighbourhood (relative prominence).
        2. The power in the peak band must be at least 1% of total signal
           power (absolute significance).
        """
        mask_peak = (frequencies >= target_freq - tolerance) & (
            frequencies <= target_freq + tolerance
        )
        if not np.any(mask_peak):
            return False
        peak_power = np.max(psd[mask_peak])

        # Absolute significance check: peak band must carry meaningful power
        peak_band_power = np.sum(psd[mask_peak])
        if total_power > 0 and (peak_band_power / total_power) < 0.01:
            return False

        # Compare against the median PSD in a wider neighbourhood
        neighbourhood = 20.0  # Hz
        mask_neighbourhood = (frequencies >= target_freq - neighbourhood) & (
            frequencies <= target_freq + neighbourhood
        )
        median_power = np.median(psd[mask_neighbourhood])
        return bool(peak_power > prominence_factor * median_power)

    # --- Baseline wander: energy below 0.5 Hz ---
    baseline_wander_ratio = band_power(0.0, 0.5)
    baseline_wander = baseline_wander_ratio > 0.05  # > 5% of total power

    # --- Powerline interference ---
    powerline_50hz = has_peak_near(50.0) if fs > 100.0 else False
    powerline_60hz = has_peak_near(60.0) if fs > 120.0 else False

    # --- High-frequency noise: energy above 100 Hz ---
    high_freq_ratio = band_power(100.0, fs / 2.0)
    high_freq_noise = high_freq_ratio > 0.10  # > 10% of total power

    # --- Build noise_bands list ---
    noise_bands: list[tuple[float, float, str]] = []
    if baseline_wander:
        noise_bands.append((0.0, 0.5, "Baseline wander"))
    if powerline_50hz:
        noise_bands.append((48.0, 52.0, "50 Hz powerline interference"))
    if powerline_60hz:
        noise_bands.append((58.0, 62.0, "60 Hz powerline interference"))
    if high_freq_noise:
        noise_bands.append((100.0, fs / 2.0, "High-frequency noise"))

    return {
        "baseline_wander": baseline_wander,
        "powerline_50hz": powerline_50hz,
        "powerline_60hz": powerline_60hz,
        "high_freq_noise": high_freq_noise,
        "noise_bands": noise_bands,
    }


def compute_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """Compute the signal-to-noise ratio (SNR) in dB.

    SNR is calculated as:

    .. math::

        \\text{SNR} = 10 \\cdot \\log_{10}
        \\left( \\frac{P_{\\text{signal}}}{P_{\\text{noise}}} \\right)

    where *P_signal* is the power of the clean signal and *P_noise* is the
    power of the difference between noisy and clean signals.

    Parameters
    ----------
    clean_signal : np.ndarray
        Reference (clean) signal.
    noisy_signal : np.ndarray
        Noisy version of the signal (same length as ``clean_signal``).

    Returns
    -------
    float
        SNR value in dB.

    Raises
    ------
    ValueError
        If the two signals have different lengths.
    """
    if len(clean_signal) != len(noisy_signal):
        raise ValueError(
            f"Signal length mismatch: clean={len(clean_signal)}, "
            f"noisy={len(noisy_signal)}"
        )

    noise = noisy_signal - clean_signal
    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise == 0:
        return float("inf")

    return float(10.0 * np.log10(power_signal / power_noise))
