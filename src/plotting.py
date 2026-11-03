"""
plotting.py — Visualization functions for the ECG signal processing pipeline.

Provides plotting utilities for raw ECG signals, noise analysis, filter analysis,
Pan-Tompkins processing stages, feature extraction, and detection validation.
All functions accept an optional ``save_path`` parameter to persist figures to disk.

Uses the non-interactive Agg backend so plots can be generated in headless
environments (servers, CI pipelines, etc.).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

# ---------------------------------------------------------------------------
# Style setup – prefer seaborn-v0_8-whitegrid, fall back to ggplot
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

# Default figure size used across the module
_DEFAULT_FIGSIZE = (14, 6)


# ===================================================================== #
#                        Helper utilities                                #
# ===================================================================== #

def _save_and_close(fig, save_path):
    """Save *fig* to *save_path* (if given) then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalise.
    save_path : str or None
        Filesystem path for the saved image.  When ``None`` the figure is
        simply closed without saving.
    """
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _time_slice(signal, fs, duration, start):
    """Return (time_array, signal_slice, start_idx, end_idx) for a segment.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array.
    fs : float
        Sampling frequency in Hz.
    duration : float
        Length of the segment in seconds.
    start : float
        Start time of the segment in seconds.

    Returns
    -------
    t : np.ndarray
        Time vector (seconds) for the segment.
    seg : np.ndarray
        Signal samples within the segment.
    start_idx : int
        First sample index (inclusive).
    end_idx : int
        Last sample index (exclusive).
    """
    start_idx = int(start * fs)
    end_idx = min(int((start + duration) * fs), len(signal))
    seg = signal[start_idx:end_idx]
    t = np.arange(start_idx, end_idx) / fs
    return t, seg, start_idx, end_idx


# ===================================================================== #
#                      Data Visualization                                #
# ===================================================================== #

def plot_raw_ecg(signal, fs, title="Raw ECG Signal", duration=10.0,
                 start=0.0, save_path=None):
    """Plot a segment of a raw (single-channel) ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : float
        Sampling frequency in Hz.
    title : str, optional
        Figure title.
    duration : float, optional
        Duration of the displayed segment in seconds (default 10 s).
    start : float, optional
        Start time of the segment in seconds (default 0 s).
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    t, seg, _, _ = _time_slice(signal, fs, duration, start)

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(t, seg, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    _save_and_close(fig, save_path)


def plot_ecg_with_annotations(signal, fs, r_peaks,
                              title="ECG with R-peak Annotations",
                              duration=10.0, start=0.0, save_path=None):
    """Plot an ECG segment with R-peak markers overlaid.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : float
        Sampling frequency in Hz.
    r_peaks : array-like of int
        Sample indices of detected R-peaks.
    title : str, optional
        Figure title.
    duration : float, optional
        Duration of the displayed segment in seconds.
    start : float, optional
        Start time in seconds.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    t, seg, start_idx, end_idx = _time_slice(signal, fs, duration, start)
    r_peaks = np.asarray(r_peaks)

    # Filter R-peaks that fall within the window
    mask = (r_peaks >= start_idx) & (r_peaks < end_idx)
    rp_in = r_peaks[mask]

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(t, seg, linewidth=0.8, color="steelblue", label="ECG")
    ax.plot(rp_in / fs, signal[rp_in], "ro", markersize=6, label="R-peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save_and_close(fig, save_path)


def plot_ecg_channels(signals_2d, fs, sig_names, duration=10.0, start=0.0,
                      save_path=None):
    """Plot both ECG channels as vertically stacked subplots.

    Parameters
    ----------
    signals_2d : np.ndarray
        2-D array of shape ``(N, 2)`` containing two ECG channels.
    fs : float
        Sampling frequency in Hz.
    sig_names : list of str
        Names for each channel (length 2).
    duration : float, optional
        Duration of the displayed segment in seconds.
    start : float, optional
        Start time in seconds.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    n_channels = signals_2d.shape[1] if signals_2d.ndim == 2 else 1
    fig, axes = plt.subplots(n_channels, 1, figsize=(_DEFAULT_FIGSIZE[0], 4 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ch = signals_2d[:, i] if signals_2d.ndim == 2 else signals_2d
        t, seg, _, _ = _time_slice(ch, fs, duration, start)
        ax.plot(t, seg, linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(sig_names[i] if i < len(sig_names) else f"Channel {i}")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("ECG Channels", fontsize=14, y=1.01)
    _save_and_close(fig, save_path)


# ===================================================================== #
#                        Noise Analysis                                  #
# ===================================================================== #

def plot_fft(freqs, magnitude, title="FFT Magnitude Spectrum", xlim=200,
             save_path=None):
    """Plot the single-sided FFT magnitude spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in Hz.
    magnitude : np.ndarray
        Corresponding magnitude values.
    title : str, optional
        Figure title.
    xlim : float, optional
        Upper frequency limit for the x-axis in Hz (default 200).
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(freqs, magnitude, linewidth=0.8, color="darkorange")
    ax.set_xlim(0, xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    _save_and_close(fig, save_path)


def plot_psd(freqs, psd, title="Power Spectral Density", xlim=200,
             save_path=None):
    """Plot the Power Spectral Density (PSD).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in Hz.
    psd : np.ndarray
        PSD values (power / Hz).
    title : str, optional
        Figure title.
    xlim : float, optional
        Upper frequency limit for the x-axis in Hz (default 200).
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.semilogy(freqs, psd, linewidth=0.8, color="teal")
    ax.set_xlim(0, xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power / Frequency (dB/Hz)")
    ax.set_title(title)
    _save_and_close(fig, save_path)


def plot_noise_comparison(signal_clean, signal_noisy, fs,
                          labels=("Clean (Record 100)", "Noisy (Record 105)"),
                          save_path=None):
    """Side-by-side PSD comparison of a clean and a noisy ECG signal.

    The PSD is estimated via Welch's method (``scipy.signal.welch``).

    Parameters
    ----------
    signal_clean : np.ndarray
        1-D clean ECG signal.
    signal_noisy : np.ndarray
        1-D noisy ECG signal.
    fs : float
        Sampling frequency in Hz.
    labels : tuple of str, optional
        Legend labels for the two signals.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    from scipy.signal import welch

    f_clean, psd_clean = welch(signal_clean, fs=fs, nperseg=1024)
    f_noisy, psd_noisy = welch(signal_noisy, fs=fs, nperseg=1024)

    fig, axes = plt.subplots(1, 2, figsize=(_DEFAULT_FIGSIZE[0], 5))

    axes[0].semilogy(f_clean, psd_clean, color="steelblue", linewidth=0.8)
    axes[0].set_title(f"PSD — {labels[0]}")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power / Frequency")
    axes[0].set_xlim(0, fs / 2)

    axes[1].semilogy(f_noisy, psd_noisy, color="crimson", linewidth=0.8)
    axes[1].set_title(f"PSD — {labels[1]}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power / Frequency")
    axes[1].set_xlim(0, fs / 2)

    _save_and_close(fig, save_path)


# ===================================================================== #
#                       Filter Analysis                                  #
# ===================================================================== #

def plot_frequency_response(freqs_hz, magnitude_db, phase_deg=None,
                            title="Filter Frequency Response",
                            save_path=None):
    """Plot filter magnitude response and, optionally, phase response.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency axis in Hz.
    magnitude_db : np.ndarray
        Magnitude response in dB.
    phase_deg : np.ndarray or None, optional
        Phase response in degrees.  When provided a second subplot is drawn.
    title : str, optional
        Figure title.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    n_plots = 2 if phase_deg is not None else 1
    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(_DEFAULT_FIGSIZE[0], 4 * n_plots),
                             sharex=True)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(freqs_hz, magnitude_db, linewidth=1.0, color="navy")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(title)

    if phase_deg is not None:
        axes[1].plot(freqs_hz, phase_deg, linewidth=1.0, color="darkgreen")
        axes[1].set_ylabel("Phase (degrees)")

    axes[-1].set_xlabel("Frequency (Hz)")
    _save_and_close(fig, save_path)


def plot_filter_comparison(signal_raw, signal_filtered, fs,
                           title="Before vs After Filtering",
                           duration=10.0, start=0.0, save_path=None):
    """Compare raw and filtered ECG signals (overlay + separate subplots).

    The figure has three rows:
    1. Raw signal
    2. Filtered signal
    3. Both signals overlaid for direct comparison

    Parameters
    ----------
    signal_raw : np.ndarray
        1-D raw ECG signal.
    signal_filtered : np.ndarray
        1-D filtered ECG signal.
    fs : float
        Sampling frequency in Hz.
    title : str, optional
        Figure title.
    duration : float, optional
        Duration of the displayed segment in seconds.
    start : float, optional
        Start time in seconds.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    t_raw, seg_raw, _, _ = _time_slice(signal_raw, fs, duration, start)
    t_filt, seg_filt, _, _ = _time_slice(signal_filtered, fs, duration, start)

    fig, axes = plt.subplots(3, 1, figsize=(_DEFAULT_FIGSIZE[0], 12), sharex=True)

    axes[0].plot(t_raw, seg_raw, linewidth=0.8, color="grey")
    axes[0].set_title("Raw Signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_filt, seg_filt, linewidth=0.8, color="steelblue")
    axes[1].set_title("Filtered Signal")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t_raw, seg_raw, linewidth=0.6, alpha=0.5, color="grey", label="Raw")
    axes[2].plot(t_filt, seg_filt, linewidth=0.8, color="steelblue", label="Filtered")
    axes[2].set_title("Overlay Comparison")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14, y=1.01)
    _save_and_close(fig, save_path)


# ===================================================================== #
#                    Pan-Tompkins Stages                                 #
# ===================================================================== #

def plot_pan_tompkins_stages(stages_dict, fs, duration=5.0, start=0.0,
                             save_path=None):
    """Plot all Pan-Tompkins processing stages as vertically stacked subplots.

    Parameters
    ----------
    stages_dict : dict
        Dictionary with the following keys (all values are 1-D np.ndarray
        except ``'r_peaks'`` which is an array of sample indices):

        - ``'original'``   — original ECG signal
        - ``'filtered'``   — bandpass-filtered signal
        - ``'derivative'`` — differentiated signal
        - ``'squared'``    — squared signal
        - ``'integrated'`` — moving-window integrated signal
        - ``'r_peaks'``    — detected R-peak sample indices

    fs : float
        Sampling frequency in Hz.
    duration : float, optional
        Duration of the displayed segment in seconds (default 5 s).
    start : float, optional
        Start time in seconds.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    stage_keys = ["original", "filtered", "derivative", "squared", "integrated"]
    n_stages = len(stage_keys)

    fig, axes = plt.subplots(n_stages, 1,
                             figsize=(_DEFAULT_FIGSIZE[0], 3 * n_stages),
                             sharex=True)

    r_peaks = np.asarray(stages_dict.get("r_peaks", []))
    start_idx = int(start * fs)
    end_idx_limit = int((start + duration) * fs)

    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for i, key in enumerate(stage_keys):
        sig = stages_dict[key]
        t, seg, s_idx, e_idx = _time_slice(sig, fs, duration, start)
        axes[i].plot(t, seg, linewidth=0.8, color=colors[i % len(colors)])
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Pan-Tompkins — {key.capitalize()}")

        # Overlay R-peaks on the original signal subplot
        if key == "original" and len(r_peaks) > 0:
            mask = (r_peaks >= s_idx) & (r_peaks < e_idx)
            rp_in = r_peaks[mask]
            if len(rp_in) > 0:
                axes[i].plot(rp_in / fs, sig[rp_in], "ro", markersize=6,
                             label="R-peaks")
                axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Pan-Tompkins Processing Stages", fontsize=14, y=1.01)
    _save_and_close(fig, save_path)


# ===================================================================== #
#                     Feature Extraction                                 #
# ===================================================================== #

def plot_heart_rate(hr, r_peaks, fs, title="Instantaneous Heart Rate",
                    save_path=None):
    """Plot instantaneous heart rate over time.

    Parameters
    ----------
    hr : np.ndarray
        Instantaneous heart rate values in BPM (length = len(r_peaks) - 1).
    r_peaks : array-like of int
        Sample indices of R-peaks.
    fs : float
        Sampling frequency in Hz.
    title : str, optional
        Figure title.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    r_peaks = np.asarray(r_peaks)
    # Time axis at midpoints between consecutive R-peaks
    t = r_peaks[1:] / fs

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(t, hr, linewidth=1.0, color="crimson", marker="o", markersize=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(title)
    _save_and_close(fig, save_path)


def plot_tachogram(rr_intervals, title="RR Interval Tachogram",
                   save_path=None):
    """Plot RR intervals vs. beat number (tachogram).

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR interval durations (seconds or milliseconds).
    title : str, optional
        Figure title.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    beat_numbers = np.arange(1, len(rr_intervals) + 1)
    ax.plot(beat_numbers, rr_intervals, linewidth=0.8, color="mediumseagreen",
            marker="o", markersize=3)
    ax.set_xlabel("Beat Number")
    ax.set_ylabel("RR Interval")
    ax.set_title(title)
    _save_and_close(fig, save_path)


def plot_hrv_psd(freqs, psd, title="HRV Power Spectral Density",
                 save_path=None):
    """Plot PSD of the RR interval series with LF and HF bands shaded.

    Standard HRV frequency bands:

    * **LF** (Low Frequency):  0.04 – 0.15 Hz
    * **HF** (High Frequency): 0.15 – 0.40 Hz

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in Hz.
    psd : np.ndarray
        PSD values.
    title : str, optional
        Figure title.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(freqs, psd, linewidth=1.0, color="black")

    # Shade LF band (0.04 – 0.15 Hz)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    ax.fill_between(freqs[lf_mask], psd[lf_mask], alpha=0.35,
                     color="dodgerblue", label="LF (0.04–0.15 Hz)")

    # Shade HF band (0.15 – 0.40 Hz)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    ax.fill_between(freqs[hf_mask], psd[hf_mask], alpha=0.35,
                     color="salmon", label="HF (0.15–0.40 Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save_and_close(fig, save_path)


# ===================================================================== #
#                         Validation                                     #
# ===================================================================== #

def plot_detection_comparison(signal, fs, detected, reference,
                              title="Detection vs Reference",
                              duration=10.0, start=0.0, save_path=None):
    """Plot ECG with both detected and reference R-peak markers.

    Detected peaks are shown as **red** dots; reference peaks as **green**
    triangles.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal array.
    fs : float
        Sampling frequency in Hz.
    detected : array-like of int
        Sample indices of detected R-peaks.
    reference : array-like of int
        Sample indices of reference (ground-truth) R-peaks.
    title : str, optional
        Figure title.
    duration : float, optional
        Duration of the displayed segment in seconds.
    start : float, optional
        Start time in seconds.
    save_path : str or None, optional
        If provided, the figure is saved to this path.
    """
    t, seg, start_idx, end_idx = _time_slice(signal, fs, duration, start)
    detected = np.asarray(detected)
    reference = np.asarray(reference)

    det_mask = (detected >= start_idx) & (detected < end_idx)
    ref_mask = (reference >= start_idx) & (reference < end_idx)
    det_in = detected[det_mask]
    ref_in = reference[ref_mask]

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(t, seg, linewidth=0.8, color="steelblue", label="ECG")

    ax.plot(det_in / fs, signal[det_in], "ro", markersize=7,
            label="Detected R-peaks", zorder=5)
    ax.plot(ref_in / fs, signal[ref_in], "g^", markersize=7,
            label="Reference R-peaks", zorder=4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save_and_close(fig, save_path)
