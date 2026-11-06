"""
Poincaré Plot and Non-Linear HRV Analysis Module
================================================

Implements Poincaré plot geometric analysis for heart rate variability (HRV),
including SD1, SD2, SD1/SD2 ratio, and confidence ellipse fitting.

Poincaré plot analysis quantifies the non-linear dynamics of autonomic cardiac
regulation by plotting consecutive RR intervals (RR_{n+1} vs RR_n):
    - SD1: Dispersion perpendicular to the line of identity (y = x).
      Reflects short-term, beat-to-beat variability (parasympathetic / vagal tone).
      SD1 = sqrt(0.5 * Var(Delta RR)) = RMSSD / sqrt(2).
    - SD2: Dispersion along the line of identity (y = x).
      Reflects continuous and long-term heart rate variability (sympathetic and parasympathetic).
      SD2 = sqrt(2 * Var(RR) - 0.5 * Var(Delta RR)) = sqrt(2 * SDNN^2 - SD1^2).
    - SD1/SD2 ratio: Balance between short-term and long-term variability.
    - Confidence Ellipse: Fitted bivariate confidence ellipse centered at (xc, yc),
      with semi-major axis = 2 * SD2 along the line of identity (45°),
      and semi-minor axis = 2 * SD1 perpendicular to it (-45° / 135°).

References:
    - Brennan, M., Palaniswami, M., & Kamen, P. (2001). "Do Existing Measures of
      Poincaré Plot Geometry Reflect Nonlinear Features of Heart Rate Variability?"
      IEEE Transactions on Biomedical Engineering, 48(11), 1342-1347.
    - Task Force of ESC/NASPE (1996). "Heart Rate Variability: Standards of
      Measurement, Physiological Interpretation, and Clinical Use."
      Circulation, 93(5), 1043-1065.
"""

from typing import Any, Dict, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling setup
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")


def compute_poincare_metrics(rr_intervals: np.ndarray) -> Dict[str, Any]:
    """Compute Poincaré plot descriptors and confidence ellipse parameters.

    Calculates SD1 (short-term variability), SD2 (long-term variability),
    the SD1/SD2 ratio, and confidence ellipse geometry from an array of
    consecutive RR intervals.

    Formulas:
        - Delta RR_n = RR_{n+1} - RR_n
        - SD1 = sqrt(0.5 * Var(Delta RR))
        - SD2 = sqrt(2 * Var(RR) - 0.5 * Var(Delta RR))
        - SD1 / SD2 ratio = SD1 / SD2
        - Center (xc, yc) = (mean(RR), mean(RR))
        - Semi-major axis = 2 * SD2
        - Semi-minor axis = 2 * SD1
        - Rotation angle = 45 degrees

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of consecutive RR interval durations (in seconds or milliseconds).
        Must contain at least 3 intervals.

    Returns
    -------
    dict
        Dictionary containing:
        - ``'sd1'``                 : float — Short-term variability (SD1).
        - ``'sd2'``                 : float — Long-term variability (SD2).
        - ``'sd_ratio'``            : float — SD1 / SD2 ratio.
        - ``'sd1_sd2_ratio'``        : float — Alias for sd_ratio.
        - ``'ellipse_center'``      : tuple (xc, yc) — Center of the ellipse.
        - ``'center'``              : tuple (xc, yc) — Alias for ellipse_center.
        - ``'xc'``                  : float — Center x-coordinate (mean RR).
        - ``'yc'``                  : float — Center y-coordinate (mean RR).
        - ``'ellipse_semi_major'``  : float — 2 * SD2 (semi-major axis length).
        - ``'ellipse_semi_minor'``  : float — 2 * SD1 (semi-minor axis length).
        - ``'semi_major'``          : float — Alias for ellipse_semi_major.
        - ``'semi_minor'``          : float — Alias for ellipse_semi_minor.
        - ``'ellipse_angle'``       : float — Rotation angle in degrees (45.0°).
        - ``'angle'``               : float — Alias for ellipse_angle.
        - ``'ellipse_area'``        : float — Standard ellipse area (pi * SD1 * SD2).
        - ``'confidence_area'``     : float — 95% confidence ellipse area (pi * 2*SD1 * 2*SD2).
        - ``'var_rr'``              : float — Sample variance of RR intervals.
        - ``'var_delta_rr'``        : float — Sample variance of successive RR differences.

    Raises
    ------
    ValueError
        If ``rr_intervals`` contains fewer than 3 values or any non-positive values.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64)
    if rr.ndim != 1:
        rr = rr.ravel()

    if len(rr) < 3:
        raise ValueError("At least 3 RR intervals are required to compute Poincaré metrics.")
    if np.any(rr <= 0):
        raise ValueError("All RR intervals must be positive.")

    # Successive differences: Delta RR_n = RR_{n+1} - RR_n
    delta_rr = np.diff(rr)

    # Sample variance with ddof=1 (unbiased estimator)
    var_rr = float(np.var(rr, ddof=1))
    var_delta_rr = float(np.var(delta_rr, ddof=1))

    # SD1: dispersion perpendicular to line of identity (short-term variability)
    sd1 = float(np.sqrt(0.5 * var_delta_rr))

    # SD2: dispersion along line of identity (long-term variability)
    term = 2.0 * var_rr - 0.5 * var_delta_rr
    sd2 = float(np.sqrt(max(0.0, term)))

    # SD1/SD2 ratio
    if sd2 > 0:
        sd_ratio = float(sd1 / sd2)
    else:
        sd_ratio = 0.0 if sd1 == 0.0 else float("inf")

    # Center is located on the line of identity at mean(RR)
    xc = float(np.mean(rr))
    yc = float(np.mean(rr))

    # Confidence ellipse parameters (bivariate 2-sigma / 95% confidence bounds)
    semi_major = float(2.0 * sd2)
    semi_minor = float(2.0 * sd1)
    rotation_angle = 45.0  # degrees

    ellipse_area = float(np.pi * sd1 * sd2)
    confidence_area = float(np.pi * semi_major * semi_minor)

    # Unit helpers
    if np.mean(rr) < 10.0:
        sd1_ms = sd1 * 1000.0
        sd2_ms = sd2 * 1000.0
        sd1_s = sd1
        sd2_s = sd2
    else:
        sd1_ms = sd1
        sd2_ms = sd2
        sd1_s = sd1 / 1000.0
        sd2_s = sd2 / 1000.0

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd1_ms": sd1_ms,
        "sd2_ms": sd2_ms,
        "sd1_s": sd1_s,
        "sd2_s": sd2_s,
        "sd_ratio": sd_ratio,
        "sd1_sd2_ratio": sd_ratio,
        "ellipse_center": (xc, yc),
        "center": (xc, yc),
        "xc": xc,
        "yc": yc,
        "ellipse_semi_major": semi_major,
        "ellipse_semi_minor": semi_minor,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "ellipse_angle": rotation_angle,
        "angle": rotation_angle,
        "ellipse_area": ellipse_area,
        "confidence_area": confidence_area,
        "var_rr": var_rr,
        "var_delta_rr": var_delta_rr,
    }


def _draw_poincare_axes(
    ax: plt.Axes,
    rr_intervals: np.ndarray,
    title: str = "Poincaré Plot",
    unit_label: str = "ms",
    point_color: str = "#1f77b4",
    ellipse_color: str = "#d62728",
) -> Dict[str, Any]:
    """Helper to draw a complete Poincaré plot on a provided Matplotlib Axes."""
    metrics = compute_poincare_metrics(rr_intervals)

    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    xc, yc = metrics["ellipse_center"]
    sd1 = metrics["sd1"]
    sd2 = metrics["sd2"]
    semi_major = metrics["ellipse_semi_major"]
    semi_minor = metrics["ellipse_semi_minor"]

    # 1. Scatter plot of consecutive RR pairs (RR_n, RR_{n+1})
    ax.scatter(
        x,
        y,
        c=point_color,
        alpha=0.45,
        s=20,
        edgecolors="none",
        label=f"RR pairs (N={len(x)})",
        zorder=3,
    )

    # Determine axis boundaries
    all_vals = np.concatenate([x, y])
    data_min = float(np.min(all_vals))
    data_max = float(np.max(all_vals))
    span = data_max - data_min
    margin = max(0.08 * span, semi_major * 0.15)
    lim_min = min(data_min, xc - semi_major) - margin
    lim_max = max(data_max, xc + semi_major) + margin

    # 2. Line of identity: y = x
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Identity line ($y = x$)",
        zorder=2,
    )

    # 3. Parametric fitted confidence ellipse
    # Unrotated parametric angle
    theta = np.linspace(0, 2 * np.pi, 250)
    alpha = np.radians(45.0)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    x_prime = semi_major * np.cos(theta)
    y_prime = semi_minor * np.sin(theta)

    # Rotated coordinates centered at (xc, yc)
    x_ell = xc + x_prime * cos_a - y_prime * sin_a
    y_ell = yc + x_prime * sin_a + y_prime * cos_a

    ax.plot(
        x_ell,
        y_ell,
        color=ellipse_color,
        linewidth=2.2,
        label=r"Confidence Ellipse ($2\cdot SD_1, 2\cdot SD_2$)",
        zorder=5,
    )
    ax.fill(
        x_ell,
        y_ell,
        color=ellipse_color,
        alpha=0.12,
        zorder=4,
    )

    # 4. SD2 Axis (Along the line of identity y = x, rotated 45°)
    # Major diameter line
    sd2_x0 = xc - semi_major * cos_a
    sd2_y0 = yc - semi_major * sin_a
    sd2_x1 = xc + semi_major * cos_a
    sd2_y1 = yc + semi_major * sin_a
    ax.plot(
        [sd2_x0, sd2_x1],
        [sd2_y0, sd2_y1],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        label=r"$SD_2$ Axis (Long-term)",
        zorder=6,
    )

    # Direction arrow from center for SD2
    ax.annotate(
        "",
        xy=(xc + sd2 * cos_a, yc + sd2 * sin_a),
        xytext=(xc, yc),
        arrowprops=dict(
            arrowstyle="->",
            color="#2ca02c",
            lw=2.2,
            mutation_scale=14,
        ),
        zorder=7,
    )

    # SD2 text badge
    ax.text(
        xc + (sd2 * 0.65) * cos_a - 0.03 * (lim_max - lim_min),
        yc + (sd2 * 0.65) * sin_a + 0.05 * (lim_max - lim_min),
        f"$SD_2$: {sd2:.1f} {unit_label}",
        color="#1b6e1b",
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eafaf1", edgecolor="#2ca02c", alpha=0.9),
        zorder=8,
    )

    # 5. SD1 Axis (Perpendicular to line of identity, rotated -45° / 135°)
    # Unit vector perpendicular to (cos_a, sin_a) is (-sin_a, cos_a)
    sd1_x0 = xc + semi_minor * sin_a
    sd1_y0 = yc - semi_minor * cos_a
    sd1_x1 = xc - semi_minor * sin_a
    sd1_y1 = yc + semi_minor * cos_a
    ax.plot(
        [sd1_x0, sd1_x1],
        [sd1_y0, sd1_y1],
        color="#e65100",
        linestyle="-.",
        linewidth=1.8,
        label=r"$SD_1$ Axis (Short-term)",
        zorder=6,
    )

    # Direction arrow from center for SD1
    ax.annotate(
        "",
        xy=(xc - sd1 * sin_a, yc + sd1 * cos_a),
        xytext=(xc, yc),
        arrowprops=dict(
            arrowstyle="->",
            color="#e65100",
            lw=2.2,
            mutation_scale=14,
        ),
        zorder=7,
    )

    # SD1 text badge
    ax.text(
        xc - (sd1 * 0.65) * sin_a - 0.08 * (lim_max - lim_min),
        yc + (sd1 * 0.65) * cos_a + 0.03 * (lim_max - lim_min),
        f"$SD_1$: {sd1:.1f} {unit_label}",
        color="#b23e00",
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3e0", edgecolor="#e65100", alpha=0.9),
        zorder=8,
    )

    # 6. Ellipse Center (xc, yc)
    ax.scatter(
        [xc],
        [yc],
        color="#222222",
        s=45,
        marker="o",
        edgecolors="white",
        linewidth=1.2,
        label=f"Center ({xc:.1f}, {yc:.1f})",
        zorder=9,
    )

    # 7. Statistics annotation box
    stats_str = (
        f"$SD_1$ (vagal / short-term) = {sd1:.2f} {unit_label}\n"
        f"$SD_2$ (long-term / total)   = {sd2:.2f} {unit_label}\n"
        f"$SD_1 / SD_2$ ratio        = {metrics['sd_ratio']:.3f}\n"
        f"Mean RR (Center)            = {xc:.1f} {unit_label}"
    )
    ax.text(
        0.03,
        0.97,
        stats_str,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#666666", alpha=0.92),
        zorder=10,
    )

    # Formatting axes and aspect
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(f"$RR_n$ ({unit_label})", fontsize=11)
    ax.set_ylabel(f"$RR_{{n+1}}$ ({unit_label})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    return metrics


def plot_poincare(
    rr_intervals: np.ndarray,
    title: str = "Poincaré Plot",
    save_path: Optional[str] = None,
    unit: str = "auto",
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """Plot Poincaré plot ($RR_{n+1}$ vs $RR_n$) with fitted confidence ellipse and $SD_1, SD_2$ axes.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of RR interval durations (in seconds or milliseconds).
    title : str, optional
        Title for the plot. Default is "Poincaré Plot".
    save_path : str or None, optional
        Filesystem path to save the generated figure. If None, figure is returned.
    unit : str, optional
        Unit for RR intervals: "auto" (default), "ms", or "s".
        If "auto" and mean RR < 10, intervals are converted from seconds to milliseconds.
    figsize : tuple of float, optional
        Figure dimensions in inches. Default is (8, 8).

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64)

    # Determine scaling and unit label
    if unit == "ms":
        if np.nanmean(rr) < 10.0:
            rr = rr * 1000.0
        unit_label = "ms"
    elif unit == "s":
        if np.nanmean(rr) >= 10.0:
            rr = rr / 1000.0
        unit_label = "s"
    else:  # auto
        if np.nanmean(rr) < 10.0:
            rr = rr * 1000.0
            unit_label = "ms"
        else:
            unit_label = "ms" if np.nanmean(rr) > 100.0 else "s"

    fig, ax = plt.subplots(figsize=figsize)
    _draw_poincare_axes(
        ax=ax,
        rr_intervals=rr,
        title=title,
        unit_label=unit_label,
        point_color="#1f77b4",
        ellipse_color="#d62728",
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_poincare_comparison(
    rr_clean: np.ndarray,
    rr_arrhythmic: np.ndarray,
    labels: Tuple[str, str] = ("Normal Sinus (Record 100)", "Arrhythmia (Record 203)"),
    save_path: Optional[str] = None,
    unit: str = "auto",
    figsize: Tuple[float, float] = (16, 7.5),
) -> plt.Figure:
    """Generate a side-by-side Poincaré plot comparison contrasting normal sinus rhythm vs arrhythmia.

    Highlights the physiological contrast between:
    - Normal Sinus Rhythm: Compact, torpedo/cigar-shaped cluster along the identity
      line indicating healthy autonomic regulation with bounded variability.
    - Arrhythmia: Wide dispersion, irregular scatter cloud, and severe outliers
      reflecting ectopic beats, premature contractions, and autonomic dysregulation.

    Parameters
    ----------
    rr_clean : np.ndarray
        RR intervals for normal sinus rhythm (e.g. MIT-BIH Record 100).
    rr_arrhythmic : np.ndarray
        RR intervals for arrhythmic recording (e.g. MIT-BIH Record 203).
    labels : tuple of str, optional
        Labels for the two recordings.
    save_path : str or None, optional
        Filesystem path to save the generated comparison figure.
    unit : str, optional
        Unit mode: "auto" (default), "ms", or "s".
    figsize : tuple of float, optional
        Dimensions of the side-by-side figure in inches. Default is (16, 7.5).

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure.
    """
    clean_arr = np.asarray(rr_clean, dtype=np.float64)
    arrhy_arr = np.asarray(rr_arrhythmic, dtype=np.float64)

    # Determine scaling and unit label
    if unit == "ms":
        if np.nanmean(clean_arr) < 10.0:
            clean_arr = clean_arr * 1000.0
        if np.nanmean(arrhy_arr) < 10.0:
            arrhy_arr = arrhy_arr * 1000.0
        unit_label = "ms"
    elif unit == "s":
        if np.nanmean(clean_arr) >= 10.0:
            clean_arr = clean_arr / 1000.0
        if np.nanmean(arrhy_arr) >= 10.0:
            arrhy_arr = arrhy_arr / 1000.0
        unit_label = "s"
    else:  # auto
        if np.nanmean(clean_arr) < 10.0:
            clean_arr = clean_arr * 1000.0
        if np.nanmean(arrhy_arr) < 10.0:
            arrhy_arr = arrhy_arr * 1000.0
        unit_label = "ms"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Clean / Normal Sinus plot (Left panel)
    clean_title = f"{labels[0]}\n[Healthy Autonomic Tone — Compact Physiological Cluster]"
    _draw_poincare_axes(
        ax=axes[0],
        rr_intervals=clean_arr,
        title=clean_title,
        unit_label=unit_label,
        point_color="#1f77b4",
        ellipse_color="#d62728",
    )

    # 2. Arrhythmic plot (Right panel)
    arrhy_title = f"{labels[1]}\n[Cardiac Arrhythmia — Wide Dispersion & Ectopic Outliers]"
    _draw_poincare_axes(
        ax=axes[1],
        rr_intervals=arrhy_arr,
        title=arrhy_title,
        unit_label=unit_label,
        point_color="#9467bd",
        ellipse_color="#d62728",
    )

    # Overall figure title
    fig.suptitle(
        "Poincaré Plot Analysis: Healthy Normal Sinus vs Cardiac Arrhythmia\n"
        r"Non-linear HRV Dynamics ($SD_1$: Short-term parasympathetic, $SD_2$: Long-term autonomic tone)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
