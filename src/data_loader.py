"""
data_loader.py — MIT-BIH Arrhythmia Database loader.

Provides utilities for loading ECG records, annotations, and metadata
from the MIT-BIH Arrhythmia Database (360 Hz, 2-channel ECG) using the
``wfdb`` library.

Typical usage::

    signal, fs = load_record('100')          # MLII lead by default
    samples, symbols = get_beat_annotations('100')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Project root is two levels up from this file (src/data_loader.py → DSP/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Relative path from project root to the MIT-BIH data files.
DATA_DIR: str = os.path.join(
    _PROJECT_ROOT,
    "mitdb-1.0.0",
    "mit-bih-arrhythmia-database-1.0.0",
)

#: AAMI beat-type symbols present in the MIT-BIH annotation files.
#: Non-beat markers (e.g. '+' for rhythm change, '~' for noise) are excluded.
BEAT_ANNOTATION_SYMBOLS: set[str] = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r",
    "F", "e", "j", "n", "E", "/", "f", "Q", "?",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_record(
    record_id: str,
    channel: int = 0,
) -> tuple[np.ndarray, float]:
    """Load a single-channel ECG signal from a MIT-BIH record.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. ``'100'``, ``'105'``.
    channel : int, optional
        Zero-based channel index.  ``0`` corresponds to the MLII lead for
        most records.  Default is ``0``.

    Returns
    -------
    signal : np.ndarray
        1-D array of physical (mV) sample values for the requested channel.
    sampling_freq : float
        Sampling frequency in Hz (typically 360.0).

    Raises
    ------
    FileNotFoundError
        If the record files do not exist in :data:`DATA_DIR`.
    ValueError
        If *channel* is out of range for the record.
    """
    record_path = os.path.join(DATA_DIR, record_id)

    try:
        record = wfdb.rdrecord(record_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Record '{record_id}' not found in {DATA_DIR}"
        )

    if channel < 0 or channel >= record.n_sig:
        raise ValueError(
            f"Channel {channel} out of range — record '{record_id}' "
            f"has {record.n_sig} channel(s) (0–{record.n_sig - 1})."
        )

    signal: np.ndarray = record.p_signal[:, channel]
    sampling_freq: float = float(record.fs)
    return signal, sampling_freq


def load_annotations(
    record_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Load *all* annotations for a MIT-BIH record.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. ``'100'``.

    Returns
    -------
    sample_indices : np.ndarray
        1-D integer array of sample indices where annotations occur.
    symbols : list[str]
        Annotation symbols aligned with *sample_indices*.

    Raises
    ------
    FileNotFoundError
        If the annotation file does not exist.
    """
    record_path = os.path.join(DATA_DIR, record_id)

    try:
        ann = wfdb.rdann(record_path, "atr")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Annotations for record '{record_id}' not found in {DATA_DIR}"
        )

    sample_indices: np.ndarray = ann.sample
    symbols: list[str] = ann.symbol
    return sample_indices, symbols


def get_beat_annotations(
    record_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Load annotations filtered to beat types only.

    Non-beat markers such as ``'+'`` (rhythm change), ``'~'`` (signal
    quality change), and ``'|'`` (isolated artifact) are excluded.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. ``'100'``.

    Returns
    -------
    sample_indices : np.ndarray
        1-D integer array of sample indices for beat annotations.
    symbols : list[str]
        Beat-type symbols aligned with *sample_indices*.

    See Also
    --------
    BEAT_ANNOTATION_SYMBOLS : The set of symbols considered beats.
    """
    sample_indices, symbols = load_annotations(record_id)

    # Boolean mask: True where the symbol is a recognised beat type.
    mask = np.array(
        [sym in BEAT_ANNOTATION_SYMBOLS for sym in symbols], dtype=bool
    )

    filtered_indices: np.ndarray = sample_indices[mask]
    filtered_symbols: list[str] = [
        sym for sym, keep in zip(symbols, mask) if keep
    ]
    return filtered_indices, filtered_symbols


def list_records() -> list[str]:
    """Return sorted list of available record IDs in :data:`DATA_DIR`.

    Record IDs are derived by scanning for ``.hea`` (header) files.

    Returns
    -------
    list[str]
        Sorted record identifiers, e.g. ``['100', '101', ...]``.

    Raises
    ------
    FileNotFoundError
        If :data:`DATA_DIR` does not exist.
    """
    data_path = Path(DATA_DIR)

    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    record_ids = sorted(
        p.stem for p in data_path.glob("*.hea")
    )
    return record_ids


def get_record_info(record_id: str) -> dict[str, Any]:
    """Return metadata for a MIT-BIH record.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. ``'100'``.

    Returns
    -------
    dict
        Keys:

        - **sig_name** (*list[str]*) – channel / lead names,
          e.g. ``['MLII', 'V5']``.
        - **fs** (*float*) – sampling frequency in Hz.
        - **n_sig** (*int*) – number of signals (channels).
        - **sig_len** (*int*) – total number of samples per channel.
        - **units** (*list[str]*) – physical units for each channel,
          e.g. ``['mV', 'mV']``.

    Raises
    ------
    FileNotFoundError
        If the record files do not exist.
    """
    record_path = os.path.join(DATA_DIR, record_id)

    try:
        record = wfdb.rdrecord(record_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Record '{record_id}' not found in {DATA_DIR}"
        )

    info: dict[str, Any] = {
        "sig_name": record.sig_name,
        "fs": float(record.fs),
        "n_sig": record.n_sig,
        "sig_len": record.sig_len,
        "units": record.units,
    }
    return info
