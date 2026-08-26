"""Chunked Gaussian summation and memory management.

Provides the final stage of the simulation pipeline: summing Gaussian
line shapes via chunked evaluation to bound peak memory usage.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import os
import sys

import numpy as np
from eprbase import spectra


def _get_available_ram() -> int:
    """Return available RAM in bytes.

    Tries :mod:`psutil` if installed, then falls back to platform-specific
    methods (``/proc/meminfo`` on Linux, ``vm_stat`` on macOS).  If all
    else fails, returns a conservative 1 GB default.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024

    if sys.platform == "darwin":
        import subprocess

        try:
            output = subprocess.check_output(["vm_stat"], text=True)
            for line in output.splitlines():
                if "free" in line.lower():
                    pages = int(line.split()[-1].rstrip("."))
                    return pages * 4096
        except (subprocess.CalledProcessError, IndexError, ValueError):
            pass

    return 1_000_000_000


def _compute_chunk_size(
    total_peaks: int, n_field: int, max_chunk_mb: int | None
) -> int:
    """Determine how many peaks fit in one chunk.

    The Gaussian array is float32, shape ``(chunk_size, n_field)``,
    consuming ``chunk_size * n_field * 4`` bytes.

    Parameters
    ----------
    total_peaks : int
        Total number of Gaussian peaks.
    n_field : int
        Number of field-axis points.
    max_chunk_mb : int or None
        Maximum chunk size in MB.  ``0`` or negative disables chunking.
        ``None`` auto-detects from available RAM (25% cap).

    Returns
    -------
    int
        Number of peaks per chunk (at least 1, at most ``total_peaks``).
    """
    if max_chunk_mb is not None and max_chunk_mb <= 0:
        return total_peaks

    bytes_per_peak = n_field * 4

    if max_chunk_mb is not None:
        max_bytes = max_chunk_mb * 1_000_000
    else:
        available = _get_available_ram()
        max_bytes = available // 4

    chunk_size = int(max_bytes // bytes_per_peak)
    return max(1, min(chunk_size, total_peaks))


def gaussian_summation(
    fields: np.ndarray,
    intensities: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
    field_axis: np.ndarray,
    max_chunk_mb: int | None = None,
) -> np.ndarray:
    """Sum Gaussian line shapes via chunked evaluation.

    Splits the peaks into chunks so that the dense ``(n_peaks_in_chunk,
    n_field)`` float32 array inside
    :class:`eprbase.spectra.Spectra` never exceeds ``max_chunk_mb``
    megabytes.  Each chunk is delegated to
    :meth:`eprbase.spectra.Spectra.by_summation`; the partial spectra
    are accumulated and returned.

    Parameters
    ----------
    fields : np.ndarray
        Resonance field centers, shape ``(n_orient, n_peaks_per_orient)``.
    intensities : np.ndarray
        Peak intensities, same shape as ``fields``.
    widths : np.ndarray
        Peak linewidths (FWHM), same shape as ``fields``.
    weights : np.ndarray
        Integration weights, broadcastable to the first dimension of
        ``fields`` (i.e. shape ``(1, n_orient)`` or ``(n_orient,)``).
    field_axis : np.ndarray
        Magnetic field axis, shape ``(n_field,)``.
    max_chunk_mb : int or None, optional
        Maximum memory in MB for a single chunk's Gaussian array.  If
        ``0`` or negative, no chunking is performed (all peaks in one
        pass).  If ``None``, the limit is auto-determined from
        available RAM via :func:`_get_available_ram`, targeting at most
        25% of available memory per chunk.

    Returns
    -------
    np.ndarray
        Real-valued spectrum, shape ``(n_field,)``.
    """
    n_orient, n_peaks = fields.shape
    n_field = field_axis.shape[0]
    total_peaks = n_orient * n_peaks

    chunk_size = _compute_chunk_size(total_peaks, n_field, max_chunk_mb)
    chunk_orient = max(1, chunk_size // n_peaks)

    weights_arr = (
        np.broadcast_to(weights, (1, n_orient)) if weights.ndim == 1 else weights
    )

    spectrum = np.zeros(n_field, dtype=np.float32)

    for start in range(0, n_orient, chunk_orient):
        end = min(start + chunk_orient, n_orient)

        chunk_fields = [fields[i] for i in range(start, end)]
        chunk_intensities = [intensities[i] for i in range(start, end)]
        chunk_widths = [widths[i] for i in range(start, end)]
        chunk_transitions = [np.zeros((n_peaks, 2)) for _ in range(end - start)]
        chunk_weights = weights_arr[:, start:end]

        spec = spectra.Spectra(
            chunk_fields,
            chunk_intensities,
            chunk_widths,
            chunk_transitions,
            weights=chunk_weights,
        )
        spectrum += spec.by_summation(field_axis)

    return spectrum
