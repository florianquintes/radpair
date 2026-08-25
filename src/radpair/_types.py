"""Typed dataclass definitions for the radpair simulation interface.

These :class:`~dataclasses.dataclass` types replace the former
``SimpleNamespace`` + ``Protocol`` approach with concrete, instantiable,
validated types.  Users construct ``Spinsystem``, ``Experiment``, and
``SimulationOptions`` objects and pass them to
:func:`radpair.core.do_simulation`.

© M. Sc. Florian Quintes, 2026

@author: Florian Quintes
"""

from dataclasses import dataclass, field, fields

import numpy as np


def _zeros3() -> np.ndarray:
    """Return a fresh 3-element zero array (default factory helper)."""
    return np.zeros(3)


@dataclass
class Spinsystem:
    """Spin-correlated radical pair with an arbitrary number of nuclei groups.

    Attributes
    ----------
    g1, g2 : np.ndarray
        Diagonal g-tensor elements of radical 1 (donor) and radical 2
        (acceptor), shape ``(3,)``.  All values must be positive.
    A_tensors : list[np.ndarray]
        Diagonal hyperfine coupling tensors for each nuclei group in MHz,
        each of shape ``(3,)``.  Inactive groups should be zero arrays.
    nuclei_n : list[int]
        Number of chemically equivalent nuclei in each group (>= 0).
        Must have the same length as ``A_tensors``.
    nuclei_I : list[float]
        Nuclear spin of each group (multiple of 0.5, >= 0).
        Must have the same length as ``A_tensors``.
    A_frames : list[np.ndarray]
        Euler angles ``[alpha, beta, gamma]`` (radians) for each hyperfine
        tensor.  Must have the same length as ``A_tensors``.
    width_gauss : float
        Gaussian linewidth in milliTesla (despite the attribute name).
        Must be positive.
    D : float
        Zero-field splitting parameter *D* in MHz.
    E : float
        Zero-field splitting parameter *E* in MHz.
    J_ex : float
        Exchange interaction *J* in MHz.
    g1_frame, g2_frame, D_frame : np.ndarray
        Euler angles ``[alpha, beta, gamma]`` (radians) for the g-tensors
        and ZFS tensor relative to the lab frame, each of shape ``(3,)``.
    donor_list : list[int]
        0-indexed positions of nuclei groups assigned to the donor
        radical.
    acceptor_list : list[int]
        0-indexed positions of nuclei groups assigned to the acceptor
        radical.
    """

    g1: np.ndarray
    g2: np.ndarray
    A_tensors: list[np.ndarray]
    nuclei_n: list[int]
    nuclei_I: list[float]
    A_frames: list[np.ndarray]
    width_gauss: float
    D: float = 0.0
    E: float = 0.0
    J_ex: float = 0.0
    g1_frame: np.ndarray = field(default_factory=_zeros3)
    g2_frame: np.ndarray = field(default_factory=_zeros3)
    D_frame: np.ndarray = field(default_factory=_zeros3)
    donor_list: list[int] = field(default_factory=list)
    acceptor_list: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.A_tensors)
        if len(self.nuclei_n) != n:
            raise ValueError(
                f"nuclei_n has {len(self.nuclei_n)} elements, "
                f"expected {n} (len(A_tensors))"
            )
        if len(self.nuclei_I) != n:
            raise ValueError(
                f"nuclei_I has {len(self.nuclei_I)} elements, "
                f"expected {n} (len(A_tensors))"
            )
        if len(self.A_frames) != n:
            raise ValueError(
                f"A_frames has {len(self.A_frames)} elements, "
                f"expected {n} (len(A_tensors))"
            )
        for i, a in enumerate(self.A_tensors):
            if a.shape != (3,):
                raise ValueError(f"A_tensors[{i}] has shape {a.shape}, expected (3,)")
        for i, f in enumerate(self.A_frames):
            if f.shape != (3,):
                raise ValueError(f"A_frames[{i}] has shape {f.shape}, expected (3,)")
        if not (self.g1 > 0).all():
            raise ValueError("All g1 values must be positive")
        if not (self.g2 > 0).all():
            raise ValueError("All g2 values must be positive")
        for i, n_i in enumerate(self.nuclei_n):
            if not isinstance(n_i, int):
                raise TypeError(f"nuclei_n[{i}] is {type(n_i).__name__}, expected int")
            if n_i < 0:
                raise ValueError(f"nuclei_n[{i}] = {n_i}, must be >= 0")
        for i, I_i in enumerate(self.nuclei_I):
            if I_i < 0:
                raise ValueError(f"nuclei_I[{i}] = {I_i}, must be >= 0")
            if (I_i % 0.5) > 1e-3:
                raise ValueError(f"nuclei_I[{i}] = {I_i}, must be a multiple of 0.5")
        if self.width_gauss <= 0:
            raise ValueError(f"width_gauss = {self.width_gauss}, must be > 0")
        all_indices = list(self.donor_list) + list(self.acceptor_list)
        for idx in all_indices:
            if idx < 0 or idx >= n:
                raise IndexError(f"nuclei index {idx} is out of range [0, {n})")
        overlap = set(self.donor_list) & set(self.acceptor_list)
        if overlap:
            raise ValueError(
                f"Indices {sorted(overlap)} appear in both donor_list and acceptor_list"
            )


@dataclass
class Experiment:
    """Experiment parameters for ``do_simulation``.

    Attributes
    ----------
    B_z : np.ndarray
        Magnetic field axis for the output spectrum in milliTesla.
    freq_mw : float
        Microwave frequency in Hz.
    magnetic_field : np.ndarray or None
        Magnetic field sweep axis in milliTesla (used by the multicore
        wrapper to split work across processes).  If ``None``, defaults
        to a copy of ``B_z``.
    """

    B_z: np.ndarray
    freq_mw: float
    magnetic_field: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.magnetic_field is None:
            self.magnetic_field = self.B_z.copy()


@dataclass
class SimulationOptions:
    """Simulation options for ``do_simulation``.

    Attributes
    ----------
    knots : int
        Number of orientation-grid knots for the spherical integration.
    refinement : int
        Interpolation factor.  ``1`` disables interpolation; values > 1
        enable interpolation onto a finer grid.
    cpu_cores : int
        Number of worker processes for multicore execution.  ``0`` means
        auto-detect via :func:`multiprocessing.cpu_count`.
    max_chunk_mb : int
        Maximum memory (in megabytes) allowed for a single Gaussian
        summation chunk.  If ``0`` or negative, no limit is enforced and
        all peaks are processed in one pass.  If ``None`` (default), the
        chunk size is determined automatically from available RAM.
        Smaller values reduce peak memory usage at the cost of more
        iterations; larger values are faster but may cause out-of-memory
        errors.  See :ref:`chunked-summation` for details.
    """

    knots: int = 12
    refinement: int = 1
    cpu_cores: int = 1
    max_chunk_mb: int | None = None


def spinsystem_field_names() -> list[str]:
    """Return the field names of :class:`Spinsystem` in definition order."""
    return [f.name for f in fields(Spinsystem)]
