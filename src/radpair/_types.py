"""Protocol definitions describing the dynamic-attribute interfaces used by
:func:`radpair.core.do_simulation`.

Users typically pass plain objects (e.g. :class:`types.SimpleNamespace` or
simple classes) whose attributes are accessed dynamically via ``getattr``
and ``setattr``.  These protocols document the required attributes so that
static type checkers and IDEs can provide autocomplete and validation.

© M. Sc. Florian Quintes, 2026

@author: Florian Quintes
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Spinsystem(Protocol):
    """Protocol for the spin-system object passed to ``do_simulation``.

    Describes a singlet-born spin-correlated radical pair with an
    arbitrary number of anisotropic nuclei groups.  Each group is
    assigned to either the donor or the acceptor radical via
    ``donor_list`` / ``acceptor_list`` (0-indexed).

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
    D : float
        Zero-field splitting parameter *D* in MHz.
    E : float
        Zero-field splitting parameter *E* in MHz.
    J_ex : float
        Exchange interaction *J* in MHz.
    width_gauss : float
        Gaussian linewidth in milliTesla (despite the attribute name).
    g1_frame, g2_frame, D_frame : np.ndarray
        Euler angles ``[alpha, beta, gamma]`` (radians) for the g-tensors
        and ZFS tensor relative to the lab frame, each of shape ``(3,)``.
    A_frames : list[np.ndarray]
        Euler angles ``[alpha, beta, gamma]`` (radians) for each hyperfine
        tensor.  Must have the same length as ``A_tensors``.
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
    D: float
    E: float
    J_ex: float
    width_gauss: float
    g1_frame: np.ndarray
    g2_frame: np.ndarray
    D_frame: np.ndarray
    A_frames: list[np.ndarray]
    donor_list: list[int]
    acceptor_list: list[int]


@runtime_checkable
class Experiment(Protocol):
    """Protocol for the experiment object passed to ``do_simulation``.

    Attributes
    ----------
    B_z : np.ndarray
        Magnetic field axis for the output spectrum in milliTesla.
    freq_mw : float
        Microwave frequency in Hz.
    magnetic_field : np.ndarray
        Magnetic field sweep axis in milliTesla (used by the multicore
        wrapper to split work across processes).
    """

    B_z: np.ndarray
    freq_mw: float
    magnetic_field: np.ndarray


@runtime_checkable
class SimulationOptions(Protocol):
    """Protocol for the simulation-options object passed to ``do_simulation``.

    Attributes
    ----------
    grid_points : int
        Number of orientation-grid knots for the spherical integration.
    refinement : int
        Interpolation factor.  ``1`` disables interpolation; values > 1
        enable interpolation onto a finer grid.
    cpu_cores : int
        Number of worker processes for multicore execution.  ``0`` means
        auto-detect via :func:`multiprocessing.cpu_count`.
    """

    grid_points: int
    refinement: int
    cpu_cores: int
