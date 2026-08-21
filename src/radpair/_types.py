"""Protocol definitions describing the dynamic-attribute interfaces used by
:func:`radpair.core.do_simulation`.

Users typically pass plain objects (e.g. :class:`types.SimpleNamespace` or
simple classes) whose attributes are accessed dynamically via ``vars()`` and
``getattr``.  These protocols document the required attributes so that static
type checkers and IDEs can provide autocomplete and validation.

© M. Sc. Florian Quintes, 2026

@author: Florian Quintes
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Spinsystem(Protocol):
    """Protocol for the spin-system object passed to ``do_simulation``.

    Describes a singlet-born spin-correlated radical pair with up to five
    anisotropic nuclei groups.  Each group is assigned to either the donor
    or the acceptor radical via ``donor_list`` / ``acceptor_list``.

    Attributes
    ----------
    g1, g2 : np.ndarray
        Diagonal g-tensor elements of radical 1 (donor) and radical 2
        (acceptor), shape ``(3,)``.  All values must be positive.
    A1, A2, A3, A4, A5 : np.ndarray
        Diagonal hyperfine coupling tensors for nuclei groups 1–5 in MHz,
        each of shape ``(3,)``.  Inactive groups should be zero arrays.
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
    A1_frame, A2_frame, A3_frame, A4_frame, A5_frame : np.ndarray
        Euler angles for hyperfine tensors 1–5, each of shape ``(3,)``.
    n1, n2, n3, n4, n5 : int
        Number of chemically equivalent nuclei in each group (>= 0).
    I1, I2, I3, I4, I5 : float
        Nuclear spin of each group (multiple of 0.5, >= 0).
    donor_list : list[int]
        Core indices (1–5) assigned to the donor radical.
    acceptor_list : list[int]
        Core indices (1–5) assigned to the acceptor radical.
    """

    g1: np.ndarray
    g2: np.ndarray
    A1: np.ndarray
    A2: np.ndarray
    A3: np.ndarray
    A4: np.ndarray
    A5: np.ndarray
    D: float
    E: float
    J_ex: float
    width_gauss: float
    g1_frame: np.ndarray
    g2_frame: np.ndarray
    D_frame: np.ndarray
    A1_frame: np.ndarray
    A2_frame: np.ndarray
    A3_frame: np.ndarray
    A4_frame: np.ndarray
    A5_frame: np.ndarray
    n1: int
    I1: float
    n2: int
    I2: float
    n3: int
    I3: float
    n4: int
    I4: float
    n5: int
    I5: float
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
