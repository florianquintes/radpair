"""Pipeline stages: unit conversion, grid setup, tensor construction, and rotation.

These functions compose the first four stages of the simulation
pipeline, converting user-supplied spin-system parameters into
orientation-dependent tensor projections ready for the Hamiltonian
solver.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from copy import deepcopy

import numpy as np
import scipy.constants as constant
from eprbase import grid

from radpair._types import Spinsystem
from radpair.hamiltonian import (
    _GAMMA_E_REF,
    _GAUSSIAN_FWHM_TO_SIGMA,
    _Z_COLUMN,
    MHz_2_T,
    get_D_diag,
    tensor_rotation,
)


def prepare_spinsystem(
    spinsystem: Spinsystem,
    freq_mw: float,
    B_z: np.ndarray,
) -> tuple[Spinsystem, float, np.ndarray]:
    """Deep-copy the spin system and convert all parameters to internal units.

    Performs unit conversion from physical units (MHz, mT, Hz) to
    internal angular-frequency units.  The original *spinsystem* is not
    modified.

    Parameters
    ----------
    spinsystem : Spinsystem
        Spin-system object with attributes in physical units.
    freq_mw : float
        Microwave frequency in Hz.
    B_z : np.ndarray
        Magnetic field axis in milliTesla.

    Returns
    -------
    sys : Spinsystem
        Deep-copied spin system with all parameters converted to
        internal angular-frequency units.
    freq_mw : float
        Microwave frequency converted to angular-frequency units.
    B_z : np.ndarray
        Magnetic field axis converted to angular-frequency units.
    """
    Sys = deepcopy(spinsystem)
    B_z = 1 * B_z
    freq_mw = 1 * freq_mw

    mu_b = constant.value("Bohr magneton in Hz/T")
    g_12 = (Sys.g1 + Sys.g2) / 2

    for core_n in Sys.acceptor_list:
        Sys.A_tensors[core_n] = MHz_2_T(Sys.A_tensors[core_n], Sys.g1)
    for core_n in Sys.donor_list:
        Sys.A_tensors[core_n] = MHz_2_T(Sys.A_tensors[core_n], Sys.g2)

    Sys.D = MHz_2_T(Sys.D, g_12)
    Sys.E = MHz_2_T(Sys.E, g_12)
    Sys.J_ex = MHz_2_T(Sys.J_ex, g_12)

    freq_mw /= 2 * mu_b
    B_z *= 1e-3
    Sys.width_gauss *= 1e-3

    for i in range(len(Sys.A_tensors)):
        a = Sys.A_tensors[i]
        if "int" in str(a.dtype):
            a = a.astype("float64")
        Sys.A_tensors[i] = a * 0.5 * _GAMMA_E_REF

    Sys.D /= 3
    Sys.D *= 0.5
    Sys.E *= 0.5

    Sys.g1 *= 0.5
    Sys.g2 *= 0.5

    Sys.D *= _GAMMA_E_REF
    Sys.E *= _GAMMA_E_REF
    Sys.J_ex *= _GAMMA_E_REF

    Sys.width_gauss *= _GAMMA_E_REF
    Sys.width_gauss = Sys.width_gauss**2 / _GAUSSIAN_FWHM_TO_SIGMA

    freq_mw *= _GAMMA_E_REF
    B_z *= _GAMMA_E_REF

    return Sys, freq_mw, B_z


def setup_orientation_grid(
    knots: int,
    refinement: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    bool,
]:
    """Set up the orientation grid for spherical integration.

    Creates a coarse (and optionally fine) orientation grid for
    orientational averaging of the EPR spectrum.

    Parameters
    ----------
    knots : int
        Number of orientation-grid knots.
    refinement : int
        Interpolation factor.  ``1`` disables interpolation; values > 1
        enable interpolation onto a finer grid.

    Returns
    -------
    theta_angles : np.ndarray
        Polar angles of the coarse grid.
    phi_angles : np.ndarray
        Azimuthal angles of the coarse grid.
    theta_fine : np.ndarray or None
        Polar angles of the fine grid (``None`` if no interpolation).
    phi_fine : np.ndarray or None
        Azimuthal angles of the fine grid (``None`` if no interpolation).
    weights : np.ndarray
        Integration weights for the (fine or coarse) grid.
    interpolation_mode : bool
        Whether interpolation is enabled.
    """
    grid_ = grid.Grid(knots=knots)
    sym = "Ci"
    spherical = grid_.get_grid(sym)
    theta_angles, phi_angles = spherical[:, 1], spherical[:, 2]

    interpolation_mode = refinement > 1
    if interpolation_mode:
        grid_fine = grid.Grid(knots=knots * int(refinement))
        spherical_fine = grid_fine.get_grid(sym)
        theta_fine, phi_fine = spherical_fine[:, 1], spherical_fine[:, 2]
        weights = grid_fine.get_areas()[np.newaxis, :]
    else:
        theta_fine = None
        phi_fine = None
        weights = grid_.get_areas()[np.newaxis, :]

    theta_angles = np.atleast_1d(theta_angles)
    phi_angles = np.atleast_1d(phi_angles)

    return theta_angles, phi_angles, theta_fine, phi_fine, weights, interpolation_mode


def build_tensors(
    Sys: Spinsystem,
) -> tuple[np.ndarray, np.ndarray]:
    """Build diagonal tensors and extract Euler frame angles from the spin system.

    Parameters
    ----------
    Sys : Spinsystem
        Spin-system object with parameters in internal units (output of
        :func:`prepare_spinsystem`).

    Returns
    -------
    all_tensors : np.ndarray
        Stacked diagonal tensors of shape ``(3 + n_nuclei, 3, 3)`` in the
        order ``[g1, g2, D, A0, A1, ...]``.
    frame_angles : np.ndarray
        Euler angles ``[alpha, beta, gamma]`` for each tensor, shape
        ``(3 + n_nuclei, 3)``.
    """
    frame_angles = np.array([Sys.g1_frame, Sys.g2_frame, Sys.D_frame, *Sys.A_frames])

    g1 = np.diag(Sys.g1)
    g2 = np.diag(Sys.g2)

    D_diag = get_D_diag(Sys.D, Sys.E)
    D = np.diag(D_diag)

    a_tensors = [np.diag(a) for a in Sys.A_tensors]

    all_tensors = np.array([g1, g2, D, *a_tensors])

    return all_tensors, frame_angles


def rotate_tensors(
    all_tensors: np.ndarray,
    frame_angles: np.ndarray,
    theta_angles: np.ndarray,
    phi_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Tilt tensors into their reference frames and rotate for each orientation.

    Performs two rotations: (1) tilt each tensor from its eigenframe to
    the lab frame using the Euler angles from ``frame_angles``, and
    (2) rotate each tilted tensor for every orientation ``(theta, phi)``
    on the integration grid.

    Parameters
    ----------
    all_tensors : np.ndarray
        Stacked diagonal tensors, shape ``(3 + n_nuclei, 3, 3)``.
    frame_angles : np.ndarray
        Euler angles for each tensor, shape ``(3 + n_nuclei, 3)``.
    theta_angles : np.ndarray
        Polar angles of the orientation grid.
    phi_angles : np.ndarray
        Azimuthal angles of the orientation grid.

    Returns
    -------
    g1 : np.ndarray
        zz-element of the rotated g1 tensor for each orientation,
        shape ``(N,)``.
    g2 : np.ndarray
        zz-element of the rotated g2 tensor, shape ``(N,)``.
    D : np.ndarray
        zz-element of the rotated D tensor, shape ``(N,)``.
    a_projections : list[np.ndarray]
        Effective hyperfine couplings for each nuclei group,
        each of shape ``(N,)``.
    """
    tilt_alpha = frame_angles[:, 0]
    tilt_beta = frame_angles[:, 1]
    tilt_gamma = frame_angles[:, 2]

    tilted_tensors = tensor_rotation(all_tensors, tilt_alpha, tilt_beta, tilt_gamma)

    g1 = tensor_rotation(tilted_tensors[0], phi_angles, theta_angles)[:, -1, -1]
    g2 = tensor_rotation(tilted_tensors[1], phi_angles, theta_angles)[:, -1, -1]
    D = tensor_rotation(tilted_tensors[2], phi_angles, theta_angles)[:, -1, -1]

    n_nuclei = all_tensors.shape[0] - 3
    a_projections = []
    for i in range(n_nuclei):
        rotated = tensor_rotation(tilted_tensors[3 + i], phi_angles, theta_angles)
        proj = np.sqrt((rotated[:, :, _Z_COLUMN] ** 2).sum(axis=1))
        a_projections.append(proj)

    return g1, g2, D, a_projections
