"""Math helper functions for the radpair package.

Provides unit conversion, tensor rotation, Pascal-triangle generation,
and chunked Gaussian spectrum summation.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import os
import sys
from copy import deepcopy
from functools import lru_cache
from itertools import product

import numpy as np
import scipy.constants as constant
from eprbase import grid, spectra

from radpair._types import Spinsystem

# Column index of the z-component in a 3×3 tensor.
_Z_COLUMN = 2

# Factor for converting a Gaussian FWHM to a squared sigma (in angular-frequency units).
_GAUSSIAN_FWHM_TO_SIGMA = 4 * np.log(2)

# Reference electron gyromagnetic ratio in rad s⁻¹ per milliTesla.
# gamma = 2*pi * g * mu_B / h, with g = 2 (reference value, not g_e).
# The 1e-3 factor converts from per-Tesla to per-milliTesla so that the
# internal angular-frequency values are consistent with the milliTesla
# field axis used throughout do_simulation.  The back-conversion
# (res_fields / _GAMMA_E_REF * 1e3) cancels this factor, producing
# resonance fields in milliTesla.
# Previously hardcoded as ``tesang = 1.75880474e8``; now derived from
# scipy.constants for traceability.
_GAMMA_E_REF = 2 * np.pi * 2 * constant.value("Bohr magneton in Hz/T") * 1e-3


def tensor_rotation(
    tensor: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray | None = None,
) -> np.ndarray:
    r"""Rotate a tensor (or batch of tensors) via Euler transformation.

    The Euler matrix *O* of the SO(3) group is set up in y-convention
    (already in multiplied form) and the orthogonal similarity
    transformation is carried out:

    .. math::

        T' = O^{\mathsf{T}} \cdot T \cdot O

    where :math:`O^{-1} = O^{\mathsf{T}}` (orthogonality).

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to be rotated.  A single 2-D array of shape ``(3, 3)``
        or a batch of 2-D arrays of shape ``(N, 3, 3)``.
    phi : np.ndarray
        Phi (Euler) angles in radians, shape ``(N,)``.
    theta : np.ndarray
        Theta (Euler) angles in radians, shape ``(N,)``.
    psi : np.ndarray, optional
        Psi (Euler) angles in radians, shape ``(N,)``.  If ``None``,
        zeros are used (default).

    Returns
    -------
    np.ndarray
        Rotated tensor(s).  Shape matches the input ``tensor`` except
        that the leading dimension becomes ``N`` (the number of angles).

    Raises
    ------
    ValueError
        If ``tensor`` does not have 2 or 3 dimensions.
    """
    if psi is None:
        psi = np.zeros(phi.size)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    costhet = np.cos(theta)
    sinthet = np.sin(theta)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    eulermatrix = np.zeros((phi.size, 3, 3))
    eulermatrix[:, 0, 0] = cosphi * costhet * cospsi - sinphi * sinpsi
    eulermatrix[:, 0, 1] = -cosphi * costhet * sinpsi - sinphi * cospsi
    eulermatrix[:, 0, 2] = cosphi * sinthet
    eulermatrix[:, 1, 0] = sinphi * costhet * cospsi + cosphi * sinpsi
    eulermatrix[:, 1, 1] = -sinphi * costhet * sinpsi + cosphi * cospsi
    eulermatrix[:, 1, 2] = sinphi * sinthet
    eulermatrix[:, 2, 0] = -sinthet * cospsi
    eulermatrix[:, 2, 1] = sinthet * sinpsi
    eulermatrix[:, 2, 2] = costhet

    if tensor.ndim == 2:
        rot_1 = np.einsum("ij, ajk -> aik", tensor, eulermatrix)
    elif tensor.ndim == 3:
        rot_1 = np.einsum("aij, ajk -> aik", tensor, eulermatrix)
    else:
        raise ValueError("Tensor has wrong dimensions!")
    rotated_tensor = np.einsum("aji, ajk -> aik", eulermatrix, rot_1)

    return rotated_tensor


@lru_cache
def get_multiplicity(spin: float) -> int:
    r"""Return the multiplicity of a particle with spin *S*.

    .. math::

        M = 2S + 1

    Parameters
    ----------
    spin : float
        Magnetic spin quantum number (must be non-negative and a
        multiple of 0.5).

    Returns
    -------
    int
        Multiplicity (number of Zeeman levels).

    Raises
    ------
    ValueError
        If ``spin`` is negative or not a multiple of 0.5.
    """
    if spin < 0.0:
        raise ValueError("Spin can't be negative!")
    if (spin % 0.5) > 1e-3:
        raise ValueError("Spin must be divisible by 0.5!")

    multiplicity = int(2 * spin + 1)

    return multiplicity


@lru_cache
def get_generalized_Pascal(number: int, spin: float) -> np.ndarray:
    """Compute a generalized Pascal triangle for ``number`` nuclei of spin ``spin``.

    Returns the relative intensities of the hyperfine lines for a group
    of ``number`` chemically equivalent nuclei with magnetic spin
    ``spin``.

    (c) Stephan Rein
    Modified by: Florian Quintes

    Parameters
    ----------
    number : int
        Number of chemically equivalent nuclei (>= 0).
    spin : float
        Magnetic spin quantum number (>= 0, multiple of 0.5).

    Returns
    -------
    np.ndarray
        Array of relative intensities.  For ``number == 0`` returns
        ``[1]``.

    Raises
    ------
    ValueError
        If ``number`` or ``spin`` is negative, or ``spin`` is not a
        multiple of 0.5.
    TypeError
        If ``number`` is not an integer.
    """
    if number < 0.0:
        raise ValueError("Number can't be negativ!")
    if spin < 0.0:
        raise ValueError("Spin can't be negativ!")
    if not isinstance(number, int):
        raise TypeError("Number must be a natural number!")
    if (spin % 0.5) > 1e-3:
        raise ValueError("Spin must be divisible by 0.5!")

    n = int(number)
    if n == 0:
        return np.ones(1)
    else:
        s0 = int(2 * spin * n + 1)
        A = np.zeros((n, s0))
        A[0, 0 : int(2 * spin + 1)] = 1
        I2 = 2 * spin
        for i in range(1, n):
            for j in range(s0):
                if j + I2 >= s0:
                    ub = int(s0 - 1)
                else:
                    ub = int(j + I2)
                if j - I2 < 0:
                    lb = 0
                else:
                    lb = int(j - I2)
                A[i, j] = np.sum(A[i - 1, lb : ub - int(2 * spin) + 1])
        return A[n - 1, :]


@lru_cache
def get_normalized_Pascal(number: int, spin: float) -> np.ndarray:
    """Compute a normalized generalized Pascal triangle (sum = 1).

    Wraps :func:`get_generalized_Pascal` and rescales the result so
    that all intensities sum to 1.

    Parameters
    ----------
    number : int
        Number of chemically equivalent nuclei (>= 0).
    spin : float
        Magnetic spin quantum number (>= 0, multiple of 0.5).

    Returns
    -------
    np.ndarray
        Normalized array of relative intensities summing to 1.
    """
    pascal_line = get_generalized_Pascal(number, spin)
    pascal_line = pascal_line / pascal_line.sum()

    return pascal_line


def get_D_diag(D: float, E: float) -> np.ndarray:
    r"""Return the diagonal elements of the ZFS *D*-tensor.

    The diagonal tensor is constructed from the zero-field splitting
    parameters *D* and *E* as:

    .. math::

        \mathrm{diag}(D) = \begin{pmatrix} D - E & 0 & 0 \\
                0 & D + E & 0 \\
                0 & 0 & -2D \end{pmatrix}

    Parameters
    ----------
    D : float
        ZFS parameter *D*.
    E : float
        ZFS parameter *E*.

    Returns
    -------
    np.ndarray
        Array of shape ``(3,)`` containing the diagonal elements.
    """
    return np.array([D - E, D + E, -2 * D])


def MHz_2_T(nu: float | np.ndarray, g_tensor: np.ndarray) -> float | np.ndarray:
    r"""Convert a frequency in MHz to the corresponding magnetic field in Tesla.

    The conversion uses the isotropic g-value and the Bohr magneton:

    .. math::

        B = \frac{\nu_{\mathrm{MHz}} \times 10^{6}}
        {g_{\mathrm{iso}} \cdot \mu_{B} \times 10^{-3}}

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency (or array of frequencies) in MHz.
    g_tensor : np.ndarray
        g-tensor diagonal elements, shape ``(3,)``.  All values must be
        positive.

    Returns
    -------
    float or np.ndarray
        Magnetic field in Tesla (scalar if ``nu`` is scalar, array
        otherwise).

    Raises
    ------
    ValueError
        If any element of ``g_tensor`` is not positive.
    """
    if not (g_tensor > 0).all():
        raise ValueError("All values of the g-Tensor need to be higher than 0!")

    mu_b = constant.value("Bohr magneton in Hz/T")
    g_iso = g_tensor.sum() / 3
    nu_tesla = 1e6 * nu / (g_iso * mu_b)

    return nu_tesla


# ---------------------------------------------------------------------------
# Composable simulation stages (extracted from do_simulation)
# ---------------------------------------------------------------------------


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

    try:
        theta_angles.shape[0]
    except IndexError:
        theta_angles = np.array([theta_angles])
        phi_angles = np.array([phi_angles])

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


def compute_hyperfine_combinations(
    Sys: Spinsystem,
    a_projections: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Compute sum and difference hyperfine matrices and spectral weights.

    For each combination of magnetic spin projections across all nuclei
    groups, computes the sum hyperfine (A_1) and difference hyperfine
    (A_2) contributions, weighted by Pascal-triangle intensities.

    Parameters
    ----------
    Sys : Spinsystem
        Spin-system object with nuclei group parameters (nuclei_n,
        nuclei_I, donor_list, acceptor_list).
    a_projections : list[np.ndarray]
        Effective hyperfine couplings for each nuclei group,
        each of shape ``(N,)`` where N is the number of orientations.

    Returns
    -------
    A_1 : np.ndarray
        Sum hyperfine multiplied by 2, shape ``(N, n_comb, 1)``.
    A_2 : np.ndarray
        Difference hyperfine multiplied by 2, shape ``(N, n_comb, 1)``.
    spec_weights : list[float]
        Spectral weights (Pascal-triangle products) for each combination.
    """
    core_data = []
    core_types = []

    for i in range(len(Sys.A_tensors)):
        if i in Sys.acceptor_list:
            ct = 1
            n = Sys.nuclei_n[i]
            I = Sys.nuclei_I[i]
        elif i in Sys.donor_list:
            ct = -1
            n = Sys.nuclei_n[i]
            I = Sys.nuclei_I[i]
        else:
            ct = 0
            n = 0
            I = 0.0

        total_spin = n * I
        multi = get_multiplicity(total_spin)
        mI_vector = np.linspace(-total_spin, total_spin, multi).astype(np.float32)
        pascal = get_normalized_Pascal(n, I)
        mI_len = mI_vector.size

        hyperfine_matrix = np.outer(mI_vector, a_projections[i - 1])

        core_data.append((hyperfine_matrix, pascal, mI_len))
        core_types.append(ct)

    spec_weights = []
    sums_hyperfine = []
    diffs_hyperfine = []

    for indices in product(*[range(cd[2]) for cd in core_data]):
        sum_hf = sum(cd[0][idx] for cd, idx in zip(core_data, indices))
        diff_hf = sum(
            ct * cd[0][idx] for ct, cd, idx in zip(core_types, core_data, indices)
        )
        weight = 1.0
        for cd, idx in zip(core_data, indices):
            weight *= cd[1][idx]

        spec_weights.append(weight)
        sums_hyperfine.append(sum_hf)
        diffs_hyperfine.append(diff_hf)

    sum_hyperfine = np.array(sums_hyperfine).T
    diff_hyperfine = np.array(diffs_hyperfine).T

    A_1 = sum_hyperfine * 2
    A_2 = diff_hyperfine * 2

    A_1 = A_1[:, :, np.newaxis]
    A_2 = A_2[:, :, np.newaxis]

    return A_1, A_2, spec_weights


def compute_resonance_fields(
    J_ex: float,
    freq_mw: float,
    g1: np.ndarray,
    g2: np.ndarray,
    D: np.ndarray,
    A_1: np.ndarray,
    A_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate resonance fields, quantum beats, and transition widths.

    Solves the analytic Hamiltonian for the four allowed transitions of
    the spin-correlated radical pair, producing resonance fields,
    frequency offsets (:math:`\Delta\omega`), quantum beat frequencies,
    and transition widths (intensity derivatives).

    Parameters
    ----------
    J_ex : float
        Exchange interaction in angular-frequency units.
    freq_mw : float
        Microwave frequency in angular-frequency units.
    g1 : np.ndarray
        zz-projection of the rotated g1 tensor, shape ``(N,)``.
    g2 : np.ndarray
        zz-projection of the rotated g2 tensor, shape ``(N,)``.
    D : np.ndarray
        zz-projection of the rotated D tensor, shape ``(N,)``.
    A_1 : np.ndarray
        Sum hyperfine, shape ``(N, n_comb, 1)``.
    A_2 : np.ndarray
        Difference hyperfine, shape ``(N, n_comb, 1)``.

    Returns
    -------
    res_fields : np.ndarray
        Resonance fields for 4 transitions, shape ``(N, n_comb, 4)``.
    delta_omega : np.ndarray
        Frequency offsets, shape ``(N, n_comb, 4)``.
    quantum_beat : np.ndarray
        Quantum beat frequencies, shape ``(N, n_comb, 4)``.
    widths : np.ndarray
        Transition widths, shape ``(N, 4, n_comb)``.
    """
    D = D[:, np.newaxis, np.newaxis]
    dj_square = (D + J_ex) ** 2

    g1 = g1[:, np.newaxis, np.newaxis]
    g2 = g2[:, np.newaxis, np.newaxis]
    g_m_g = g1 - g2
    g_m_g_s = g_m_g**2
    g_p_g = g1 + g2
    g_p_g_s = g_p_g**2
    g_s_m_g_s = g1**2 - g2**2
    g_g = g1 * g2

    pre_part_1_1 = +J_ex - 2 * D + freq_mw - 0.5 * A_1
    pre_part_1_2 = -J_ex + 2 * D + freq_mw - 0.5 * A_1
    pre_part_2 = 0.5 * A_2 * g_m_g

    pre_1 = pre_part_1_1 * g_p_g + pre_part_2
    pre_2 = pre_part_1_2 * g_p_g + pre_part_2

    sqrt_a_1 = J_ex**2 + 0.25 * A_2**2
    sqrt_a_2 = 8 * J_ex * D + 4 * D**2

    sqrt_1_1 = A_2 * (J_ex - 0.5 * A_1 + freq_mw - 2 * D)
    sqrt_1_2 = (
        -J_ex * A_1
        + 0.25 * A_1**2
        + 2 * J_ex * freq_mw
        - A_1 * freq_mw
        + 2 * A_1 * D
        + freq_mw**2
        - 4 * freq_mw * D
        - 4 * J_ex * D
        + 4 * D**2
    )

    sqrt_2_1 = A_2 * (-J_ex - 0.5 * A_1 + freq_mw + 2 * D)
    sqrt_2_2 = (
        J_ex * A_1
        + 0.25 * A_1**2
        - 2 * J_ex * freq_mw
        - A_1 * freq_mw
        - 2 * A_1 * D
        + freq_mw**2
        - 4 * freq_mw * D
        - 4 * J_ex * D
        + 4 * D**2
    )

    sqrt_a = sqrt_a_1 * g_p_g_s + sqrt_a_2 * g_g
    sqrt_1 = sqrt_a + sqrt_1_1 * g_s_m_g_s + sqrt_1_2 * g_m_g_s
    sqrt_2 = sqrt_a + sqrt_2_1 * g_s_m_g_s + sqrt_2_2 * g_m_g_s

    sqrt_1 = np.sqrt(sqrt_1)
    sqrt_2 = np.sqrt(sqrt_2)

    B_12 = (pre_1 - sqrt_1) / (2 * g_g)
    B_34 = (pre_2 - sqrt_2) / (2 * g_g)
    B_13 = (pre_1 + sqrt_1) / (2 * g_g)
    B_24 = (pre_2 + sqrt_2) / (2 * g_g)

    res_fields = np.squeeze(np.stack([B_12, B_34, B_13, B_24], axis=2), axis=3)
    delta_omega = 0.5 * (res_fields * g_m_g + A_2)
    quantum_beat = np.sqrt(dj_square + delta_omega**2)
    dE_dB = (g_m_g**2 * res_fields**2 - g_m_g * A_2) * 0.5 / quantum_beat / _GAMMA_E_REF

    dB_12_dB = g_p_g[:, :, 0] - dE_dB[:, :, 0]
    dB_34_dB = g_p_g[:, :, 0] - dE_dB[:, :, 1]
    dB_13_dB = g_p_g[:, :, 0] + dE_dB[:, :, 2]
    dB_24_dB = g_p_g[:, :, 0] + dE_dB[:, :, 3]

    widths = 1 / np.stack([dB_12_dB, dB_34_dB, dB_13_dB, dB_24_dB], axis=1)

    return res_fields, delta_omega, quantum_beat, widths


def compute_intensities(
    delta_omega: np.ndarray,
    quantum_beat: np.ndarray,
) -> np.ndarray:
    r"""Calculate line intensities from phase angles.

    The phase angle :math:`\varphi` is obtained from
    :math:`\sin(2\varphi) = \Delta\omega / \Omega_{\mathrm{beat}}` and
    the intensity is :math:`\sin^2\varphi \cdot \cos^2\varphi` with an
    alternating sign pattern ``[+1, -1, +1, -1]`` for the four
    transitions.

    Parameters
    ----------
    delta_omega : np.ndarray
        Frequency offsets, shape ``(N, n_comb, 4)``.
    quantum_beat : np.ndarray
        Quantum beat frequencies, shape ``(N, n_comb, 4)``.

    Returns
    -------
    np.ndarray
        Line intensities with absorptive/emissive pattern
        ``[+1, -1, +1, -1]``, shape ``(N, n_comb, 4)``.
    """
    phase_angle = 0.5 * np.arcsin(delta_omega / quantum_beat)
    sin_phase_angle = np.sin(phase_angle) ** 2
    cos_phase_angle = 1 - sin_phase_angle

    intensities = (sin_phase_angle * cos_phase_angle) * np.array(
        [1, -1, 1, -1]
    ).reshape((1, 1, 4))

    return intensities


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
