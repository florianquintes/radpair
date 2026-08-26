"""Physics math: tensor rotation, unit conversion, and Hamiltonian stages.

Provides the analytic solution of the spin Hamiltonian for singlet-born
spin-correlated radical pairs, including tensor rotation, resonance-field
computation, and line-intensity calculation.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import numpy as np
import scipy.constants as constant

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
