"""Core simulation routines for cw-EPR spectra of radical pairs.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from copy import deepcopy
from itertools import product

import numpy as np
import scipy.constants as constant
from eprbase import grid, spectra
from eprbase import interpolation as interp

import radpair.classes as cl
import radpair.functions as fun
from radpair._types import Experiment, SimulationOptions, Spinsystem
from radpair._wrappers import multicore

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


def do_simulation(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> np.ndarray:
    """Simulate a cw-EPR spectrum for a spin-correlated radical pair.

    Solves the spin Hamiltonian analytically using a pseudo-secular
    approximation for the hyperfine couplings.  Supports up to five
    anisotropic nuclei groups, each assignable to the donor or acceptor
    radical.  Zero-field splitting (*D*, *E*) and exchange interaction
    (*J*) are included.

    Parameters
    ----------
    spinsystem : Spinsystem
        Spin-system object describing the radical pair.  See the
        :class:`~radpair._types.Spinsystem` protocol for required
        attributes.
    experiment : Experiment
        Experiment object with the magnetic-field axis and microwave
        frequency.  See the :class:`~radpair._types.Experiment` protocol.
    simopt : SimulationOptions
        Simulation options (grid density, interpolation, CPU cores).
        See the :class:`~radpair._types.SimulationOptions` protocol.

    Returns
    -------
    np.ndarray
        Real-valued intensity array of the simulated spectrum, matching
        the shape of ``experiment.B_z``.
    """
    Sys = deepcopy(spinsystem)
    B_z = 1 * experiment.B_z
    freq_mw = 1 * experiment.freq_mw

    "Transforming everything to Tesla"
    mu_b = constant.value("Bohr magneton in Hz/T")

    g_12 = (Sys.g1 + Sys.g2) / 2

    for core_n in Sys.acceptor_list:
        attr = "A" + str(core_n)
        setattr(Sys, attr, fun.MHz_2_T(getattr(Sys, attr), Sys.g1))
    for core_n in Sys.donor_list:
        attr = "A" + str(core_n)
        setattr(Sys, attr, fun.MHz_2_T(getattr(Sys, attr), Sys.g2))

    Sys.D = fun.MHz_2_T(Sys.D, g_12)
    Sys.E = fun.MHz_2_T(Sys.E, g_12)
    Sys.J_ex = fun.MHz_2_T(Sys.J_ex, g_12)

    freq_mw /= 2 * mu_b
    B_z *= 1e-3
    Sys.width_gauss *= 1e-3

    "rescale parameters"

    for i in range(1, 6):
        attr = "A" + str(i)
        a = getattr(Sys, attr)
        if "int" in str(a.dtype):
            a = a.astype("float64")
        setattr(Sys, attr, a * 0.5 * _GAMMA_E_REF)

    Sys.D /= 3
    Sys.D *= 0.5
    Sys.E *= 0.5

    Sys.g1 *= 0.5
    Sys.g2 *= 0.5

    "convert everything from Tesla to angular frequency"

    Sys.D *= _GAMMA_E_REF
    Sys.E *= _GAMMA_E_REF
    Sys.J_ex *= _GAMMA_E_REF

    Sys.width_gauss *= _GAMMA_E_REF
    Sys.width_gauss = Sys.width_gauss**2 / _GAUSSIAN_FWHM_TO_SIGMA

    freq_mw *= _GAMMA_E_REF
    B_z *= _GAMMA_E_REF

    "some initialization"
    grid_ = grid.Grid(knots=simopt.grid_points)
    sym = "Ci"
    spherical = grid_.get_grid(sym)
    theta_angles, phi_angles = spherical[:, 1], spherical[:, 2]

    interpolation_mode = simopt.refinement > 1
    if interpolation_mode:
        grid_fine = grid.Grid(knots=simopt.grid_points * int(simopt.refinement))
        spherical_fine = grid_fine.get_grid(sym)
        theta_fine, phi_fine = spherical_fine[:, 1], spherical_fine[:, 2]
        weights = grid_fine.get_areas()[np.newaxis, :]
    else:
        weights = grid_.get_areas()[np.newaxis, :]

    try:
        theta_angles.shape[0]
    except IndexError:
        theta_angles = np.array([theta_angles])
        phi_angles = np.array([phi_angles])

    frame_names = ["g1_frame", "g2_frame", "D_frame"] + [
        f"A{i}_frame" for i in range(1, 6)
    ]
    frame_angles = np.array([getattr(Sys, name) for name in frame_names])
    tilt_alpha = frame_angles[:, 0]
    tilt_beta = frame_angles[:, 1]
    tilt_gamma = frame_angles[:, 2]

    "set up matrices"

    g1 = np.diag(Sys.g1)
    g2 = np.diag(Sys.g2)

    D_diag = fun.get_D_diag(Sys.D, Sys.E)
    D = np.diag(D_diag)

    a_tensors = [np.diag(getattr(Sys, "A" + str(i))) for i in range(1, 6)]

    "rotate matrices into reference frame"

    all_tensors = np.array([g1, g2, D, *a_tensors])
    tilted_tensors = fun.tensor_rotation(all_tensors, tilt_alpha, tilt_beta, tilt_gamma)

    g1 = cl.Matrix(tilted_tensors[0])
    g2 = cl.Matrix(tilted_tensors[1])
    D = cl.Matrix(tilted_tensors[2])
    a_matrices = [cl.Matrix(tilted_tensors[3 + i]) for i in range(5)]

    "set up cores"

    cores = [
        cl.Core(getattr(Sys, "n" + str(i)), getattr(Sys, "I" + str(i)))
        for i in range(1, 6)
    ]

    core_type = {str(i): 0 for i in range(1, 6)}

    for core_num in Sys.acceptor_list:
        core_type[str(core_num)] = 1
    for core_num in Sys.donor_list:
        core_type[str(core_num)] = -1

    cores = [
        cl.Core(0, 0) if core_type[str(i)] == 0 else cores[i - 1] for i in range(1, 6)
    ]

    "theta/phi loop, rotate matrices"

    for m in [g1, g2, D, *a_matrices]:
        m.matrot(theta=theta_angles, phi=phi_angles)
    g1 = g1.matrix_rot[:, -1, -1]
    g2 = g2.matrix_rot[:, -1, -1]
    D = D.matrix_rot[:, -1, -1]
    a_projections = [a.get_hyperfine_projection() for a in a_matrices]

    "calculate sum_g, diff_g, dj_square"
    D = D[:, np.newaxis, np.newaxis]
    dj_square = (D + Sys.J_ex) ** 2

    "hyperfine structure"

    for core, a_proj in zip(cores, a_projections):
        core.set_hyperfine_matrix(a_proj)

    """Pre calc"""
    g1 = g1[:, np.newaxis, np.newaxis]
    g2 = g2[:, np.newaxis, np.newaxis]
    g_m_g = g1 - g2
    g_m_g_s = g_m_g**2
    g_p_g = g1 + g2
    g_p_g_s = g_p_g**2
    g_s_m_g_s = g1**2 - g2**2
    g_g = g1 * g2

    spec_weights = []
    sums_hyperfine = []
    diffs_hyperfine = []

    for indices in product(*[range(c.mI_len) for c in cores]):
        sum_hyperfine = sum(c.hyperfine_matrix[idx] for c, idx in zip(cores, indices))
        diff_hyperfine = sum(
            core_type[str(i + 1)] * c.hyperfine_matrix[idx]
            for i, (c, idx) in enumerate(zip(cores, indices))
        )
        spectrum_weight = 1.0
        for c, idx in zip(cores, indices):
            spectrum_weight *= c.pascal[idx]

        spec_weights.append(spectrum_weight)
        sums_hyperfine.append(sum_hyperfine)
        diffs_hyperfine.append(diff_hyperfine)

    sum_hyperfine = np.array(sums_hyperfine).T
    diff_hyperfine = np.array(diffs_hyperfine).T

    A_1 = sum_hyperfine * 2
    A_2 = diff_hyperfine * 2

    A_1 = A_1[:, :, np.newaxis]
    A_2 = A_2[:, :, np.newaxis]

    """calculate resonance fields"""

    pre_part_1_1 = +Sys.J_ex - 2 * D + freq_mw - 0.5 * A_1
    pre_part_1_2 = -Sys.J_ex + 2 * D + freq_mw - 0.5 * A_1
    pre_part_2 = 0.5 * A_2 * g_m_g

    pre_1 = pre_part_1_1 * g_p_g + pre_part_2
    pre_2 = pre_part_1_2 * g_p_g + pre_part_2

    sqrt_a_1 = Sys.J_ex**2 + 0.25 * A_2**2
    sqrt_a_2 = 8 * Sys.J_ex * D + 4 * D**2

    sqrt_1_1 = A_2 * (Sys.J_ex - 0.5 * A_1 + freq_mw - 2 * D)
    sqrt_1_2 = (
        -Sys.J_ex * A_1
        + 0.25 * A_1**2
        + 2 * Sys.J_ex * freq_mw
        - A_1 * freq_mw
        + 2 * A_1 * D
        + freq_mw**2
        - 4 * freq_mw * D
        - 4 * Sys.J_ex * D
        + 4 * D**2
    )

    sqrt_2_1 = A_2 * (-Sys.J_ex - 0.5 * A_1 + freq_mw + 2 * D)
    sqrt_2_2 = (
        Sys.J_ex * A_1
        + 0.25 * A_1**2
        - 2 * Sys.J_ex * freq_mw
        - A_1 * freq_mw
        - 2 * A_1 * D
        + freq_mw**2
        - 4 * freq_mw * D
        - 4 * Sys.J_ex * D
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

    "calculate the phase angle"
    phase_angle = 0.5 * np.arcsin(delta_omega / quantum_beat)
    sin_phase_angle = np.sin(phase_angle) ** 2
    cos_phase_angle = 1 - sin_phase_angle

    "calculate line intensities"
    intensities = (sin_phase_angle * cos_phase_angle) * np.array(
        [1, -1, 1, -1]
    ).reshape((1, 1, 4))

    "reformat results for interpolation"
    fields = res_fields / _GAMMA_E_REF * 1e3
    transitions = np.zeros((*fields.shape, 2))

    shp = (theta_angles.size, fields.shape[1] * fields.shape[2])
    fields = fields.reshape(shp)
    widths = widths.reshape(shp)
    intensities = intensities.reshape(shp)
    transitions = transitions.reshape((*shp, 2))

    if interpolation_mode:
        data = (fields, intensities, widths, transitions)

        interp_ = interp.Interpolator(theta_angles, phi_angles, data)

        fields = interp_.get_positions(theta_fine, phi_fine)
        intensities = interp_.get_intensities(theta_fine, phi_fine)
        widths = interp_.get_widths(theta_fine, phi_fine)
        transitions = interp_.get_transitions(theta_fine.shape[0])

    spec_weights = np.repeat(np.array([spec_weights]), 4)
    spec_weights = spec_weights.reshape((1, spec_weights.size))

    spectra_ = spectra.Spectra(
        fields,
        intensities * spec_weights,
        spinsystem.width_gauss * widths,
        transitions,
        weights=weights,
    )
    intensity = spectra_.by_summation(experiment.B_z)

    return np.nan_to_num(intensity)


def do_simulation_multicore(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> np.ndarray:
    """Simulate a cw-EPR spectrum using multiple CPU cores.

    Wraps :func:`do_simulation` with the :func:`~radpair._wrappers.multicore`
    decorator, which splits the magnetic-field axis across
    ``simopt.cpu_cores`` processes via :class:`multiprocessing.Pool`.
    Recommended for single simulations where wall-clock time matters.

    Parameters
    ----------
    spinsystem : Spinsystem
        Spin-system object (see :class:`~radpair._types.Spinsystem`).
    experiment : Experiment
        Experiment object (see :class:`~radpair._types.Experiment`).
    simopt : SimulationOptions
        Simulation options (see :class:`~radpair._types.SimulationOptions`).
        ``simopt.cpu_cores`` controls the number of worker processes
        (``0`` = auto-detect).

    Returns
    -------
    np.ndarray
        Real-valued intensity array of the simulated spectrum, matching
        the shape of ``experiment.B_z``.
    """
    multicore_sim = multicore(do_simulation)
    intensity = multicore_sim(spinsystem, experiment, simopt)

    return intensity
