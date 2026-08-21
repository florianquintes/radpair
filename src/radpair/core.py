"""Core simulation routines for cw-EPR spectra of radical pairs.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from copy import deepcopy

import numpy as np
import scipy.constants as constant
from eprbase import grid, spectra
from eprbase import interpolation as interp

import radpair.classes as cl
import radpair.functions as fun
from radpair._types import Experiment, SimulationOptions, Spinsystem
from radpair._wrappers import multicore


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
        vars(Sys)["A" + str(core_n)] = fun.MHz_2_T(vars(Sys)["A" + str(core_n)], Sys.g1)
    for core_n in Sys.donor_list:
        vars(Sys)["A" + str(core_n)] = fun.MHz_2_T(vars(Sys)["A" + str(core_n)], Sys.g2)

    Sys.D = fun.MHz_2_T(Sys.D, g_12)
    Sys.E = fun.MHz_2_T(Sys.E, g_12)
    Sys.J_ex = fun.MHz_2_T(Sys.J_ex, g_12)

    freq_mw /= 2 * mu_b
    B_z *= 1e-3
    Sys.width_gauss *= 1e-3

    "rescale parameters"

    for i in range(1, 6):
        if "int" in str(vars(Sys)["A" + str(i)].dtype):
            vars(Sys)["A" + str(i)] = vars(Sys)["A" + str(i)].astype("float64")

    Sys.A1 *= 0.5
    Sys.A2 *= 0.5
    Sys.A3 *= 0.5
    Sys.A4 *= 0.5
    Sys.A5 *= 0.5

    Sys.D /= 3
    Sys.D *= 0.5
    Sys.E *= 0.5

    Sys.g1 *= 0.5
    Sys.g2 *= 0.5

    "convert everything from Tesla to angular frequency"

    tesang = 1.75880474e8

    Sys.A1 *= tesang
    Sys.A2 *= tesang
    Sys.A3 *= tesang
    Sys.A4 *= tesang
    Sys.A5 *= tesang

    Sys.D *= tesang
    Sys.E *= tesang
    Sys.J_ex *= tesang

    Sys.width_gauss *= tesang
    Sys.width_gauss = Sys.width_gauss**2 / (4 * np.log(2))

    freq_mw *= tesang
    B_z *= tesang

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

    tilt_alpha = np.array(
        [
            Sys.g1_frame[0],
            Sys.g2_frame[0],
            Sys.D_frame[0],
            Sys.A1_frame[0],
            Sys.A2_frame[0],
            Sys.A3_frame[0],
            Sys.A4_frame[0],
            Sys.A5_frame[0],
        ]
    )
    tilt_beta = np.array(
        [
            Sys.g1_frame[1],
            Sys.g2_frame[1],
            Sys.D_frame[1],
            Sys.A1_frame[1],
            Sys.A2_frame[1],
            Sys.A3_frame[1],
            Sys.A4_frame[1],
            Sys.A5_frame[1],
        ]
    )
    tilt_gamma = np.array(
        [
            Sys.g1_frame[2],
            Sys.g2_frame[2],
            Sys.D_frame[2],
            Sys.A1_frame[2],
            Sys.A2_frame[2],
            Sys.A3_frame[2],
            Sys.A4_frame[2],
            Sys.A5_frame[2],
        ]
    )

    "set up matrices"

    g1 = np.diag(Sys.g1)
    g2 = np.diag(Sys.g2)

    D_diag = fun.get_D_diag(Sys.D, Sys.E)
    D = np.diag(D_diag)

    a1 = np.diag(Sys.A1)
    a2 = np.diag(Sys.A2)
    a3 = np.diag(Sys.A3)
    a4 = np.diag(Sys.A4)
    a5 = np.diag(Sys.A5)

    "rotate matrices into reference frame"

    tilted_tensors = fun.tensor_rotation(
        np.array([g1, g2, D, a1, a2, a3, a4, a5]),
        tilt_alpha,
        tilt_beta,
        tilt_gamma,
    )

    g1 = cl.Matrix(tilted_tensors[0])
    g2 = cl.Matrix(tilted_tensors[1])
    D = cl.Matrix(tilted_tensors[2])
    a1 = cl.Matrix(tilted_tensors[3])
    a2 = cl.Matrix(tilted_tensors[4])
    a3 = cl.Matrix(tilted_tensors[5])
    a4 = cl.Matrix(tilted_tensors[6])
    a5 = cl.Matrix(tilted_tensors[7])

    "set up cores"

    core_1 = cl.Core(Sys.n1, Sys.I1)
    core_2 = cl.Core(Sys.n2, Sys.I2)
    core_3 = cl.Core(Sys.n3, Sys.I3)
    core_4 = cl.Core(Sys.n4, Sys.I4)
    core_5 = cl.Core(Sys.n5, Sys.I5)

    core_type = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    for core_num in Sys.acceptor_list:
        core_type[str(core_num)] = 1
    for core_num in Sys.donor_list:
        core_type[str(core_num)] = -1

    if core_type["1"] == 0:
        core_1 = cl.Core(0, 0)
    if core_type["2"] == 0:
        core_2 = cl.Core(0, 0)
    if core_type["3"] == 0:
        core_3 = cl.Core(0, 0)
    if core_type["4"] == 0:
        core_4 = cl.Core(0, 0)
    if core_type["5"] == 0:
        core_5 = cl.Core(0, 0)

    "theta/phi loop, rotate matrices"

    g1.matrot(theta=theta_angles, phi=phi_angles)
    g2.matrot(theta=theta_angles, phi=phi_angles)
    D.matrot(theta=theta_angles, phi=phi_angles)
    a1.matrot(theta=theta_angles, phi=phi_angles)
    a2.matrot(theta=theta_angles, phi=phi_angles)
    a3.matrot(theta=theta_angles, phi=phi_angles)
    a4.matrot(theta=theta_angles, phi=phi_angles)
    a5.matrot(theta=theta_angles, phi=phi_angles)
    g1 = g1.matrix_rot[:, -1, -1]
    g2 = g2.matrix_rot[:, -1, -1]
    D = D.matrix_rot[:, -1, -1]
    a1 = a1.get_hyperfine_projection()
    a2 = a2.get_hyperfine_projection()
    a3 = a3.get_hyperfine_projection()
    a4 = a4.get_hyperfine_projection()
    a5 = a5.get_hyperfine_projection()

    "calculate sum_g, diff_g, dj_square"
    D = D[:, np.newaxis, np.newaxis]
    dj_square = (D + Sys.J_ex) ** 2

    "hyperfine structure"

    core_1.set_hyperfine_matrix(a1)
    core_2.set_hyperfine_matrix(a2)
    core_3.set_hyperfine_matrix(a3)
    core_4.set_hyperfine_matrix(a4)
    core_5.set_hyperfine_matrix(a5)

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

    for mI_1 in range(core_1.mI_len):
        for mI_2 in range(core_2.mI_len):
            for mI_3 in range(core_3.mI_len):
                for mI_4 in range(core_4.mI_len):
                    for mI_5 in range(core_5.mI_len):
                        sum_hyperfine = (
                            core_1.hyperfine_matrix[mI_1]
                            + core_2.hyperfine_matrix[mI_2]
                            + core_3.hyperfine_matrix[mI_3]
                            + core_4.hyperfine_matrix[mI_4]
                            + core_5.hyperfine_matrix[mI_5]
                        )

                        diff_hyperfine = (
                            core_type["1"] * core_1.hyperfine_matrix[mI_1]
                            + core_type["2"] * core_2.hyperfine_matrix[mI_2]
                            + core_type["3"] * core_3.hyperfine_matrix[mI_3]
                            + core_type["4"] * core_4.hyperfine_matrix[mI_4]
                            + core_type["5"] * core_5.hyperfine_matrix[mI_5]
                        )

                        spectrum_weight = (
                            core_1.pascal[mI_1]
                            * core_2.pascal[mI_2]
                            * core_3.pascal[mI_3]
                            * core_4.pascal[mI_4]
                            * core_5.pascal[mI_5]
                        )

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
    dE_dB = (g_m_g**2 * res_fields**2 - g_m_g * A_2) * 0.5 / quantum_beat / tesang

    dB_12_dB = g_p_g[:, :, 0] - dE_dB[:, :, 0]
    dB_34_dB = g_p_g[:, :, 0] - dE_dB[:, :, 1]
    dB_13_dB = g_p_g[:, :, 0] + dE_dB[:, :, 2]
    dB_24_dB = g_p_g[:, :, 0] + dE_dB[:, :, 3]

    width_1 = 1 / dB_12_dB
    width_2 = 1 / dB_34_dB
    width_3 = 1 / dB_13_dB
    width_4 = 1 / dB_24_dB

    "calculate the phase angle"
    phase_angle = 0.5 * np.arcsin(delta_omega / quantum_beat)
    sin_phase_angle = np.sin(phase_angle) ** 2
    cos_phase_angle = 1 - sin_phase_angle

    "calculate line intensities"
    intensity = (sin_phase_angle * cos_phase_angle) * np.array([1, -1, 1, -1]).reshape(
        (1, 1, 4)
    )

    "reformat results for interpolation"
    res_fields = res_fields / tesang * 1e3
    width = np.stack([width_1, width_2, width_3, width_4], axis=1)

    transition = np.zeros((*res_fields.shape, 2))

    if True:
        fields = res_fields.copy()
        widths = width.copy()
        intensities = intensity.copy()
        transitions = transition.copy()

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
