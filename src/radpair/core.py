"""Core simulation routines for cw-EPR spectra of radical pairs.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import numpy as np

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

    The simulation is composed of eight stages, each implemented as a
    separate function in :mod:`radpair.functions`:

    1. :func:`~radpair.functions.prepare_spinsystem` — unit conversion
    2. :func:`~radpair.functions.setup_orientation_grid` — grid setup
    3. :func:`~radpair.functions.build_tensors` — diagonal tensors
    4. :func:`~radpair.functions.rotate_tensors` — frame + orientation rotation
    5. :func:`~radpair.functions.compute_hyperfine_combinations` — hyperfine sums
    6. :func:`~radpair.functions.compute_resonance_fields` — resonance fields
    7. :func:`~radpair.functions.compute_intensities` — line intensities
    8. :func:`~radpair.functions.assemble_spectrum` — final spectrum

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
    Sys, freq_mw, _ = fun.prepare_spinsystem(
        spinsystem, experiment.freq_mw, experiment.B_z
    )

    theta, phi, theta_fine, phi_fine, weights, interp_mode = fun.setup_orientation_grid(
        simopt.grid_points, simopt.refinement
    )

    all_tensors, frame_angles = fun.build_tensors(Sys)

    g1, g2, D, a_projections = fun.rotate_tensors(all_tensors, frame_angles, theta, phi)

    A_1, A_2, spec_weights = fun.compute_hyperfine_combinations(Sys, a_projections)

    res_fields, delta_omega, quantum_beat, widths = fun.compute_resonance_fields(
        Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
    )

    intensities = fun.compute_intensities(delta_omega, quantum_beat)

    intensity = fun.assemble_spectrum(
        res_fields,
        intensities,
        widths,
        spec_weights,
        spinsystem.width_gauss,
        experiment.B_z,
        theta,
        phi,
        theta_fine,
        phi_fine,
        weights,
        interp_mode,
    )

    return intensity


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
