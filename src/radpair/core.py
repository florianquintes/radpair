"""Core simulation routines for cw-EPR spectra of radical pairs.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import numpy as np
from eprbase import interpolation as interp

from radpair._types import Experiment, SimulationOptions, Spinsystem
from radpair.hamiltonian import (
    _GAMMA_E_REF,
    compute_intensities,
    compute_resonance_fields,
)
from radpair.hyperfine import compute_hyperfine_combinations
from radpair.pipeline import (
    build_tensors,
    prepare_spinsystem,
    rotate_tensors,
    setup_orientation_grid,
)
from radpair.summation import _compute_chunk_size, gaussian_summation


def _run_pipeline(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> tuple:
    """Run stages 1–7 and flatten/interpolate for Gaussian summation.

    Returns
    -------
    tuple
        ``(fields, intensities, widths, weights, field_axis, max_chunk_mb)``
        ready to pass to :func:`~radpair.summation.gaussian_summation`.
    """
    Sys, freq_mw, _ = prepare_spinsystem(spinsystem, experiment.freq_mw, experiment.B_z)

    theta, phi, theta_fine, phi_fine, weights, interp_mode = setup_orientation_grid(
        simopt.knots, simopt.refinement
    )

    all_tensors, frame_angles = build_tensors(Sys)

    g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)

    A_1, A_2, spec_weights = compute_hyperfine_combinations(Sys, a_projections)

    res_fields, delta_omega, quantum_beat, widths = compute_resonance_fields(
        Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
    )

    intensities = compute_intensities(delta_omega, quantum_beat)

    # Flatten to (n_orient, n_comb * 4)
    fields_mT = res_fields / _GAMMA_E_REF * 1e3
    shp = (theta.size, fields_mT.shape[1] * fields_mT.shape[2])
    fields_flat = fields_mT.reshape(shp)
    widths_flat = widths.reshape(shp)
    intensities_flat = intensities.reshape(shp)

    if interp_mode:
        transitions = np.zeros((*fields_flat.shape, 2))
        data = (fields_flat, intensities_flat, widths_flat, transitions)
        interp_ = interp.Interpolator(theta, phi, data)
        fields_flat = interp_.get_positions(theta_fine, phi_fine)
        intensities_flat = interp_.get_intensities(theta_fine, phi_fine)
        widths_flat = interp_.get_widths(theta_fine, phi_fine)

    spec_weights_arr = np.repeat(np.array([spec_weights]), 4)
    spec_weights_arr = spec_weights_arr.reshape((1, spec_weights_arr.size))

    return (
        fields_flat,
        intensities_flat * spec_weights_arr,
        spinsystem.width_gauss * widths_flat,
        weights,
        experiment.B_z,
        simopt.max_chunk_mb,
    )


def do_simulation(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> np.ndarray:
    """Simulate a cw-EPR spectrum for a spin-correlated radical pair.

    Solves the spin Hamiltonian analytically using a pseudo-secular
    approximation for the hyperfine couplings.  Supports an arbitrary
    number of anisotropic nuclei groups, each assignable to the donor or
    acceptor radical.  Zero-field splitting (*D*, *E*) and exchange
    interaction (*J*) are included.

    The simulation is composed of eight stages:

    1. :func:`~radpair.pipeline.prepare_spinsystem` — unit conversion
    2. :func:`~radpair.pipeline.setup_orientation_grid` — grid setup
    3. :func:`~radpair.pipeline.build_tensors` — diagonal tensors
    4. :func:`~radpair.pipeline.rotate_tensors` — frame + orientation rotation
    5. :func:`~radpair.hyperfine.compute_hyperfine_combinations` — hyperfine sums
    6. :func:`~radpair.hamiltonian.compute_resonance_fields` — resonance fields
    7. :func:`~radpair.hamiltonian.compute_intensities` — line intensities
    8. :func:`~radpair.summation.gaussian_summation` — Gaussian line-shape summation

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
    data = _run_pipeline(spinsystem, experiment, simopt)
    intensity = gaussian_summation(*data)
    return np.nan_to_num(intensity)


def do_simulation_multicore(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> np.ndarray:
    """Simulate a cw-EPR spectrum using multiple CPU cores.

    Runs the analytic pipeline (stages 1–7) on the main process, then
    distributes the Gaussian summation across ``simopt.cpu_cores``
    worker processes.  Each worker handles a chunk of peaks, keeping
    per-process memory bounded by ``simopt.max_chunk_mb``.

    Parameters
    ----------
    spinsystem : Spinsystem
        Spin-system object (see :class:`~radpair._types.Spinsystem`).
    experiment : Experiment
        Experiment object (see :class:`~radpair._types.Experiment`).
    simopt : SimulationOptions
        Simulation options (see :class:`~radpair._types.SimulationOptions`).
        ``simopt.cpu_cores`` controls the number of worker processes
        (``0`` = auto-detect).  ``simopt.max_chunk_mb`` controls the
        per-chunk memory limit.

    Returns
    -------
    np.ndarray
        Real-valued intensity array of the simulated spectrum, matching
        the shape of ``experiment.B_z``.
    """
    from multiprocessing import Pool, cpu_count

    n_cores = simopt.cpu_cores if simopt.cpu_cores > 0 else cpu_count()

    fields, intensities, widths, weights, field_axis, max_chunk_mb = _run_pipeline(
        spinsystem, experiment, simopt
    )

    n_orient, n_peaks = fields.shape
    total_peaks = n_orient * n_peaks

    chunk_size = _compute_chunk_size(total_peaks, field_axis.shape[0], max_chunk_mb)
    chunk_orient = max(1, chunk_size // n_peaks)

    if n_cores > 1:
        max_orient_for_parallelism = max(1, n_orient // n_cores)
        chunk_orient = min(chunk_orient, max_orient_for_parallelism)

    weights_arr = (
        np.broadcast_to(weights, (1, n_orient)) if weights.ndim == 1 else weights
    )

    args_list = []
    for start in range(0, n_orient, chunk_orient):
        end = min(start + chunk_orient, n_orient)
        args_list.append(
            (
                fields[start:end],
                intensities[start:end],
                widths[start:end],
                weights_arr[:, start:end],
                field_axis,
                0,
            )
        )

    n_chunks = len(args_list)

    if n_cores <= 1 or n_chunks <= 1:
        spectrum = np.zeros(field_axis.shape[0], dtype=np.float32)
        for args in args_list:
            spectrum += gaussian_summation(*args)
    else:
        with Pool(processes=min(n_cores, n_chunks)) as pool:
            results = pool.starmap(gaussian_summation, args_list)
        spectrum = np.sum(results, axis=0).astype(np.float32)

    return np.nan_to_num(spectrum)
