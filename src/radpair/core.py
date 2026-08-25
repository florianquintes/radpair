"""Core simulation routines for cw-EPR spectra of radical pairs.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import numpy as np

import radpair.functions as fun
from radpair._types import Experiment, SimulationOptions, Spinsystem


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
        simopt.knots, simopt.refinement
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
        max_chunk_mb=simopt.max_chunk_mb,
    )

    return intensity


def _run_pipeline_through_intensities(
    spinsystem: Spinsystem,
    experiment: Experiment,
    simopt: SimulationOptions,
) -> tuple:
    """Run stages 1–7 and return all data needed for Gaussian summation."""
    Sys, freq_mw, _ = fun.prepare_spinsystem(
        spinsystem, experiment.freq_mw, experiment.B_z
    )
    theta, phi, theta_fine, phi_fine, weights, interp_mode = fun.setup_orientation_grid(
        simopt.knots, simopt.refinement
    )
    all_tensors, frame_angles = fun.build_tensors(Sys)
    g1, g2, D, a_projections = fun.rotate_tensors(all_tensors, frame_angles, theta, phi)
    A_1, A_2, spec_weights = fun.compute_hyperfine_combinations(Sys, a_projections)
    res_fields, delta_omega, quantum_beat, widths = fun.compute_resonance_fields(
        Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
    )
    intensities = fun.compute_intensities(delta_omega, quantum_beat)

    fields_mT = res_fields / fun._GAMMA_E_REF * 1e3
    shp = (theta.size, fields_mT.shape[1] * fields_mT.shape[2])
    fields_flat = fields_mT.reshape(shp)
    widths_flat = widths.reshape(shp)
    intensities_flat = intensities.reshape(shp)

    if interp_mode:
        transitions = np.zeros((*fields_flat.shape, 2))
        data = (fields_flat, intensities_flat, widths_flat, transitions)
        interp_ = fun.interp.Interpolator(theta, phi, data)
        fields_flat = interp_.get_positions(theta_fine, phi_fine)
        intensities_flat = interp_.get_intensities(theta_fine, phi_fine)
        widths_flat = interp_.get_widths(theta_fine, phi_fine)
        weights = fun.grid.Grid(knots=simopt.knots * simopt.refinement).get_areas()[
            np.newaxis, :
        ]

    spec_weights_arr = np.repeat(np.array([spec_weights]), 4)
    spec_weights_arr = spec_weights_arr.reshape((1, spec_weights_arr.size))

    return (
        fields_flat,
        intensities_flat * spec_weights_arr,
        widths_flat,
        weights,
        spinsystem.width_gauss,
        experiment.B_z,
        simopt.max_chunk_mb,
    )


def _gaussian_summation_worker(args: tuple) -> np.ndarray:
    """Worker function for chunked parallel Gaussian summation.

    Parameters
    ----------
    args : tuple
        ``(fields_chunk, intensities_chunk, widths_chunk, weights_chunk,
        width_gauss, field_axis)`` where the first three are 2D arrays
        of shape ``(n_orient_chunk, n_peaks)``.

    Returns
    -------
    np.ndarray
        Partial spectrum, shape ``(n_field,)``.
    """
    from eprbase import spectra as _spectra

    (
        fields_chunk,
        intensities_chunk,
        widths_chunk,
        weights_chunk,
        width_gauss,
        field_axis,
    ) = args

    n_orient, n_peaks = fields_chunk.shape

    chunk_fields = [fields_chunk[i] for i in range(n_orient)]
    chunk_intensities = [intensities_chunk[i] for i in range(n_orient)]
    chunk_widths = [(width_gauss * widths_chunk[i]) for i in range(n_orient)]
    chunk_transitions = [np.zeros((n_peaks, 2)) for _ in range(n_orient)]

    spec = _spectra.Spectra(
        chunk_fields,
        chunk_intensities,
        chunk_widths,
        chunk_transitions,
        weights=weights_chunk,
    )
    return spec.by_summation(field_axis)


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

    (
        fields_flat,
        intensities_flat,
        widths_flat,
        weights,
        width_gauss,
        field_axis,
        max_chunk_mb,
    ) = _run_pipeline_through_intensities(spinsystem, experiment, simopt)

    n_orient, n_peaks = fields_flat.shape
    total_peaks = n_orient * n_peaks

    chunk_size = fun._compute_chunk_size(total_peaks, field_axis.shape[0], max_chunk_mb)
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
                fields_flat[start:end],
                intensities_flat[start:end],
                widths_flat[start:end],
                weights_arr[:, start:end],
                width_gauss,
                field_axis,
            )
        )

    n_chunks = len(args_list)

    if n_cores <= 1 or n_chunks <= 1:
        spectrum = np.zeros(field_axis.shape[0], dtype=np.float32)
        for args in args_list:
            spectrum += _gaussian_summation_worker(args)
    else:
        with Pool(processes=min(n_cores, n_chunks)) as pool:
            results = pool.map(_gaussian_summation_worker, args_list)
        spectrum = np.sum(results, axis=0).astype(np.float32)

    return np.nan_to_num(spectrum)
