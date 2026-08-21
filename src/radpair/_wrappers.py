"""Provide decorators and multicore helpers for simulation routines.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni-freiburg.de

@author: Florian Quintes
"""

from collections.abc import Callable
from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool, cpu_count
from time import time
from typing import Any

import numpy as np

from radpair._types import Experiment, SimulationOptions, Spinsystem


def timer[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Measure the wall-clock runtime of a single function call.

    Parameters
    ----------
    func : callable
        Function whose runtime will be measured.

    Returns
    -------
    callable
        Wrapper that prints the runtime and returns the original result.
    """

    def time_wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time()
        res = func(*args, **kwargs)
        runtime = time() - start
        print(f"The runtime of {func.__name__} is {runtime:.3f} s")
        return res

    return time_wrap


def function_benchmark(
    func: Callable[..., Any], niter: int = 100
) -> Callable[..., None]:
    """Run *func* *niter* times and print best, worst, and average runtime.

    Parameters
    ----------
    func : callable
        Function which will be benchmarked.
    niter : int, optional
        Number of function calls (default is 100).

    Returns
    -------
    callable
        Wrapper that runs the benchmark and prints timing statistics.
        The wrapper does **not** return the original result.
    """

    def benchmarked_function(*args: Any, **kwargs: Any) -> None:
        times = np.empty(niter)
        for i in range(times.shape[0]):
            start = time()
            func(*args, **kwargs)
            times[i] = time() - start

        print(f"Runned a benchmark of {func.__name__}.")

        if times.min() > 1:
            unit = "s"
        elif times.min() > 1e-3:
            unit = "ms"
            times *= 1e3
        else:
            unit = "μs"
            times *= 1e6

        print(f"Average time: {times.mean():.3f} {unit}")
        print(f"Best time: {times.min():.3f} {unit}")
        print(f"Worst time: {times.max():3f} {unit}")

    return benchmarked_function


def multicore(
    simulation: Callable[..., np.ndarray],
) -> Callable[..., np.ndarray]:
    """Parallelise a simulation routine using :class:`multiprocessing.Pool`.

    The decorated function must accept ``(spinsystem, experiment, simopt)``
    and is executed on ``simopt.cpu_cores`` processes, each handling a
    slice of the magnetic-field axis.

    Parameters
    ----------
    simulation : callable
        Simulation function with the signature
        ``(spinsystem, experiment, simopt) -> np.ndarray``.

    Returns
    -------
    callable
        Wrapper with the same signature that distributes the work across
        CPU cores and returns the concatenated spectrum.
    """

    def multicore_wrapper(
        spinsystem: Spinsystem,
        experiment: Experiment,
        simopt: SimulationOptions,
    ) -> np.ndarray:
        if simopt.cpu_cores == 0:
            simopt.cpu_cores = cpu_count()

        whole_spectrum = 1 * experiment.magnetic_field
        whole_B_z = 1 * experiment.B_z
        n_field_points = whole_spectrum.shape[0]

        points_per_core = n_field_points // simopt.cpu_cores

        Exp_list = np.empty(simopt.cpu_cores, dtype=object)
        for core in range(simopt.cpu_cores):
            Experimental = deepcopy(experiment)
            start = core * points_per_core
            if core + 1 < simopt.cpu_cores:
                end = (core + 1) * points_per_core
                Experimental.magnetic_field = whole_spectrum[start:end]
                Experimental.B_z = whole_B_z[start:end]
            else:
                Experimental.magnetic_field = whole_spectrum[start:]
                Experimental.B_z = whole_B_z[start:]
            Exp_list[core] = Experimental

        pool = Pool(processes=simopt.cpu_cores)
        single_intensities = pool.starmap(
            simulation, zip(repeat(spinsystem), Exp_list, repeat(simopt))
        )
        pool.close()
        pool.join()

        intens_arr_tuple = tuple(single_intensities)
        intensity = np.hstack(intens_arr_tuple)

        return intensity

    return multicore_wrapper
