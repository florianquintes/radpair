"""Smoke tests verifying that the shared fixtures produce valid simulations."""

import numpy as np

from radpair.core import do_simulation, do_simulation_multicore

# ---------------------------------------------------------------------------
# do_simulation — single-core smoke tests
# ---------------------------------------------------------------------------


def test_minimal_simulation_runs(minimal_spinsystem, experiment, simopt_basic):
    """A minimal single-core simulation produces a valid spectrum."""
    result = do_simulation(minimal_spinsystem, experiment, simopt_basic)
    assert result.shape == experiment.B_z.shape
    assert np.all(np.isreal(result))
    assert not np.any(np.isnan(result))


def test_full_simulation_runs(full_spinsystem, experiment, simopt_basic):
    """A full single-core simulation produces a valid spectrum."""
    result = do_simulation(full_spinsystem, experiment, simopt_basic)
    assert result.shape == experiment.B_z.shape
    assert np.all(np.isreal(result))
    assert not np.any(np.isnan(result))
    assert np.any(result != 0)


def test_interpolation_simulation_runs(
    minimal_spinsystem, experiment, simopt_interpolation
):
    """The interpolation path (refinement > 1) produces a valid spectrum."""
    result = do_simulation(minimal_spinsystem, experiment, simopt_interpolation)
    assert result.shape == experiment.B_z.shape
    assert np.all(np.isreal(result))
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# do_simulation_multicore — parallel smoke tests
# ---------------------------------------------------------------------------


def test_multicore_matches_single_core(
    minimal_spinsystem, experiment, simopt_basic, simopt_multicore
):
    """Multicore output matches single-core output for the same inputs."""
    single = do_simulation(minimal_spinsystem, experiment, simopt_basic)
    multi = do_simulation_multicore(minimal_spinsystem, experiment, simopt_multicore)
    assert multi.shape == single.shape
    assert np.allclose(multi, single)


def test_multicore_auto_cores(minimal_spinsystem, experiment, simopt_auto_cores):
    """Multicore with cpu_cores=0 (auto-detect) produces a valid spectrum."""
    result = do_simulation_multicore(minimal_spinsystem, experiment, simopt_auto_cores)
    assert result.shape == experiment.B_z.shape
    assert not np.any(np.isnan(result))
