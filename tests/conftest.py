"""Shared pytest fixtures for the radpair test suite.

The fixtures construct ``Spinsystem``, ``Experiment``, and
``SimulationOptions`` objects using the typed dataclasses from
:mod:`radpair._types`.

Unit conventions (verified against the source code):

* ``Experiment.B_z``            — magnetic field axis in **milliTesla**
* ``Experiment.freq_mw``        — microwave frequency in **Hz**
* ``Spinsystem.width_gauss``    — linewidth in **milliTesla** (despite the name)
* ``Spinsystem.A_tensors``      — list of hyperfine couplings in **MHz** (3-element arrays)
* ``Spinsystem.nuclei_n``       — list of equivalent-nuclei counts (int, >= 0)
* ``Spinsystem.nuclei_I``       — list of nuclear spins (float, multiple of 0.5, >= 0)
* ``Spinsystem.A_frames``       — list of Euler angles in **radians** (3-element: [alpha, beta, gamma])
* ``Spinsystem.D``, ``Spinsystem.E``   — ZFS parameters in **MHz**
* ``Spinsystem.J_ex``           — exchange interaction in **MHz**
* ``Spinsystem.g1``, ``Spinsystem.g2`` — g-tensor diagonal (dimensionless, all positive)
* ``Spinsystem.g1_frame``, ``Spinsystem.g2_frame``, ``Spinsystem.D_frame`` — Euler angles in **radians**
* ``Spinsystem.donor_list``     — 0-indexed list of nuclei group positions on the donor radical
* ``Spinsystem.acceptor_list``  — 0-indexed list of nuclei group positions on the acceptor radical
* ``SimulationOptions.knots`` — number of orientation-grid knots
* ``SimulationOptions.refinement``  — interpolation factor (1 = no interpolation)
* ``SimulationOptions.cpu_cores``   — worker processes for multicore (0 = auto-detect)
"""

import multiprocessing as mp

import numpy as np
import pytest

from radpair._types import Experiment, SimulationOptions, Spinsystem

# Use "forkserver" instead of the default "fork" to avoid
# DeprecationWarning when forking a multi-threaded process (NumPy/SciPy).
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

# ---------------------------------------------------------------------------
# Spinsystem fixtures
# ---------------------------------------------------------------------------

_ZERO_A = np.array([0.0, 0.0, 0.0])
_ZERO_FRAME = np.array([0.0, 0.0, 0.0])
_FIVE_ZEROS_A = [_ZERO_A, _ZERO_A, _ZERO_A, _ZERO_A, _ZERO_A]
_FIVE_ZEROS_FRAME = [_ZERO_FRAME, _ZERO_FRAME, _ZERO_FRAME, _ZERO_FRAME, _ZERO_FRAME]
_FIVE_ZEROS_N = [0, 0, 0, 0, 0]
_FIVE_ZEROS_I = [0.0, 0.0, 0.0, 0.0, 0.0]


def _make_minimal_spinsystem() -> Spinsystem:
    """Return a minimal spinsystem with one donor nucleus, no ZFS or exchange.

    Isotropic g-tensors and hyperfine coupling; all Euler frames at zero
    (no rotation from the lab frame).
    """
    return Spinsystem(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A_tensors=[
            np.array([1.5, 1.5, 1.5]),
            _ZERO_A,
            _ZERO_A,
            _ZERO_A,
            _ZERO_A,
        ],
        nuclei_n=[1, 0, 0, 0, 0],
        nuclei_I=[0.5, 0.0, 0.0, 0.0, 0.0],
        A_frames=list(_FIVE_ZEROS_FRAME),
        width_gauss=0.5,
        donor_list=[0],
        acceptor_list=[],
    )


def _make_full_spinsystem() -> Spinsystem:
    """Return a spinsystem exercising all five nuclei groups and anisotropy.

    Three active nuclei groups (two donor, one acceptor) with anisotropic
    g-tensors, nonzero ZFS (D, E), and exchange interaction (J_ex).  Some
    Euler frames are nonzero to test tensor rotation.
    """
    return Spinsystem(
        g1=np.array([2.002, 2.006, 2.010]),
        g2=np.array([2.005, 2.003, 2.001]),
        A_tensors=[
            np.array([5.0, 5.0, 5.0]),
            np.array([3.0, 3.0, 3.0]),
            np.array([2.0, 2.0, 2.0]),
            _ZERO_A,
            _ZERO_A,
        ],
        nuclei_n=[1, 1, 1, 0, 0],
        nuclei_I=[0.5, 0.5, 1.0, 0.0, 0.0],
        A_frames=[
            _ZERO_FRAME,
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.0]),
            _ZERO_FRAME,
            _ZERO_FRAME,
        ],
        width_gauss=0.5,
        D=10.0,
        E=2.0,
        J_ex=5.0,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        donor_list=[0, 2],
        acceptor_list=[1],
    )


def _make_donor_only_spinsystem() -> Spinsystem:
    """Return a spinsystem with only donor nuclei (two groups)."""
    return Spinsystem(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A_tensors=[
            np.array([3.0, 3.0, 3.0]),
            np.array([1.5, 1.5, 1.5]),
            _ZERO_A,
            _ZERO_A,
            _ZERO_A,
        ],
        nuclei_n=[2, 1, 0, 0, 0],
        nuclei_I=[0.5, 0.5, 0.0, 0.0, 0.0],
        A_frames=list(_FIVE_ZEROS_FRAME),
        width_gauss=0.3,
        D=5.0,
        E=1.0,
        J_ex=2.0,
        donor_list=[0, 1],
        acceptor_list=[],
    )


def _make_acceptor_only_spinsystem() -> Spinsystem:
    """Return a spinsystem with only acceptor nuclei (two groups)."""
    return Spinsystem(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A_tensors=[
            np.array([3.0, 3.0, 3.0]),
            np.array([1.5, 1.5, 1.5]),
            _ZERO_A,
            _ZERO_A,
            _ZERO_A,
        ],
        nuclei_n=[2, 1, 0, 0, 0],
        nuclei_I=[0.5, 0.5, 0.0, 0.0, 0.0],
        A_frames=list(_FIVE_ZEROS_FRAME),
        width_gauss=0.3,
        D=5.0,
        E=1.0,
        J_ex=2.0,
        donor_list=[],
        acceptor_list=[0, 1],
    )


@pytest.fixture
def minimal_spinsystem() -> Spinsystem:
    """Minimal spinsystem: 1 donor nucleus, isotropic, no ZFS/exchange."""
    return _make_minimal_spinsystem()


@pytest.fixture
def full_spinsystem() -> Spinsystem:
    """Full spinsystem: 3 active nuclei groups, anisotropic, nonzero D/E/J."""
    return _make_full_spinsystem()


@pytest.fixture
def donor_only_spinsystem() -> Spinsystem:
    """Spinsystem with only donor nuclei (2 groups)."""
    return _make_donor_only_spinsystem()


@pytest.fixture
def acceptor_only_spinsystem() -> Spinsystem:
    """Spinsystem with only acceptor nuclei (2 groups)."""
    return _make_acceptor_only_spinsystem()


# ---------------------------------------------------------------------------
# Experiment fixtures
# ---------------------------------------------------------------------------


def _make_experiment(
    n_points: int = 500, field_min: float = 320.0, field_max: float = 370.0
) -> Experiment:
    """Return a standard X-band experiment (9.5 GHz, field sweep in mT)."""
    field_axis = np.linspace(field_min, field_max, n_points)
    return Experiment(B_z=field_axis, freq_mw=9.5e9)


@pytest.fixture
def experiment() -> Experiment:
    """X-band experiment: 500 points, 320–370 mT, 9.5 GHz."""
    return _make_experiment()


# ---------------------------------------------------------------------------
# SimOpt fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simopt_basic() -> SimulationOptions:
    """Basic simulation options: 10 grid knots, no interpolation, single core."""
    return SimulationOptions(knots=10, refinement=1, cpu_cores=1)


@pytest.fixture
def simopt_multicore() -> SimulationOptions:
    """Multicore simulation options: 10 grid knots, 2 CPU cores."""
    return SimulationOptions(knots=10, refinement=1, cpu_cores=2)


@pytest.fixture
def simopt_auto_cores() -> SimulationOptions:
    """Auto-detect CPU cores: 10 grid knots, ``cpu_cores=0``."""
    return SimulationOptions(knots=10, refinement=1, cpu_cores=0)


@pytest.fixture
def simopt_interpolation() -> SimulationOptions:
    """Interpolation mode: 5 grid knots, refinement factor 3, single core."""
    return SimulationOptions(knots=5, refinement=3, cpu_cores=1)
