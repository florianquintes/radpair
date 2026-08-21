"""Shared pytest fixtures for the radpair test suite.

The fixtures construct minimal valid ``Spinsystem``, ``Exp``, and ``SimOpt``
objects using :class:`types.SimpleNamespace`, which supports the dynamic-
attribute interface (``vars(obj)``, ``getattr``, ``setattr``) that
:func:`radpair.core.do_simulation` expects.

Unit conventions (verified against the source code):

* ``Exp.B_z``            — magnetic field axis in **milliTesla**
* ``Exp.freq_mw``        — microwave frequency in **Hz**
* ``Sys.width_gauss``    — linewidth in **milliTesla** (despite the name)
* ``Sys.A1``–``Sys.A5``  — hyperfine couplings in **MHz** (3-element arrays)
* ``Sys.D``, ``Sys.E``   — ZFS parameters in **MHz**
* ``Sys.J_ex``           — exchange interaction in **MHz**
* ``Sys.g1``, ``Sys.g2`` — g-tensor diagonal (dimensionless, all positive)
* ``Sys.*_frame``        — Euler angles in **radians** (3-element: [alpha, beta, gamma])
* ``Sys.n*``             — number of equivalent nuclei (int, >= 0)
* ``Sys.I*``             — nuclear spin (float, multiple of 0.5, >= 0)
* ``Sys.donor_list``     — list of core indices assigned to the donor radical
* ``Sys.acceptor_list``  — list of core indices assigned to the acceptor radical
* ``SimOpt.grid_points`` — number of orientation-grid knots
* ``SimOpt.refinement``  — interpolation factor (1 = no interpolation)
* ``SimOpt.cpu_cores``   — worker processes for multicore (0 = auto-detect)
"""

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest

# Use "forkserver" instead of the default "fork" to avoid
# DeprecationWarning when forking a multi-threaded process (NumPy/SciPy).
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

# ---------------------------------------------------------------------------
# Spinsystem fixtures
# ---------------------------------------------------------------------------


def _make_minimal_spinsystem() -> SimpleNamespace:
    """Return a minimal spinsystem with one donor nucleus, no ZFS or exchange.

    Isotropic g-tensors and hyperfine coupling; all Euler frames at zero
    (no rotation from the lab frame).
    """
    return SimpleNamespace(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A1=np.array([1.5, 1.5, 1.5]),
        A2=np.array([0.0, 0.0, 0.0]),
        A3=np.array([0.0, 0.0, 0.0]),
        A4=np.array([0.0, 0.0, 0.0]),
        A5=np.array([0.0, 0.0, 0.0]),
        D=0.0,
        E=0.0,
        J_ex=0.0,
        width_gauss=0.5,
        g1_frame=np.array([0.0, 0.0, 0.0]),
        g2_frame=np.array([0.0, 0.0, 0.0]),
        D_frame=np.array([0.0, 0.0, 0.0]),
        A1_frame=np.array([0.0, 0.0, 0.0]),
        A2_frame=np.array([0.0, 0.0, 0.0]),
        A3_frame=np.array([0.0, 0.0, 0.0]),
        A4_frame=np.array([0.0, 0.0, 0.0]),
        A5_frame=np.array([0.0, 0.0, 0.0]),
        n1=1,
        I1=0.5,
        n2=0,
        I2=0.0,
        n3=0,
        I3=0.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[1],
        acceptor_list=[],
    )


def _make_full_spinsystem() -> SimpleNamespace:
    """Return a spinsystem exercising all five nuclei groups and anisotropy.

    Three active nuclei groups (two donor, one acceptor) with anisotropic
    g-tensors, nonzero ZFS (D, E), and exchange interaction (J_ex).  Some
    Euler frames are nonzero to test tensor rotation.
    """
    return SimpleNamespace(
        g1=np.array([2.002, 2.006, 2.010]),
        g2=np.array([2.005, 2.003, 2.001]),
        A1=np.array([5.0, 5.0, 5.0]),
        A2=np.array([3.0, 3.0, 3.0]),
        A3=np.array([2.0, 2.0, 2.0]),
        A4=np.array([0.0, 0.0, 0.0]),
        A5=np.array([0.0, 0.0, 0.0]),
        D=10.0,
        E=2.0,
        J_ex=5.0,
        width_gauss=0.5,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=np.array([0.0, 0.0, 0.0]),
        A2_frame=np.array([0.1, 0.0, 0.0]),
        A3_frame=np.array([0.0, 0.1, 0.0]),
        A4_frame=np.array([0.0, 0.0, 0.0]),
        A5_frame=np.array([0.0, 0.0, 0.0]),
        n1=1,
        I1=0.5,
        n2=1,
        I2=0.5,
        n3=1,
        I3=1.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[1, 3],
        acceptor_list=[2],
    )


def _make_donor_only_spinsystem() -> SimpleNamespace:
    """Return a spinsystem with only donor nuclei (two groups)."""
    return SimpleNamespace(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A1=np.array([3.0, 3.0, 3.0]),
        A2=np.array([1.5, 1.5, 1.5]),
        A3=np.array([0.0, 0.0, 0.0]),
        A4=np.array([0.0, 0.0, 0.0]),
        A5=np.array([0.0, 0.0, 0.0]),
        D=5.0,
        E=1.0,
        J_ex=2.0,
        width_gauss=0.3,
        g1_frame=np.array([0.0, 0.0, 0.0]),
        g2_frame=np.array([0.0, 0.0, 0.0]),
        D_frame=np.array([0.0, 0.0, 0.0]),
        A1_frame=np.array([0.0, 0.0, 0.0]),
        A2_frame=np.array([0.0, 0.0, 0.0]),
        A3_frame=np.array([0.0, 0.0, 0.0]),
        A4_frame=np.array([0.0, 0.0, 0.0]),
        A5_frame=np.array([0.0, 0.0, 0.0]),
        n1=2,
        I1=0.5,
        n2=1,
        I2=0.5,
        n3=0,
        I3=0.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[1, 2],
        acceptor_list=[],
    )


def _make_acceptor_only_spinsystem() -> SimpleNamespace:
    """Return a spinsystem with only acceptor nuclei (two groups)."""
    return SimpleNamespace(
        g1=np.array([2.003, 2.003, 2.003]),
        g2=np.array([2.007, 2.007, 2.007]),
        A1=np.array([3.0, 3.0, 3.0]),
        A2=np.array([1.5, 1.5, 1.5]),
        A3=np.array([0.0, 0.0, 0.0]),
        A4=np.array([0.0, 0.0, 0.0]),
        A5=np.array([0.0, 0.0, 0.0]),
        D=5.0,
        E=1.0,
        J_ex=2.0,
        width_gauss=0.3,
        g1_frame=np.array([0.0, 0.0, 0.0]),
        g2_frame=np.array([0.0, 0.0, 0.0]),
        D_frame=np.array([0.0, 0.0, 0.0]),
        A1_frame=np.array([0.0, 0.0, 0.0]),
        A2_frame=np.array([0.0, 0.0, 0.0]),
        A3_frame=np.array([0.0, 0.0, 0.0]),
        A4_frame=np.array([0.0, 0.0, 0.0]),
        A5_frame=np.array([0.0, 0.0, 0.0]),
        n1=2,
        I1=0.5,
        n2=1,
        I2=0.5,
        n3=0,
        I3=0.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[],
        acceptor_list=[1, 2],
    )


@pytest.fixture
def minimal_spinsystem() -> SimpleNamespace:
    """Minimal spinsystem: 1 donor nucleus, isotropic, no ZFS/exchange."""
    return _make_minimal_spinsystem()


@pytest.fixture
def full_spinsystem() -> SimpleNamespace:
    """Full spinsystem: 3 active nuclei groups, anisotropic, nonzero D/E/J."""
    return _make_full_spinsystem()


@pytest.fixture
def donor_only_spinsystem() -> SimpleNamespace:
    """Spinsystem with only donor nuclei (2 groups)."""
    return _make_donor_only_spinsystem()


@pytest.fixture
def acceptor_only_spinsystem() -> SimpleNamespace:
    """Spinsystem with only acceptor nuclei (2 groups)."""
    return _make_acceptor_only_spinsystem()


# ---------------------------------------------------------------------------
# Experiment fixtures
# ---------------------------------------------------------------------------


def _make_experiment(
    n_points: int = 500, field_min: float = 320.0, field_max: float = 370.0
) -> SimpleNamespace:
    """Return a standard X-band experiment (9.5 GHz, field sweep in mT)."""
    field_axis = np.linspace(field_min, field_max, n_points)
    return SimpleNamespace(
        B_z=field_axis,
        freq_mw=9.5e9,
        magnetic_field=field_axis.copy(),
    )


@pytest.fixture
def experiment() -> SimpleNamespace:
    """X-band experiment: 500 points, 320–370 mT, 9.5 GHz."""
    return _make_experiment()


# ---------------------------------------------------------------------------
# SimOpt fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simopt_basic() -> SimpleNamespace:
    """Basic simulation options: 10 grid knots, no interpolation, single core."""
    return SimpleNamespace(
        grid_points=10,
        refinement=1,
        cpu_cores=1,
    )


@pytest.fixture
def simopt_multicore() -> SimpleNamespace:
    """Multicore simulation options: 10 grid knots, 2 CPU cores."""
    return SimpleNamespace(
        grid_points=10,
        refinement=1,
        cpu_cores=2,
    )


@pytest.fixture
def simopt_auto_cores() -> SimpleNamespace:
    """Auto-detect CPU cores: 10 grid knots, ``cpu_cores=0``."""
    return SimpleNamespace(
        grid_points=10,
        refinement=1,
        cpu_cores=0,
    )


@pytest.fixture
def simopt_interpolation() -> SimpleNamespace:
    """Interpolation mode: 5 grid knots, refinement factor 3, single core."""
    return SimpleNamespace(
        grid_points=5,
        refinement=3,
        cpu_cores=1,
    )
