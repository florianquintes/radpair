"""Spinsystem definitions for the radpair example scripts.

This module defines seven radical-pair spin systems (S1–S7) that cover
increasing levels of complexity, from a bare radical pair with no nuclei
to a full five-nuclei system.  The same definitions are used by the test
suite (see ``tests/generate_reference_spectra.py``) and the documentation
plots, ensuring consistency.

All systems share a common X-band experiment (9.75 GHz, 344–350 mT)
and simulation options (12 grid points, no interpolation, single core).
"""

from types import SimpleNamespace

import numpy as np

FREQ_MW = 9.75e9  # Hz, X-band EPR
N_POINTS = 500
FIELD_MIN = 344.0  # mT
FIELD_MAX = 350.0  # mT
GRID_POINTS = 12
REFINEMENT = 1
CPU_CORES = 1
LINEWIDTH = 0.05  # mT

FIELD_AXIS = np.linspace(FIELD_MIN, FIELD_MAX, N_POINTS)


def make_experiment() -> SimpleNamespace:
    """Return the standard X-band experiment used by all examples."""
    return SimpleNamespace(
        B_z=FIELD_AXIS.copy(),
        freq_mw=FREQ_MW,
        magnetic_field=FIELD_AXIS.copy(),
    )


def make_simopt() -> SimpleNamespace:
    """Return the standard simulation options used by all examples."""
    return SimpleNamespace(
        grid_points=GRID_POINTS,
        refinement=REFINEMENT,
        cpu_cores=CPU_CORES,
    )


def _zero_A() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])


def _zero_frame() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])


def make_S1() -> SimpleNamespace:
    """S1 — bare radical pair, 0 nuclei groups."""
    return SimpleNamespace(
        g1=np.array([2.0023, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A1=_zero_A(),
        A2=_zero_A(),
        A3=_zero_A(),
        A4=_zero_A(),
        A5=_zero_A(),
        D=8.0,
        E=1.5,
        J_ex=2.0,
        width_gauss=LINEWIDTH,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=_zero_frame(),
        A2_frame=_zero_frame(),
        A3_frame=_zero_frame(),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
        n1=0,
        I1=0.0,
        n2=0,
        I2=0.0,
        n3=0,
        I3=0.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[],
        acceptor_list=[],
    )


def make_S2() -> SimpleNamespace:
    """S2 — 1 donor nucleus (¹H), isotropic baseline."""
    return SimpleNamespace(
        g1=np.array([2.0030, 2.0030, 2.0030]),
        g2=np.array([2.0090, 2.0090, 2.0090]),
        A1=np.array([1.5, 1.5, 1.5]),
        A2=_zero_A(),
        A3=_zero_A(),
        A4=_zero_A(),
        A5=_zero_A(),
        D=0.0,
        E=0.0,
        J_ex=0.1,
        width_gauss=LINEWIDTH,
        g1_frame=_zero_frame(),
        g2_frame=_zero_frame(),
        D_frame=_zero_frame(),
        A1_frame=_zero_frame(),
        A2_frame=_zero_frame(),
        A3_frame=_zero_frame(),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
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


def make_S3() -> SimpleNamespace:
    """S3 — 2 nuclei (1 donor ¹H + 1 acceptor 2×¹⁴N), anisotropic."""
    return SimpleNamespace(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A1=np.array([5.0, 3.0, 4.0]),
        A2=np.array([2.5, 1.8, 3.2]),
        A3=_zero_A(),
        A4=_zero_A(),
        A5=_zero_A(),
        D=8.0,
        E=1.5,
        J_ex=3.0,
        width_gauss=LINEWIDTH,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=np.array([0.0, 0.1, 0.0]),
        A2_frame=np.array([0.1, 0.0, 0.0]),
        A3_frame=_zero_frame(),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
        n1=1,
        I1=0.5,
        n2=2,
        I2=1.0,
        n3=0,
        I3=0.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[1],
        acceptor_list=[2],
    )


def make_S4() -> SimpleNamespace:
    """S4 — Swap of S3 (1 acceptor ¹H + 1 donor 2×¹⁴N)."""
    sys = make_S3()
    sys.donor_list = [2]
    sys.acceptor_list = [1]
    return sys


def make_S5() -> SimpleNamespace:
    """S5 — 3 nuclei (2 donor + 1 acceptor), includes n=3 methyl group."""
    return SimpleNamespace(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A1=np.array([4.5, 3.0, 5.5]),
        A2=np.array([1.2, 1.5, 0.8]),
        A3=np.array([2.5, 1.8, 3.2]),
        A4=_zero_A(),
        A5=_zero_A(),
        D=10.0,
        E=2.0,
        J_ex=5.0,
        width_gauss=LINEWIDTH,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=np.array([0.1, 0.0, 0.0]),
        A2_frame=np.array([0.0, 0.1, 0.0]),
        A3_frame=np.array([0.1, 0.1, 0.0]),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
        n1=3,
        I1=0.5,
        n2=1,
        I2=0.5,
        n3=2,
        I3=1.0,
        n4=0,
        I4=0.0,
        n5=0,
        I5=0.0,
        donor_list=[1, 2],
        acceptor_list=[3],
    )


def make_S6() -> SimpleNamespace:
    """S6 — 4 nuclei (1 donor + 3 acceptor), iso g1 + aniso g2, I=3/2 (³⁵Cl)."""
    return SimpleNamespace(
        g1=np.array([2.0030, 2.0030, 2.0030]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A1=np.array([1.5, 1.5, 1.5]),
        A2=np.array([3.0, 2.0, 4.0]),
        A3=np.array([8.0, 5.0, 10.0]),
        A4=np.array([0.8, 1.2, 0.6]),
        A5=_zero_A(),
        D=6.0,
        E=1.0,
        J_ex=1.5,
        width_gauss=LINEWIDTH,
        g1_frame=_zero_frame(),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=_zero_frame(),
        A2_frame=np.array([0.1, 0.0, 0.0]),
        A3_frame=np.array([0.0, 0.2, 0.0]),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
        n1=1,
        I1=0.5,
        n2=1,
        I2=1.0,
        n3=1,
        I3=1.5,
        n4=2,
        I4=0.5,
        n5=0,
        I5=0.0,
        donor_list=[1],
        acceptor_list=[2, 3, 4],
    )


def make_S7() -> SimpleNamespace:
    """S7 — 5 nuclei (3 donor + 2 acceptor), maximum complexity."""
    return SimpleNamespace(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A1=np.array([4.5, 3.0, 5.5]),
        A2=np.array([2.0, 1.5, 2.8]),
        A3=np.array([6.0, 4.0, 8.0]),
        A4=np.array([1.5, 1.0, 2.0]),
        A5=np.array([3.0, 2.0, 4.0]),
        D=10.0,
        E=2.0,
        J_ex=5.0,
        width_gauss=LINEWIDTH,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A1_frame=np.array([0.1, 0.0, 0.0]),
        A2_frame=np.array([0.0, 0.1, 0.0]),
        A3_frame=np.array([0.1, 0.1, 0.0]),
        A4_frame=_zero_frame(),
        A5_frame=_zero_frame(),
        n1=1,
        I1=0.5,
        n2=2,
        I2=1.0,
        n3=1,
        I3=1.5,
        n4=1,
        I4=0.5,
        n5=2,
        I5=0.5,
        donor_list=[1, 2, 3],
        acceptor_list=[4, 5],
    )


SYSTEMS: dict[str, type[SimpleNamespace]] = {
    "S1": make_S1,
    "S2": make_S2,
    "S3": make_S3,
    "S4": make_S4,
    "S5": make_S5,
    "S6": make_S6,
    "S7": make_S7,
}

DESCRIPTIONS: dict[str, str] = {
    "S1": "Bare radical pair (0 nuclei, anisotropic g, D/E/J)",
    "S2": "Single donor ¹H, isotropic g and A",
    "S3": "2 nuclei: donor ¹H + acceptor 2×¹⁴N, anisotropic",
    "S4": "Swap of S3: acceptor ¹H + donor 2×¹⁴N",
    "S5": "3 nuclei: methyl (3×¹H) + ¹H + 2×¹⁴N",
    "S6": "4 nuclei: ¹H + ¹⁴N + ³⁵Cl + 2×¹H, mixed iso/aniso g",
    "S7": "5 nuclei: all groups active, maximum complexity",
}
