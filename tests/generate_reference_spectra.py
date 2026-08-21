"""Generate reference cw-EPR spectra for the radpair test suite.

Produces 7 physically plausible spectra covering 0–5 nuclei groups,
isotropic and anisotropic g/A tensors, donor/acceptor swaps, and
multiple nuclear spins (I = 1/2, 1, 3/2) and multiplicities (n = 1–3).

Each spectrum is saved as an .npz file in tests/reference_data/
containing the field axis, intensity array, and all input parameters
as metadata.

Run::

    uv run python tests/generate_reference_spectra.py
"""

from types import SimpleNamespace

import numpy as np

from radpair.core import do_simulation

# ---------------------------------------------------------------------------
# Common experiment and simulation options
# ---------------------------------------------------------------------------

FREQ_MW = 9.75e9  # Hz, X-band EPR
N_POINTS = 500
FIELD_MIN = 344.0  # mT
FIELD_MAX = 350.0  # mT
GRID_POINTS = 12
REFINEMENT = 1
CPU_CORES = 1

FIELD_AXIS = np.linspace(FIELD_MIN, FIELD_MAX, N_POINTS)


def _make_experiment() -> SimpleNamespace:
    return SimpleNamespace(
        B_z=FIELD_AXIS.copy(),
        freq_mw=FREQ_MW,
        magnetic_field=FIELD_AXIS.copy(),
    )


def _make_simopt() -> SimpleNamespace:
    return SimpleNamespace(
        grid_points=GRID_POINTS,
        refinement=REFINEMENT,
        cpu_cores=CPU_CORES,
    )


def _zero_A() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])


def _zero_frame() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Spinsystem definitions
# ---------------------------------------------------------------------------


def _make_S1() -> SimpleNamespace:
    """S1 — 0 nuclei groups (bare radical pair)."""
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
        width_gauss=0.05,
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


def _make_S2() -> SimpleNamespace:
    """S2 — 1 nucleus (donor 1H), isotropic baseline."""
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
        width_gauss=0.05,
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


def _make_S3() -> SimpleNamespace:
    """S3 — 2 nuclei (1 donor 1H + 1 acceptor 2x14N), anisotropic."""
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
        width_gauss=0.05,
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


def _make_S4() -> SimpleNamespace:
    """S4 — Swap of S3 (1 acceptor 1H + 1 donor 2x14N)."""
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
        width_gauss=0.05,
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
        donor_list=[2],
        acceptor_list=[1],
    )


def _make_S5() -> SimpleNamespace:
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
        width_gauss=0.05,
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


def _make_S6() -> SimpleNamespace:
    """S6 — 4 nuclei (1 donor + 3 acceptor), iso g1 + aniso g2, I=3/2 (35Cl)."""
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
        width_gauss=0.05,
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


def _make_S7() -> SimpleNamespace:
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
        width_gauss=0.05,
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


SPECTRA = {
    "S1": _make_S1,
    "S2": _make_S2,
    "S3": _make_S3,
    "S4": _make_S4,
    "S5": _make_S5,
    "S6": _make_S6,
    "S7": _make_S7,
}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _spinsystem_to_metadata(sys: SimpleNamespace) -> dict[str, object]:
    """Extract all spinsystem attributes into a flat dict for npz storage."""
    meta: dict[str, object] = {}
    for key, val in vars(sys).items():
        if isinstance(val, np.ndarray):
            meta[key] = val
        elif isinstance(val, list):
            meta[key] = np.array(val)
        else:
            meta[key] = val
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(out_dir, exist_ok=True)

    exp = _make_experiment()
    simopt = _make_simopt()

    for name, make_sys in sorted(SPECTRA.items()):
        sys = make_sys()
        intensity = do_simulation(sys, exp, simopt)

        meta = _spinsystem_to_metadata(sys)
        meta["B_z"] = FIELD_AXIS
        meta["freq_mw"] = FREQ_MW
        meta["grid_points"] = GRID_POINTS
        meta["refinement"] = REFINEMENT
        meta["cpu_cores"] = CPU_CORES
        meta["intensity"] = intensity

        out_path = os.path.join(out_dir, f"{name}.npz")
        np.savez(out_path, **meta)
        print(
            f"{name}: {intensity.shape}, min={intensity.min():.6e}, "
            f"max={intensity.max():.6e}, sum={intensity.sum():.6e} -> {out_path}"
        )

    print(f"\nAll {len(SPECTRA)} spectra saved to {out_dir}/")


if __name__ == "__main__":
    main()
