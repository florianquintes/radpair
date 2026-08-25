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

import numpy as np

from radpair._types import Experiment, SimulationOptions, Spinsystem
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


def _make_experiment() -> Experiment:
    return Experiment(
        B_z=FIELD_AXIS.copy(),
        freq_mw=FREQ_MW,
    )


def _make_simopt() -> SimulationOptions:
    return SimulationOptions(
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


def _make_S1() -> Spinsystem:
    """S1 — 0 nuclei groups (bare radical pair)."""
    return Spinsystem(
        g1=np.array([2.0023, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[_zero_A(), _zero_A(), _zero_A(), _zero_A(), _zero_A()],
        nuclei_n=[0, 0, 0, 0, 0],
        nuclei_I=[0.0, 0.0, 0.0, 0.0, 0.0],
        D=8.0,
        E=1.5,
        J_ex=2.0,
        width_gauss=0.05,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[],
        acceptor_list=[],
    )


def _make_S2() -> Spinsystem:
    """S2 — 1 nucleus (donor 1H), isotropic baseline."""
    return Spinsystem(
        g1=np.array([2.0030, 2.0030, 2.0030]),
        g2=np.array([2.0090, 2.0090, 2.0090]),
        A_tensors=[
            np.array([1.5, 1.5, 1.5]),
            _zero_A(),
            _zero_A(),
            _zero_A(),
            _zero_A(),
        ],
        nuclei_n=[1, 0, 0, 0, 0],
        nuclei_I=[0.5, 0.0, 0.0, 0.0, 0.0],
        D=0.0,
        E=0.0,
        J_ex=0.1,
        width_gauss=0.05,
        g1_frame=_zero_frame(),
        g2_frame=_zero_frame(),
        D_frame=_zero_frame(),
        A_frames=[
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[0],
        acceptor_list=[],
    )


def _make_S3() -> Spinsystem:
    """S3 — 2 nuclei (1 donor 1H + 1 acceptor 2x14N), anisotropic."""
    return Spinsystem(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[
            np.array([5.0, 3.0, 4.0]),
            np.array([2.5, 1.8, 3.2]),
            _zero_A(),
            _zero_A(),
            _zero_A(),
        ],
        nuclei_n=[1, 2, 0, 0, 0],
        nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
        D=8.0,
        E=1.5,
        J_ex=3.0,
        width_gauss=0.05,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            np.array([0.0, 0.1, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[0],
        acceptor_list=[1],
    )


def _make_S4() -> Spinsystem:
    """S4 — Swap of S3 (1 acceptor 1H + 1 donor 2x14N)."""
    return Spinsystem(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[
            np.array([5.0, 3.0, 4.0]),
            np.array([2.5, 1.8, 3.2]),
            _zero_A(),
            _zero_A(),
            _zero_A(),
        ],
        nuclei_n=[1, 2, 0, 0, 0],
        nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
        D=8.0,
        E=1.5,
        J_ex=3.0,
        width_gauss=0.05,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            np.array([0.0, 0.1, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            _zero_frame(),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[1],
        acceptor_list=[0],
    )


def _make_S5() -> Spinsystem:
    """S5 — 3 nuclei (2 donor + 1 acceptor), includes n=3 methyl group."""
    return Spinsystem(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[
            np.array([4.5, 3.0, 5.5]),
            np.array([1.2, 1.5, 0.8]),
            np.array([2.5, 1.8, 3.2]),
            _zero_A(),
            _zero_A(),
        ],
        nuclei_n=[3, 1, 2, 0, 0],
        nuclei_I=[0.5, 0.5, 1.0, 0.0, 0.0],
        D=10.0,
        E=2.0,
        J_ex=5.0,
        width_gauss=0.05,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.0]),
            np.array([0.1, 0.1, 0.0]),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[0, 1],
        acceptor_list=[2],
    )


def _make_S6() -> Spinsystem:
    """S6 — 4 nuclei (1 donor + 3 acceptor), iso g1 + aniso g2, I=3/2 (35Cl)."""
    return Spinsystem(
        g1=np.array([2.0030, 2.0030, 2.0030]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[
            np.array([1.5, 1.5, 1.5]),
            np.array([3.0, 2.0, 4.0]),
            np.array([8.0, 5.0, 10.0]),
            np.array([0.8, 1.2, 0.6]),
            _zero_A(),
        ],
        nuclei_n=[1, 1, 1, 2, 0],
        nuclei_I=[0.5, 1.0, 1.5, 0.5, 0.0],
        D=6.0,
        E=1.0,
        J_ex=1.5,
        width_gauss=0.05,
        g1_frame=_zero_frame(),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            _zero_frame(),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.2, 0.0]),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[0],
        acceptor_list=[1, 2, 3],
    )


def _make_S7() -> Spinsystem:
    """S7 — 5 nuclei (3 donor + 2 acceptor), maximum complexity."""
    return Spinsystem(
        g1=np.array([2.0020, 2.0040, 2.0060]),
        g2=np.array([2.0080, 2.0100, 2.0120]),
        A_tensors=[
            np.array([4.5, 3.0, 5.5]),
            np.array([2.0, 1.5, 2.8]),
            np.array([6.0, 4.0, 8.0]),
            np.array([1.5, 1.0, 2.0]),
            np.array([3.0, 2.0, 4.0]),
        ],
        nuclei_n=[1, 2, 1, 1, 2],
        nuclei_I=[0.5, 1.0, 1.5, 0.5, 0.5],
        D=10.0,
        E=2.0,
        J_ex=5.0,
        width_gauss=0.05,
        g1_frame=np.array([0.1, 0.2, 0.0]),
        g2_frame=np.array([0.0, 0.3, 0.1]),
        D_frame=np.array([0.2, 0.1, 0.0]),
        A_frames=[
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.0]),
            np.array([0.1, 0.1, 0.0]),
            _zero_frame(),
            _zero_frame(),
        ],
        donor_list=[0, 1, 2],
        acceptor_list=[3, 4],
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


def _spinsystem_to_metadata(sys: Spinsystem) -> dict[str, object]:
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
