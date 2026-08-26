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

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples._systems import SYSTEMS, make_experiment, make_simopt
from radpair.core import do_simulation


def _spinsystem_to_metadata(sys) -> dict[str, object]:
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


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(out_dir, exist_ok=True)

    exp = make_experiment()
    simopt = make_simopt()

    for name, make_sys in sorted(SYSTEMS.items()):
        sys = make_sys()
        intensity = do_simulation(sys, exp, simopt)

        meta = _spinsystem_to_metadata(sys)
        meta["B_z"] = exp.B_z
        meta["freq_mw"] = exp.freq_mw
        meta["grid_knots"] = simopt.knots
        meta["refinement"] = simopt.refinement
        meta["cpu_cores"] = simopt.cpu_cores
        meta["intensity"] = intensity

        out_path = os.path.join(out_dir, f"{name}.npz")
        np.savez(out_path, **meta)
        print(
            f"{name}: {intensity.shape}, min={intensity.min():.6e}, "
            f"max={intensity.max():.6e}, sum={intensity.sum():.6e} -> {out_path}"
        )

    print(f"\nAll {len(SYSTEMS)} spectra saved to {out_dir}/")


if __name__ == "__main__":
    main()
