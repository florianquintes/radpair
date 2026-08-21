"""Example: multi-core simulation.

Demonstrates ``do_simulation_multicore``, which splits the field axis
across multiple CPU processes via ``multiprocessing.Pool``.  The result
is numerically identical to the single-core call.

Run::

    uv run python examples/multicore.py
"""

import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from _systems import FIELD_AXIS, make_experiment, make_S3

from radpair.core import do_simulation, do_simulation_multicore


def main() -> None:
    sys = make_S3()
    exp = make_experiment()
    simopt = SimpleNamespace(grid_points=12, refinement=1, cpu_cores=4)

    intensity_sc = do_simulation(sys, exp, simopt)
    intensity_mc = do_simulation_multicore(sys, exp, simopt)

    print(f"Single-core sum: {intensity_sc.sum():.10e}")
    print(f"Multi-core  sum: {intensity_mc.sum():.10e}")
    print(f"Max abs diff:    {np.max(np.abs(intensity_sc - intensity_mc)):.2e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(FIELD_AXIS, intensity_mc, lw=0.8, color="C4", label="4 cores")
    ax.set_xlabel("Magnetic field $B_z$ (mT)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_title("S3 — Multi-core simulation (cpu_cores = 4)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "multicore.png")
    fig.savefig(out, dpi=150)
    print(f"Spectrum saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
