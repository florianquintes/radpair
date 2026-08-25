"""Example: five nuclei, maximum complexity (S7).

A complex system with all five nuclei groups active (3 donor + 2
acceptor), anisotropic g-tensors, full ZFS (D = 10, E = 2 MHz),
exchange (J = 5 MHz), and multiple nuclear spins (I = ½, 1, 3/2) and
multiplicities (n = 1, 2).

Run::

    uv run python examples/full_five_nuclei.py
"""

import os

import matplotlib.pyplot as plt
from _systems import FIELD_AXIS, make_experiment, make_S7, make_simopt

from radpair.core import do_simulation


def main() -> None:
    sys = make_S7()
    exp = make_experiment()
    simopt = make_simopt()

    intensity = do_simulation(sys, exp, simopt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(FIELD_AXIS, intensity, lw=0.8, color="C3")
    ax.set_xlabel("Magnetic field $B_z$ (mT)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_title("S7 — 5 nuclei (3 donor + 2 acceptor), maximum complexity")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "s7_full_five_nuclei.png")
    fig.savefig(out, dpi=150)
    print(f"Spectrum saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
