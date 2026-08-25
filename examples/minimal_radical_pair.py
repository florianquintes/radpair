"""Example: minimal radical pair with no nuclei (S1).

Demonstrates the simplest anisotropic radical pair — two radicals with
anisotropic g-tensors, nonzero ZFS (D, E) and exchange (J_ex), but no
resolved hyperfine couplings.

Run::

    uv run python examples/minimal_radical_pair.py
"""

import os

import matplotlib.pyplot as plt
from _systems import FIELD_AXIS, make_experiment, make_S1, make_simopt

from radpair.core import do_simulation


def main() -> None:
    sys = make_S1()
    exp = make_experiment()
    simopt = make_simopt()

    intensity = do_simulation(sys, exp, simopt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(FIELD_AXIS, intensity, lw=0.8, color="C0")
    ax.set_xlabel("Magnetic field $B_z$ (mT)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_title("S1 — Bare radical pair (0 nuclei)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "s1_bare_radical_pair.png")
    fig.savefig(out, dpi=150)
    print(f"Spectrum saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
