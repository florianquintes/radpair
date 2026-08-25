"""Example: single donor nucleus, isotropic system (S2).

The simplest system with a resolved hyperfine coupling: one donor proton
with isotropic g-tensors and isotropic A.  The acceptor is "silent"
(no nuclei, isotropic g).  A small exchange interaction (J = 0.1 MHz)
is present; ZFS is zero.

Run::

    uv run python examples/single_nucleus.py
"""

import os

import matplotlib.pyplot as plt
from _systems import FIELD_AXIS, make_experiment, make_S2, make_simopt

from radpair.core import do_simulation


def main() -> None:
    sys = make_S2()
    exp = make_experiment()
    simopt = make_simopt()

    intensity = do_simulation(sys, exp, simopt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(FIELD_AXIS, intensity, lw=0.8, color="C1")
    ax.set_xlabel("Magnetic field $B_z$ (mT)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_title("S2 — Single donor ¹H, isotropic")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "s2_single_nucleus.png")
    fig.savefig(out, dpi=150)
    print(f"Spectrum saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
