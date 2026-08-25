"""Example: two anisotropic nuclei, mixed donor/acceptor (S3).

A more realistic system: the donor carries one ¹H with anisotropic
hyperfine coupling, and the acceptor carries two equivalent ¹⁴N nuclei
(I = 1).  Both g-tensors are anisotropic, and ZFS (D = 8, E = 1.5 MHz)
and exchange (J = 3 MHz) are nonzero.  Several Euler frames are nonzero,
so tensors are rotated relative to the lab frame.

Run::

    uv run python examples/anisotropic_two_nuclei.py
"""

import os

import matplotlib.pyplot as plt
from _systems import FIELD_AXIS, make_experiment, make_S3, make_simopt

from radpair.core import do_simulation


def main() -> None:
    sys = make_S3()
    exp = make_experiment()
    simopt = make_simopt()

    intensity = do_simulation(sys, exp, simopt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(FIELD_AXIS, intensity, lw=0.8, color="C2")
    ax.set_xlabel("Magnetic field $B_z$ (mT)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_title("S3 — 2 nuclei (donor ¹H + acceptor 2×¹⁴N), anisotropic")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "s3_anisotropic_two_nuclei.png")
    fig.savefig(out, dpi=150)
    print(f"Spectrum saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
