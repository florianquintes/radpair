"""Example: donor/acceptor swap comparison (S3 vs S4).

The two systems S3 and S4 are identical except that the donor and
acceptor nuclei assignments are swapped.  This example demonstrates
how the same nuclei produce different spectra depending on which
radical they belong to.

Run::

    uv run python examples/donor_acceptor_swap.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _systems import FIELD_AXIS, make_experiment, make_S3, make_S4, make_simopt

from radpair.core import do_simulation


def main() -> None:
    exp = make_experiment()
    simopt = make_simopt()

    s3 = do_simulation(make_S3(), exp, simopt)
    s4 = do_simulation(make_S4(), exp, simopt)

    print(f"S3 sum: {s3.sum():.10e}")
    print(f"S4 sum: {s4.sum():.10e}")
    print(f"Max abs diff: {np.max(np.abs(s3 - s4)):.2e}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(FIELD_AXIS, s3, lw=0.8, color="C2")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("S3 — donor ¹H + acceptor 2×¹⁴N")
    axes[0].axhline(0, color="gray", lw=0.5, ls="--")

    axes[1].plot(FIELD_AXIS, s4, lw=0.8, color="C5")
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("S4 — acceptor ¹H + donor 2×¹⁴N (swap)")
    axes[1].axhline(0, color="gray", lw=0.5, ls="--")

    axes[2].plot(FIELD_AXIS, s3 - s4, lw=0.8, color="C7")
    axes[2].set_ylabel("S3 − S4")
    axes[2].set_xlabel("Magnetic field $B_z$ (mT)")
    axes[2].set_title("Difference")
    axes[2].axhline(0, color="gray", lw=0.5, ls="--")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "donor_acceptor_swap.png")
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
