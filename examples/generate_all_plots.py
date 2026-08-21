"""Generate all 7 spectra plots and a combined overview for the docs.

This script runs ``do_simulation`` for each of the seven example spin
systems (S1–S7) and saves:

- Individual PNG files in ``docs/source/_static/``
- A combined 7-panel overview PNG

Run::

    uv run python examples/generate_all_plots.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _systems import DESCRIPTIONS, FIELD_AXIS, SYSTEMS, make_experiment, make_simopt

from radpair.core import do_simulation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_static")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    exp = make_experiment()
    simopt = make_simopt()

    spectra: dict[str, np.ndarray] = {}

    for name in sorted(SYSTEMS):
        sys = SYSTEMS[name]()
        intensity = do_simulation(sys, exp, simopt)
        spectra[name] = intensity
        print(
            f"{name}: shape={intensity.shape}, "
            f"min={intensity.min():.6e}, max={intensity.max():.6e}, "
            f"sum={intensity.sum():.6e}"
        )

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

    for name in sorted(SYSTEMS):
        idx = int(name[1]) - 1
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(FIELD_AXIS, spectra[name], lw=0.8, color=colors[idx])
        ax.set_xlabel("Magnetic field $B_z$ (mT)")
        ax.set_ylabel("Intensity (arb. u.)")
        ax.set_title(f"{name} — {DESCRIPTIONS[name]}")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"spectrum_{name.lower()}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")

    fig, axes = plt.subplots(7, 1, figsize=(9, 18), sharex=True)
    for ax, name in zip(axes, sorted(SYSTEMS)):
        idx = int(name[1]) - 1
        ax.plot(FIELD_AXIS, spectra[name], lw=0.7, color=colors[idx])
        ax.set_ylabel("Intensity", fontsize=9)
        ax.set_title(f"{name} — {DESCRIPTIONS[name]}", fontsize=10)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Magnetic field $B_z$ (mT)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "spectrum_overview.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nCombined overview -> {out}")


if __name__ == "__main__":
    main()
