"""Load the 7 reference spectra and save a combined PDF plot.

Run::

    uv run python tests/plot_reference_spectra.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "reference_data")

SPECTRA_NAMES = [f"S{i}" for i in range(1, 8)]

TITLES = {
    "S1": "S1 — 0 nuclei (bare radical pair)",
    "S2": "S2 — 1 nucleus (donor ¹H, isotropic)",
    "S3": "S3 — 2 nuclei (donor ¹H + acceptor 2×¹⁴N)",
    "S4": "S4 — Swap of S3 (acceptor ¹H + donor 2×¹⁴N)",
    "S5": "S5 — 3 nuclei (2 donor + 1 acceptor, n=3 methyl)",
    "S6": "S6 — 4 nuclei (1 donor + 3 acceptor, I=3/2 ³⁵Cl)",
    "S7": "S7 — 5 nuclei (3 donor + 2 acceptor, max complexity)",
}


def main() -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, name in enumerate(SPECTRA_NAMES):
        path = os.path.join(DATA_DIR, f"{name}.npz")
        data = np.load(path)
        b_z = data["B_z"]
        intensity = data["intensity"]

        ax = axes_flat[idx]
        ax.plot(b_z, intensity, linewidth=0.8, color="black")
        ax.set_title(TITLES[name], fontsize=11)
        ax.set_xlabel("B_z (mT)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")

    axes_flat[-1].set_visible(False)

    combined_path = os.path.join(DATA_DIR, "all_spectra.pdf")
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Combined plot saved: {combined_path}")


if __name__ == "__main__":
    main()
