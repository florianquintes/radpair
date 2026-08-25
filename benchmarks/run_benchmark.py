"""Benchmark script for single-core radpair simulation performance.

Measures three performance trends:

1. **Nuclei scaling** — runtime vs. number of active nuclei groups (0–5),
   using systems S1–S7 from the example definitions.

2. **Interpolation scaling** — runtime vs. refinement factor (1, 2, 3, 4),
   using a fixed 2-nuclei system (S3).

3. **Full suite** — total wall-clock time for all 7 example spectra
   (S1–S7) with default settings.

Results are written to ``BENCHMARK.md`` in the repository root as
Markdown tables.

Usage::

    uv run python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

from _common import (
    DESCRIPTIONS,
    FIELD_AXIS,
    N_KNOTS,
    N_REPEATS,
    N_WARMUP,
    REFINEMENT_VALUES,
    SYSTEMS,
    bench,
    count_active_nuclei,
    env_info,
    format_md_table,
    make_experiment,
    make_simopt,
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "BENCHMARK.md")


def bench_nuclei_scaling() -> list[dict[str, object]]:
    """Benchmark S1–S7 (0 to 5 active nuclei groups)."""
    exp = make_experiment()
    simopt = make_simopt()
    results: list[dict[str, object]] = []

    for name in sorted(SYSTEMS):
        sys = SYSTEMS[name]()
        n_nuclei = count_active_nuclei(sys)
        stats = bench(sys, exp, simopt)
        results.append(
            {
                "system": name,
                "n_nuclei": n_nuclei,
                "description": DESCRIPTIONS[name],
                **stats,
            }
        )
        print(
            f"  {name} ({n_nuclei} nuclei): "
            f"mean={stats['mean']:.4f}s  min={stats['min']:.4f}s  "
            f"max={stats['max']:.4f}s  std={stats['std']:.4f}s"
        )

    return results


def bench_interpolation_scaling() -> list[dict[str, object]]:
    """Benchmark S3 with refinement factors 1, 2, 3, 4."""
    sys = SYSTEMS["S3"]()
    exp = make_experiment()
    results: list[dict[str, object]] = []

    for refinement in REFINEMENT_VALUES:
        simopt = SimpleNamespace(
            knots=N_KNOTS,
            refinement=refinement,
            cpu_cores=1,
        )
        stats = bench(sys, exp, simopt)
        n_orientations = N_KNOTS * refinement
        results.append(
            {
                "refinement": refinement,
                "n_orientations": n_orientations,
                **stats,
            }
        )
        print(
            f"  refinement={refinement} ({n_orientations} orientations): "
            f"mean={stats['mean']:.4f}s  min={stats['min']:.4f}s  "
            f"max={stats['max']:.4f}s  std={stats['std']:.4f}s"
        )

    return results


def bench_full_suite() -> dict[str, object]:
    """Benchmark all 7 spectra sequentially and report total time."""
    exp = make_experiment()
    simopt = make_simopt()
    per_system: list[dict[str, object]] = []
    total = 0.0

    for name in sorted(SYSTEMS):
        sys = SYSTEMS[name]()
        stats = bench(sys, exp, simopt, repeats=1)
        per_system.append({"system": name, "time_s": stats["mean"]})
        total += stats["mean"]
        print(f"  {name}: {stats['mean']:.4f}s")

    return {"per_system": per_system, "total_s": total}


def write_markdown(
    nuclei_results: list[dict[str, object]],
    interp_results: list[dict[str, object]],
    suite_result: dict[str, object],
) -> None:
    """Write the full benchmark report to BENCHMARK.md."""
    info = env_info()
    lines: list[str] = []
    lines.append("# radpair — Benchmark Results (single-core)\n")
    lines.append(f"Generated: {info['timestamp']}\n")
    lines.append("## Environment\n")
    lines.append(f"- Python: {info['python']}")
    lines.append(f"- CPU: {info['cpu']} ({info['nproc']} cores)")
    lines.append(f"- Platform: {info['platform']}")
    lines.append(f"- Warmup runs: {N_WARMUP}")
    lines.append(f"- Measurement runs: {N_REPEATS}")
    lines.append(f"- Grid knots: {N_KNOTS}")
    lines.append(f"- Field axis: {len(FIELD_AXIS)} points\n")

    # --- Nuclei scaling ---
    lines.append("## 1. Nuclei scaling (S1–S7)\n")
    lines.append("Each system is simulated with default settings ")
    lines.append(f"(grid={N_KNOTS}, refinement=1, single core).\n")
    headers = [
        "System",
        "Nuclei",
        "Description",
        "Mean (s)",
        "Min (s)",
        "Max (s)",
        "Std (s)",
    ]
    rows: list[list[str]] = []
    for r in nuclei_results:
        rows.append(
            [
                str(r["system"]),
                str(r["n_nuclei"]),
                str(r["description"]),
                f"{r['mean']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
                f"{r['std']:.4f}",
            ]
        )
    lines.append(format_md_table(headers, rows))
    lines.append("")

    # --- Interpolation scaling ---
    lines.append("## 2. Interpolation scaling (S3, refinement 1–4)\n")
    lines.append("System S3 (2 anisotropic nuclei) with increasing ")
    lines.append("refinement factor.  Orientations = knots × refinement.\n")
    headers = [
        "Refinement",
        "Orientations",
        "Mean (s)",
        "Min (s)",
        "Max (s)",
        "Std (s)",
    ]
    rows = []
    for r in interp_results:
        rows.append(
            [
                str(r["refinement"]),
                str(r["n_orientations"]),
                f"{r['mean']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
                f"{r['std']:.4f}",
            ]
        )
    lines.append(format_md_table(headers, rows))
    lines.append("")

    # --- Full suite ---
    lines.append("## 3. Full suite (S1–S7, single run each)\n")
    lines.append("All 7 spectra simulated sequentially with default settings.\n")
    headers = ["System", "Time (s)"]
    rows = []
    for r in suite_result["per_system"]:
        rows.append([str(r["system"]), f"{r['time_s']:.4f}"])
    rows.append(["**Total**", f"**{suite_result['total_s']:.4f}**"])
    lines.append(format_md_table(headers, rows))
    lines.append("")

    content = "\n".join(lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"\nBenchmark report written to {OUTPUT_FILE}")


def main() -> None:
    print("=" * 60)
    print("radpair benchmark (single-core)")
    print("=" * 60)

    print("\n[1/3] Nuclei scaling (S1–S7)...")
    nuclei_results = bench_nuclei_scaling()

    print("\n[2/3] Interpolation scaling (S3, refinement 1–4)...")
    interp_results = bench_interpolation_scaling()

    print("\n[3/3] Full suite (S1–S7, single run each)...")
    suite_result = bench_full_suite()
    print(f"  Total: {suite_result['total_s']:.4f}s")

    write_markdown(nuclei_results, interp_results, suite_result)


if __name__ == "__main__":
    main()
