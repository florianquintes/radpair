"""Benchmark script for multi-core radpair simulation performance.

Measures four performance trends:

1. **Nuclei scaling** — runtime vs. number of active nuclei groups (0–5),
   using systems S1–S7 with a fixed CPU-core count.

2. **Interpolation scaling** — runtime vs. refinement factor (1, 2, 3, 4),
   using a fixed 2-nuclei system (S3) with a fixed CPU-core count.

3. **CPU-core scaling** — runtime vs. number of CPU cores (1, 2, 4, 8),
   using a fixed 5-nuclei system (S7, the slowest) to show parallel
   speedup.

4. **Full suite** — total wall-clock time for all 7 example spectra
   (S1–S7) with default settings and a fixed CPU-core count.

Results are written to ``BENCHMARK_MULTICORE.md`` in the repository
root as Markdown tables.

Usage::

    uv run python benchmarks/run_benchmark_multicore.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _common import (
    CPU_CORES_VALUES,
    DESCRIPTIONS,
    FIELD_AXIS,
    N_KNOTS,
    N_REPEATS,
    N_WARMUP,
    REFINEMENT_VALUES,
    SYSTEMS,
    bench,
    bench_call_multicore,
    count_active_nuclei,
    env_info,
    format_md_table,
    make_experiment,
)

from radpair._types import SimulationOptions

DEFAULT_CPU_CORES = 4
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "BENCHMARK_MULTICORE.md")


def bench_nuclei_scaling() -> list[dict[str, object]]:
    """Benchmark S1–S7 (0 to 5 active nuclei groups) with multicore."""
    exp = make_experiment()
    simopt = SimulationOptions(
        knots=N_KNOTS,
        refinement=1,
        cpu_cores=DEFAULT_CPU_CORES,
    )
    results: list[dict[str, object]] = []

    for name in sorted(SYSTEMS):
        sys = SYSTEMS[name]()
        n_nuclei = count_active_nuclei(sys)
        stats = bench(sys, exp, simopt, call_fn=bench_call_multicore)
        results.append(
            {
                "system": name,
                "n_nuclei": n_nuclei,
                "description": DESCRIPTIONS[name],
                **stats,
            }
        )
        print(
            f"  {name} ({n_nuclei} nuclei, {DEFAULT_CPU_CORES} cores): "
            f"mean={stats['mean']:.4f}s  min={stats['min']:.4f}s  "
            f"max={stats['max']:.4f}s  std={stats['std']:.4f}s"
        )

    return results


def bench_interpolation_scaling() -> list[dict[str, object]]:
    """Benchmark S3 with refinement factors 1, 2, 3, 4 with multicore."""
    sys = SYSTEMS["S3"]()
    exp = make_experiment()
    results: list[dict[str, object]] = []

    for refinement in REFINEMENT_VALUES:
        simopt = SimulationOptions(
            knots=N_KNOTS,
            refinement=refinement,
            cpu_cores=DEFAULT_CPU_CORES,
        )
        stats = bench(sys, exp, simopt, call_fn=bench_call_multicore)
        n_orientations = N_KNOTS * refinement
        results.append(
            {
                "refinement": refinement,
                "n_orientations": n_orientations,
                **stats,
            }
        )
        print(
            f"  refinement={refinement} ({n_orientations} orientations, "
            f"{DEFAULT_CPU_CORES} cores): "
            f"mean={stats['mean']:.4f}s  min={stats['min']:.4f}s  "
            f"max={stats['max']:.4f}s  std={stats['std']:.4f}s"
        )

    return results


def bench_cpu_cores_scaling() -> list[dict[str, object]]:
    """Benchmark S7 with varying CPU cores (1, 2, 4, 8).

    S7 (5 nuclei, slowest system) is used to show parallel speedup.
    Also includes the single-core baseline for comparison.
    """
    sys = SYSTEMS["S7"]()
    exp = make_experiment()
    results: list[dict[str, object]] = []

    for n_cores in CPU_CORES_VALUES:
        simopt = SimulationOptions(
            knots=N_KNOTS,
            refinement=1,
            cpu_cores=n_cores,
        )
        stats = bench(sys, exp, simopt, call_fn=bench_call_multicore)
        results.append(
            {
                "cpu_cores": n_cores,
                **stats,
            }
        )
        print(
            f"  cpu_cores={n_cores}: "
            f"mean={stats['mean']:.4f}s  min={stats['min']:.4f}s  "
            f"max={stats['max']:.4f}s  std={stats['std']:.4f}s"
        )

    return results


def bench_full_suite() -> dict[str, object]:
    """Benchmark all 7 spectra sequentially with multicore and report total."""
    exp = make_experiment()
    simopt = SimulationOptions(
        knots=N_KNOTS,
        refinement=1,
        cpu_cores=DEFAULT_CPU_CORES,
    )
    per_system: list[dict[str, object]] = []
    total = 0.0

    for name in sorted(SYSTEMS):
        sys = SYSTEMS[name]()
        stats = bench(
            sys,
            exp,
            simopt,
            call_fn=bench_call_multicore,
            repeats=1,
        )
        per_system.append({"system": name, "time_s": stats["mean"]})
        total += stats["mean"]
        print(f"  {name}: {stats['mean']:.4f}s")

    return {"per_system": per_system, "total_s": total}


def write_markdown(
    nuclei_results: list[dict[str, object]],
    interp_results: list[dict[str, object]],
    cores_results: list[dict[str, object]],
    suite_result: dict[str, object],
) -> None:
    """Write the full multicore benchmark report to BENCHMARK_MULTICORE.md."""
    info = env_info()
    lines: list[str] = []
    lines.append("# radpair — Benchmark Results (multi-core)\n")
    lines.append(f"Generated: {info['timestamp']}\n")
    lines.append("## Environment\n")
    lines.append(f"- Python: {info['python']}")
    lines.append(f"- CPU: {info['cpu']} ({info['nproc']} cores)")
    lines.append(f"- Platform: {info['platform']}")
    lines.append(f"- Warmup runs: {N_WARMUP}")
    lines.append(f"- Measurement runs: {N_REPEATS}")
    lines.append(f"- Grid knots: {N_KNOTS}")
    lines.append(f"- Field axis: {len(FIELD_AXIS)} points")
    lines.append(f"- Default CPU cores: {DEFAULT_CPU_CORES}\n")

    # --- Nuclei scaling ---
    lines.append(f"## 1. Nuclei scaling (S1–S7, {DEFAULT_CPU_CORES} cores)\n")
    lines.append("Each system is simulated with default settings ")
    lines.append(f"(grid={N_KNOTS}, refinement=1, {DEFAULT_CPU_CORES} cores).\n")
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
    lines.append(
        f"## 2. Interpolation scaling (S3, refinement 1–4, {DEFAULT_CPU_CORES} cores)\n"
    )
    lines.append("System S3 (2 anisotropic nuclei) with increasing ")
    lines.append(
        f"refinement factor, {DEFAULT_CPU_CORES} CPU cores.  "
        "Orientations = knots × refinement.\n"
    )
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

    # --- CPU-core scaling ---
    lines.append("## 3. CPU-core scaling (S7, 1–8 cores)\n")
    lines.append(
        "System S7 (5 nuclei, slowest) with increasing CPU cores.  "
        "Speedup is relative to the 1-core run.\n"
    )
    base_mean = cores_results[0]["mean"]
    headers = [
        "CPU cores",
        "Mean (s)",
        "Min (s)",
        "Max (s)",
        "Std (s)",
        "Speedup",
    ]
    rows = []
    for r in cores_results:
        speedup = base_mean / r["mean"] if r["mean"] > 0 else 0
        rows.append(
            [
                str(r["cpu_cores"]),
                f"{r['mean']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
                f"{r['std']:.4f}",
                f"{speedup:.2f}×",
            ]
        )
    lines.append(format_md_table(headers, rows))
    lines.append("")

    # --- Full suite ---
    lines.append(
        f"## 4. Full suite (S1–S7, single run each, {DEFAULT_CPU_CORES} cores)\n"
    )
    lines.append(
        f"All 7 spectra simulated sequentially with {DEFAULT_CPU_CORES} CPU cores "
        "and default settings.\n"
    )
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
    print(f"radpair benchmark (multi-core, default {DEFAULT_CPU_CORES} cores)")
    print("=" * 60)

    print(f"\n[1/4] Nuclei scaling (S1–S7, {DEFAULT_CPU_CORES} cores)...")
    nuclei_results = bench_nuclei_scaling()

    print(
        f"\n[2/4] Interpolation scaling (S3, refinement 1–4, {DEFAULT_CPU_CORES} cores)..."
    )
    interp_results = bench_interpolation_scaling()

    print("\n[3/4] CPU-core scaling (S7, 1–8 cores)...")
    cores_results = bench_cpu_cores_scaling()

    print(f"\n[4/4] Full suite (S1–S7, single run each, {DEFAULT_CPU_CORES} cores)...")
    suite_result = bench_full_suite()
    print(f"  Total: {suite_result['total_s']:.4f}s")

    write_markdown(nuclei_results, interp_results, cores_results, suite_result)


if __name__ == "__main__":
    main()
