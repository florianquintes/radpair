"""Shared helpers for radpair benchmark scripts."""

from __future__ import annotations

import os
import platform
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from _systems import (
    DESCRIPTIONS,
    FIELD_AXIS,
    GRID_POINTS,
    SYSTEMS,
    make_experiment,
    make_simopt,
)

from radpair.core import do_simulation, do_simulation_multicore

__all__ = [
    "CPU_CORES_VALUES",
    "DESCRIPTIONS",
    "FIELD_AXIS",
    "GRID_POINTS",
    "N_REPEATS",
    "N_WARMUP",
    "REFINEMENT_VALUES",
    "REPO_ROOT",
    "SYSTEMS",
    "bench",
    "bench_call",
    "bench_call_multicore",
    "count_active_nuclei",
    "do_simulation",
    "do_simulation_multicore",
    "env_info",
    "format_md_table",
    "make_experiment",
    "make_simopt",
]

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
N_WARMUP = 1
N_REPEATS = 3
REFINEMENT_VALUES = [1, 2, 3, 4]
CPU_CORES_VALUES = [1, 2, 4, 8]


def count_active_nuclei(sys: SimpleNamespace) -> int:
    """Count the number of active nuclei groups (nuclei_n[i] > 0)."""
    return sum(1 for n in sys.nuclei_n if n > 0)


def bench_call(
    sys: SimpleNamespace,
    exp: SimpleNamespace,
    simopt: SimpleNamespace,
) -> float:
    """Run a single simulation and return wall-clock time in seconds."""
    start = time.perf_counter()
    do_simulation(deepcopy(sys), deepcopy(exp), deepcopy(simopt))
    return time.perf_counter() - start


def bench_call_multicore(
    sys: SimpleNamespace,
    exp: SimpleNamespace,
    simopt: SimpleNamespace,
) -> float:
    """Run a single multicore simulation and return wall-clock time in seconds."""
    start = time.perf_counter()
    do_simulation_multicore(deepcopy(sys), deepcopy(exp), deepcopy(simopt))
    return time.perf_counter() - start


def bench(
    sys: SimpleNamespace,
    exp: SimpleNamespace,
    simopt: SimpleNamespace,
    call_fn=bench_call,
    repeats: int = N_REPEATS,
) -> dict[str, float]:
    """Benchmark a configuration with warmup and repeated runs.

    Returns a dict with 'mean', 'min', 'max', and 'std' times.
    """
    for _ in range(N_WARMUP):
        call_fn(sys, exp, simopt)

    times = [call_fn(sys, exp, simopt) for _ in range(repeats)]
    arr = np.array(times)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def format_md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a Markdown table from headers and rows."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def env_info() -> dict[str, str]:
    """Return environment info for the benchmark header."""
    return {
        "python": sys.version.split()[0],
        "cpu": platform.processor() or "unknown",
        "nproc": str(os.cpu_count() or "unknown"),
        "platform": platform.platform(),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
