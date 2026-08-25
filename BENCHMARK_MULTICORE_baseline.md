# radpair — Benchmark Results (multi-core)

Generated: 2026-08-25 08:29 UTC

## Environment

- Python: 3.13.14
- CPU: unknown (32 cores)
- Platform: Linux-6.12.96-1-MANJARO-x86_64-with-glibc2.44
- Warmup runs: 1
- Measurement runs: 3
- Grid points: 12
- Field axis: 500 points
- Default CPU cores: 4

## 1. Nuclei scaling (S1–S7, 4 cores)

Each system is simulated with default settings 
(grid=12, refinement=1, 4 cores).

| System | Nuclei | Description | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|---|
| S1 | 0 | Bare radical pair (0 nuclei, anisotropic g, D/E/J) | 0.0265 | 0.0258 | 0.0274 | 0.0007 |
| S2 | 1 | Single donor ¹H, isotropic g and A | 0.0288 | 0.0277 | 0.0299 | 0.0009 |
| S3 | 2 | 2 nuclei: donor ¹H + acceptor 2×¹⁴N, anisotropic | 0.0557 | 0.0536 | 0.0568 | 0.0015 |
| S4 | 2 | Swap of S3: acceptor ¹H + donor 2×¹⁴N | 0.0511 | 0.0504 | 0.0523 | 0.0008 |
| S5 | 3 | 3 nuclei: methyl (3×¹H) + ¹H + 2×¹⁴N | 0.1356 | 0.1337 | 0.1372 | 0.0014 |
| S6 | 4 | 4 nuclei: ¹H + ¹⁴N + ³⁵Cl + 2×¹H, mixed iso/aniso g | 0.2235 | 0.2186 | 0.2275 | 0.0037 |
| S7 | 5 | 5 nuclei: all groups active, maximum complexity | 0.6715 | 0.6691 | 0.6740 | 0.0020 |

## 2. Interpolation scaling (S3, refinement 1–4, 4 cores)

System S3 (2 anisotropic nuclei) with increasing 
refinement factor, 4 CPU cores.  Orientations = grid_points × refinement.

| Refinement | Orientations | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|
| 1 | 12 | 0.0527 | 0.0526 | 0.0528 | 0.0001 |
| 2 | 24 | 0.2154 | 0.2117 | 0.2213 | 0.0042 |
| 3 | 36 | 0.4067 | 0.3996 | 0.4119 | 0.0052 |
| 4 | 48 | 0.7309 | 0.7156 | 0.7407 | 0.0110 |

## 3. CPU-core scaling (S7, 1–8 cores)

System S7 (5 nuclei, slowest) with increasing CPU cores.  Speedup is relative to the 1-core run.

| CPU cores | Mean (s) | Min (s) | Max (s) | Std (s) | Speedup |
|---|---|---|---|---|---|
| 1 | 1.7706 | 1.7476 | 1.7830 | 0.0162 | 1.00× |
| 2 | 1.0045 | 0.9943 | 1.0113 | 0.0073 | 1.76× |
| 4 | 0.6585 | 0.6542 | 0.6626 | 0.0034 | 2.69× |
| 8 | 0.5569 | 0.5514 | 0.5604 | 0.0039 | 3.18× |

## 4. Full suite (S1–S7, single run each, 4 cores)

All 7 spectra simulated sequentially with 4 CPU cores and default settings.

| System | Time (s) |
|---|---|
| S1 | 0.0250 |
| S2 | 0.0273 |
| S3 | 0.0511 |
| S4 | 0.0564 |
| S5 | 0.1359 |
| S6 | 0.2295 |
| S7 | 0.6546 |
| **Total** | **1.1797** |
