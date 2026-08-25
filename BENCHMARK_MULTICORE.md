# radpair — Benchmark Results (multi-core)

Generated: 2026-08-25 13:25 UTC

## Environment

- Python: 3.13.14
- CPU: unknown (32 cores)
- Platform: Linux-6.12.103-1-MANJARO-x86_64-with-glibc2.44
- Warmup runs: 1
- Measurement runs: 3
- Grid knots: 12
- Field axis: 500 points
- Default CPU cores: 4

## 1. Nuclei scaling (S1–S7, 4 cores)

Each system is simulated with default settings 
(grid=12, refinement=1, 4 cores).

| System | Nuclei | Description | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|---|
| S1 | 0 | Bare radical pair (0 nuclei, anisotropic g, D/E/J) | 0.0244 | 0.0240 | 0.0247 | 0.0003 |
| S2 | 1 | Single donor ¹H, isotropic g and A | 0.0270 | 0.0250 | 0.0287 | 0.0015 |
| S3 | 2 | 2 nuclei: donor ¹H + acceptor 2×¹⁴N, anisotropic | 0.0498 | 0.0488 | 0.0512 | 0.0010 |
| S4 | 2 | Swap of S3: acceptor ¹H + donor 2×¹⁴N | 0.0512 | 0.0482 | 0.0546 | 0.0026 |
| S5 | 3 | 3 nuclei: methyl (3×¹H) + ¹H + 2×¹⁴N | 0.1321 | 0.1293 | 0.1341 | 0.0020 |
| S6 | 4 | 4 nuclei: ¹H + ¹⁴N + ³⁵Cl + 2×¹H, mixed iso/aniso g | 0.2210 | 0.2194 | 0.2223 | 0.0012 |
| S7 | 5 | 5 nuclei: all groups active, maximum complexity | 0.6463 | 0.6415 | 0.6501 | 0.0036 |

## 2. Interpolation scaling (S3, refinement 1–4, 4 cores)

System S3 (2 anisotropic nuclei) with increasing 
refinement factor, 4 CPU cores.  Orientations = knots × refinement.

| Refinement | Orientations | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|
| 1 | 12 | 0.0486 | 0.0482 | 0.0493 | 0.0005 |
| 2 | 24 | 0.2119 | 0.2095 | 0.2141 | 0.0019 |
| 3 | 36 | 0.4135 | 0.4071 | 0.4190 | 0.0049 |
| 4 | 48 | 0.7247 | 0.6984 | 0.7538 | 0.0227 |

## 3. CPU-core scaling (S7, 1–8 cores)

System S7 (5 nuclei, slowest) with increasing CPU cores.  Speedup is relative to the 1-core run.

| CPU cores | Mean (s) | Min (s) | Max (s) | Std (s) | Speedup |
|---|---|---|---|---|---|
| 1 | 1.7148 | 1.7087 | 1.7265 | 0.0083 | 1.00× |
| 2 | 1.0135 | 1.0072 | 1.0221 | 0.0063 | 1.69× |
| 4 | 0.6473 | 0.6411 | 0.6505 | 0.0043 | 2.65× |
| 8 | 0.5315 | 0.5287 | 0.5344 | 0.0023 | 3.23× |

## 4. Full suite (S1–S7, single run each, 4 cores)

All 7 spectra simulated sequentially with 4 CPU cores and default settings.

| System | Time (s) |
|---|---|
| S1 | 0.0241 |
| S2 | 0.0275 |
| S3 | 0.0510 |
| S4 | 0.0526 |
| S5 | 0.1350 |
| S6 | 0.2259 |
| S7 | 0.6437 |
| **Total** | **1.1599** |
