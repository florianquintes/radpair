# radpair — Benchmark Results (single-core)

Generated: 2026-08-25 08:29 UTC

## Environment

- Python: 3.13.14
- CPU: unknown (32 cores)
- Platform: Linux-6.12.96-1-MANJARO-x86_64-with-glibc2.44
- Warmup runs: 1
- Measurement runs: 3
- Grid points: 12
- Field axis: 500 points

## 1. Nuclei scaling (S1–S7)

Each system is simulated with default settings 
(grid=12, refinement=1, single core).

| System | Nuclei | Description | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|---|
| S1 | 0 | Bare radical pair (0 nuclei, anisotropic g, D/E/J) | 0.0106 | 0.0096 | 0.0112 | 0.0007 |
| S2 | 1 | Single donor ¹H, isotropic g and A | 0.0152 | 0.0138 | 0.0162 | 0.0010 |
| S3 | 2 | 2 nuclei: donor ¹H + acceptor 2×¹⁴N, anisotropic | 0.0710 | 0.0690 | 0.0748 | 0.0027 |
| S4 | 2 | Swap of S3: acceptor ¹H + donor 2×¹⁴N | 0.0688 | 0.0678 | 0.0697 | 0.0008 |
| S5 | 3 | 3 nuclei: methyl (3×¹H) + ¹H + 2×¹⁴N | 0.2815 | 0.2805 | 0.2825 | 0.0008 |
| S6 | 4 | 4 nuclei: ¹H + ¹⁴N + ³⁵Cl + 2×¹H, mixed iso/aniso g | 0.4930 | 0.4908 | 0.4965 | 0.0025 |
| S7 | 5 | 5 nuclei: all groups active, maximum complexity | 1.7157 | 1.6916 | 1.7476 | 0.0235 |

## 2. Interpolation scaling (S3, refinement 1–4)

System S3 (2 anisotropic nuclei) with increasing 
refinement factor.  Orientations = grid_points × refinement.

| Refinement | Orientations | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|
| 1 | 12 | 0.0725 | 0.0711 | 0.0737 | 0.0010 |
| 2 | 24 | 0.3635 | 0.3600 | 0.3681 | 0.0034 |
| 3 | 36 | 0.7643 | 0.7570 | 0.7685 | 0.0052 |
| 4 | 48 | 1.3077 | 1.2963 | 1.3215 | 0.0104 |

## 3. Full suite (S1–S7, single run each)

All 7 spectra simulated sequentially with default settings.

| System | Time (s) |
|---|---|
| S1 | 0.0062 |
| S2 | 0.0087 |
| S3 | 0.0691 |
| S4 | 0.0674 |
| S5 | 0.2993 |
| S6 | 0.5437 |
| S7 | 1.7028 |
| **Total** | **2.6971** |
