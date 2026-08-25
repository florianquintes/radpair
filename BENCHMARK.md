# radpair — Benchmark Results (single-core)

Generated: 2026-08-25 13:24 UTC

## Environment

- Python: 3.13.14
- CPU: unknown (32 cores)
- Platform: Linux-6.12.103-1-MANJARO-x86_64-with-glibc2.44
- Warmup runs: 1
- Measurement runs: 3
- Grid knots: 12
- Field axis: 500 points

## 1. Nuclei scaling (S1–S7)

Each system is simulated with default settings 
(grid=12, refinement=1, single core).

| System | Nuclei | Description | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|---|
| S1 | 0 | Bare radical pair (0 nuclei, anisotropic g, D/E/J) | 0.0101 | 0.0094 | 0.0104 | 0.0005 |
| S2 | 1 | Single donor ¹H, isotropic g and A | 0.0160 | 0.0151 | 0.0168 | 0.0007 |
| S3 | 2 | 2 nuclei: donor ¹H + acceptor 2×¹⁴N, anisotropic | 0.0721 | 0.0694 | 0.0774 | 0.0037 |
| S4 | 2 | Swap of S3: acceptor ¹H + donor 2×¹⁴N | 0.0694 | 0.0692 | 0.0696 | 0.0002 |
| S5 | 3 | 3 nuclei: methyl (3×¹H) + ¹H + 2×¹⁴N | 0.3033 | 0.2989 | 0.3061 | 0.0032 |
| S6 | 4 | 4 nuclei: ¹H + ¹⁴N + ³⁵Cl + 2×¹H, mixed iso/aniso g | 0.5312 | 0.5290 | 0.5355 | 0.0030 |
| S7 | 5 | 5 nuclei: all groups active, maximum complexity | 1.7122 | 1.6983 | 1.7336 | 0.0154 |

## 2. Interpolation scaling (S3, refinement 1–4)

System S3 (2 anisotropic nuclei) with increasing 
refinement factor.  Orientations = knots × refinement.

| Refinement | Orientations | Mean (s) | Min (s) | Max (s) | Std (s) |
|---|---|---|---|---|---|
| 1 | 12 | 0.0671 | 0.0661 | 0.0679 | 0.0007 |
| 2 | 24 | 0.3682 | 0.3663 | 0.3708 | 0.0019 |
| 3 | 36 | 0.7726 | 0.7700 | 0.7752 | 0.0021 |
| 4 | 48 | 1.2996 | 1.2895 | 1.3053 | 0.0071 |

## 3. Full suite (S1–S7, single run each)

All 7 spectra simulated sequentially with default settings.

| System | Time (s) |
|---|---|
| S1 | 0.0064 |
| S2 | 0.0094 |
| S3 | 0.0707 |
| S4 | 0.0706 |
| S5 | 0.3033 |
| S6 | 0.5214 |
| S7 | 1.7553 |
| **Total** | **2.7372** |
