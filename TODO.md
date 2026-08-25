# radpair — TODO / Issue Tracker

Organized by milestone. Each section maps to one or more GitHub Issues.

---

## Milestone 1 — Documentation

### Issue: ~~Switch Sphinx theme to `sphinx-rtd-theme`~~ ✅ Done

- Replaced `alabaster` with `sphinx_rtd_theme` in `docs/source/conf.py`.

### Issue: ~~Auto-import version from `pyproject.toml`~~ ✅ Done

- `conf.py` now uses `importlib.metadata.version("radpair")` to populate `version` and `release` dynamically.

### Issue: ~~Enable Sphinx extensions for autodoc~~ ✅ Done

- Enabled `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `sphinx-autodoc-typehints`, and `sphinx-copybutton` in `conf.py`.
- Configured `autodoc_typehints = "description"`.
- Added `sphinx-copybutton` as dev dependency.

### Issue: Write documentation content — Introduction, Installation, Examples, API Reference, Development, Contributing, License

Restructure `docs/source/index.rst` and add the following pages under `docs/source/`:

1. **Introduction** (`introduction.rst`) ✅ Done
   - What `radpair` does: analytic simulation of cw-EPR spectra of singlet-born spin-correlated radical pairs.
   - Physics background: analytic Hamiltonian solution, pseudo-secular approximation.
   - Based on a Fortran 77 program from a thesis.
   - **TODO**: Add reference to original thesis and publications (user will provide later).
   - Module overview table added.

2. **Installation** (`installation.rst`) ✅ Done
   - Prerequisites: Python 3.13, `uv`.
   - `uv sync --dev` for development install.
   - Note the `eprbase` dependency and where to get it.

3. **Examples** (`examples.rst`) ✅ Done (placeholder)
   - End-to-end usage example: construct `Spinsystem`, `Exp`, `SimOpt` objects and call `do_simulation` / `do_simulation_multicore`.
   - Show required attributes for each object (document the dynamic-attribute interface).
   - Include a plotted sample spectrum if possible.
   - **TODO**: Fill in worked examples in a future release.

4. **API Reference** (`api_reference.rst`) ✅ Done
   - Automate via `automodule` directives for `radpair.core`, `radpair.classes`, `radpair.functions`, `radpair._wrappers`.
   - Ensure every public function and class has a numpydoc-style docstring rendering correctly.

5. **Development** (`development.rst`) ✅ Done
   - Commands: test, lint, format, docs, build.
   - CI pipeline overview (3 workflows: ci.yml, format.yml, docs.yml).
   - Release flow (tag `v*.*.*` → auto-publish to PyPI).

6. **Contributing** (`contributing.rst`) ✅ Done
   - Project structure overview for `src/radpair/` modules.
   - Branch / PR conventions.
   - Code style: ruff defaults, numpydoc docstrings, type annotations required.

7. **License** (`license.rst`) ✅ Done
   - References `LICENSE` file and GPLv3 summary.
   - `LICENSE` file exists in repo root.

### Issue: ~~Wire up toctree and verify docs build~~ ✅ Done

- All 7 `.rst` pages created and filled (`introduction`, `installation`, `examples`, `api_reference`, `development`, `contributing`, `license`).
- `index.rst` contains project overview and toctree.
- API reference uses `automodule` directives with `:no-index:` to avoid duplicate descriptions.
- License page references `LICENSE` file and GPLv3 summary.
- Docs build passes with zero warnings.
- **Remaining**: Fill in worked examples (`examples.rst`); add literature reference to `introduction.rst`.

---

## Milestone 2 — Tests

### Issue: ~~Set up test infrastructure~~ ✅ Done

- `tests/conftest.py` created with shared fixtures using `types.SimpleNamespace`:
  - **Spinsystem fixtures**: `minimal_spinsystem` (1 donor nucleus, isotropic, no ZFS/exchange), `full_spinsystem` (3 active nuclei groups, anisotropic, nonzero D/E/J), `donor_only_spinsystem` (2 donor groups), `acceptor_only_spinsystem` (2 acceptor groups).
  - **Experiment fixture**: `experiment` (X-band, 9.5 GHz, 500 points, 320–370 mT).
  - **SimOpt fixtures**: `simopt_basic` (10 knots, no interpolation, 1 core), `simopt_multicore` (2 cores), `simopt_auto_cores` (0 = auto-detect), `simopt_interpolation` (5 knots, refinement 3).
- Unit conventions documented in the conftest module docstring (field axis in milliTesla, freq in Hz, couplings in MHz, etc.).
- `tests/test_smoke.py` created with 14 smoke tests:
  - Fixture attribute validation (all required attrs present on each fixture).
  - Single-core `do_simulation` runs (minimal, full, interpolation).
  - Multicore `do_simulation_multicore` matches single-core output.
  - Multicore auto-detect (`cpu_cores=0`).
- All 14 tests pass; lint and format checks clean.

### Issue: ~~Tests for `radpair/functions.py`~~ ✅ Done

- `tests/test_functions.py` — 33 tests across 8 test classes:
  - `tensor_rotation` (8 tests): 2D/3D identity rotation, known-angle rotations (180° about z, 90° about z), `psi=None` default, invalid dimensions, multiple angles, orthogonality preservation.
  - `get_multiplicity` (6 tests): parametrized valid spins (0, 0.5, 1, 1.5), negative spin error, non-half-integer error.
  - `vector_product_combinations` (2 tests): shape and values.
  - `get_generalized_Pascal` (8 tests): n=0 returns `[1]`, known cases (n=2/s=0.5→`[1,2,1]`, n=1/s=0.5→`[1,1]`, n=1/s=1→`[1,1,1]`), all error cases.
  - `get_normalized_Pascal` (3 tests): sums to 1, n=0, matches rescaled generalized.
  - `rescale_array` (4 tests): default norm, custom norm, values, zero-sum error.
  - `get_D_diag` (3 tests): known D/E values, zero E, shape.
  - `MHz_2_T` (4 tests): scalar, array, anisotropic g, negative-g error.

### Issue: ~~Tests for `radpair/classes.py`~~ ✅ Done

- `tests/test_classes.py` — 26 tests across 6 test classes:
  - `Matrix.__init__` (2 tests): stores matrix, `matrix_rot` is `None`.
  - `Matrix.matrot` (3 tests): identity rotation preserves matrix, original matrix unchanged, rotated shape.
  - `Matrix.get_hyperfine_projection` (3 tests): isotropic diagonal, shape, all non-negative.
  - `Core.__init__` (10 tests): stores number/spin, total_spin, zero core, pascal, mI_len, all error cases (negative number/spin, non-integer number, non-half-integer spin).
  - `Core.set_hyperfine_matrix` (3 tests): shape, values, zero core.
  - `Core.get_magnetic_spin_vector` (5 tests): spin-1/2 one/two nuclei, spin-1, dtype float32, symmetry.

### Issue: ~~Tests for `radpair/_wrappers.py`~~ ✅ Done

- `tests/test_wrappers.py` — 11 tests across 3 test classes:
  - `timer` (3 tests): returns original result, prints runtime line, preserves kwargs.
  - `function_benchmark` (3 tests): does not return original result, prints statistics (average/best/worst), runs exactly niter times.
  - `multicore` (4 tests): output matches single-core concatenation, `cpu_cores=0` auto-detects, single core, uneven split.
  - Uses a trivial `_identity_simulation` function (no `eprbase` dependency) to test `multicore` in isolation.

### Issue: ~~Tests for `radpair/core.py` — `do_simulation`~~ ✅ Done

- `tests/test_simulation_regression.py` — `TestDoSimulationInvariants` (11 tests):
  - Output shape matches `Exp.B_z` for minimal (1-nucleus) and full (5-nucleus) systems.
  - Output is real-valued (not complex) for both systems.
  - No NaNs and no Infs in output for both systems.
  - Interpolation mode (`refinement > 1`) produces correct shape, no NaNs, and different output than no-interpolation mode.

### Issue: ~~Tests for `radpair/core.py` — `do_simulation_multicore`~~ ✅ Done

- `tests/test_simulation_regression.py` — `TestDoSimulationMulticore` (4 tests):
  - Multicore (2 cores) output matches single-core output for minimal and full systems.
  - `cpu_cores = 0` (auto-detect) produces correct shape and no NaNs.
  - `cpu_cores = 1` (degenerate case) matches single-core output.

### Issue: ~~Create reference test spectra and end-to-end comparison tests~~ ✅ Done

- `tests/reference_data/` directory contains 7 `.npz` files (S1–S7) with field axis, intensity, and all input parameters as metadata.
- `tests/generate_reference_spectra.py` — script to regenerate reference data (`uv run python tests/generate_reference_spectra.py`).
- `tests/plot_reference_spectra.py` — generates `all_spectra.pdf` combined plot.
- `tests/reference_data/README.md` — full documentation with per-spectrum parameter tables, coverage matrix, and regeneration instructions.
- `tests/test_simulation_regression.py` — end-to-end regression tests (14 tests):
  - 7 parametrized `test_reference_spectrum[S1–S7]` tests: load each `.npz`, reconstruct inputs from metadata, re-run `do_simulation`, assert `np.allclose(result, reference, rtol=1e-10, atol=1e-12)`.
  - `test_reference_spectra_have_nonzero_variance` — all spectra have meaningful structure.
  - `test_reference_spectra_have_negative_and_positive` — spin-correlated RP hallmark (both absorptive and emissive lines).
  - `test_swap_pair_s3_s4_have_similar_sums` — donor/acceptor swap preserves spectral sum.
- Coverage: 0–5 nuclei groups, iso/aniso g-tensors, iso/aniso A-tensors, donor-only/mixed/acceptor-heavy, I=½+1+3/2, n=1+2+3, zero/nonzero D/E/J, zero/nonzero frames, donor/acceptor swap pair.

---

## Milestone 3 — Codebase Refactoring (Pre-structure)

### Issue: ~~Add type annotations to all public and private functions~~ ✅ Done

- Created `src/radpair/_types.py` with `@runtime_checkable` `Protocol` classes (`Spinsystem`, `Experiment`, `SimulationOptions`) documenting all required attributes.
- `core.py` — `do_simulation` and `do_simulation_multicore` now use the Protocol types instead of `object`; parameters renamed to lowercase (`spinsystem`, `experiment`, `simopt`).
- `classes.py` — `Matrix.__init__` parameter typed as `np.ndarray`; `matrot` returns `-> None`; `get_hyperfine_projection` returns `-> np.ndarray`; `Core.set_hyperfine_matrix` returns `-> None`; `Core.get_magnetic_spin_vector` returns `-> None`; `matrix_rot` attribute typed as `np.ndarray | None`.
- `functions.py` — All `np.array` type hints replaced with `np.ndarray`; `psi` parameter typed as `np.ndarray | None`; return types added to all functions; `MHz_2_T` uses `float | np.ndarray` for both parameter and return.
- `_wrappers.py` — `timer` uses PEP 695 type parameters (`[**P, R]`); `function_benchmark` returns `Callable[..., None]`; `multicore` returns `Callable[..., np.ndarray]`; inner `multicore_wrapper` uses Protocol types for parameters.

### Issue: ~~Eliminate anti-patterns~~ ✅ Done

- `core.py` — `if True:` no-op block removed; variables renamed directly instead of copying.
- `core.py` — `A1`–`A5` dtype conversion and scaling (0.5 and `_GAMMA_E_REF`) merged into a single `for i in range(1, 6)` loop using `vars(Sys)`.
- `core.py` — Three near-identical `tilt_alpha`/`tilt_beta`/`tilt_gamma` arrays replaced with a single `frame_angles` array indexed by column.
- `core.py` — Repeated `np.diag` calls for `a1`–`a5` replaced with a list comprehension; `cl.Matrix` construction uses a list comprehension for `a_matrices`.
- `core.py` — Five `if core_type["N"] == 0:` blocks replaced with a single list comprehension.
- `core.py` — Five `.matrot()` calls and five `get_hyperfine_projection()` calls replaced with a loop over `[g1, g2, D, *a_matrices]` and a list comprehension.
- `core.py` — Five `set_hyperfine_matrix` calls replaced with a `zip` loop.
- `core.py` — Five-deep nested `for` loop replaced with `itertools.product`.
- `core.py` — Four width variables (`width_1`–`width_4`) and subsequent `np.stack` merged into a single `widths = 1 / np.stack(...)`.
- `classes.py` — Magic index `2` in `get_hyperfine_projection` replaced with named constant `_Z_COLUMN`.

### Issue: ~~Replace `float or np.array` type annotation~~ ✅ Done

- `functions.py` — `nu: float or np.array` replaced with `nu: float | np.ndarray`; return type updated to `float | np.ndarray`.

### Issue: ~~Consolidate unit-conversion magic numbers~~ ✅ Done

- `core.py` — `tesang = 1.75880474e8` replaced with module-level constant `_GAMMA_E_REF`, derived from `scipy.constants` as `2 * np.pi * 2 * constant.value("Bohr magneton in Hz/T") * 1e-3` (electron gyromagnetic ratio with g=2, in rad/s per milliTesla). All 13 usages updated. Comment explains the derivation, units, and the 1e-3 mT factor.
- `core.py` — `4 * np.log(2)` replaced with `_GAUSSIAN_FWHM_TO_SIGMA` (FWHM-to-sigma² factor for Gaussian lineshape).

### Issue: ~~Improve docstrings to numpydoc / Sphinx-compatible style~~ ✅ Done

- All functions and classes now have complete numpydoc docstrings with `Parameters`, `Returns`, and where applicable `Raises`, `Notes`, `Examples` sections.
- `.. math::` directives fixed to use proper LaTeX (e.g. `O^{\mathsf{T}}` instead of `O^{-1}`).
- `classes.py` — Docstring parameter mismatch fixed (`diag` → `mat`).
- `core.py` — Parameters standardized to lowercase (`spinsystem`, `experiment`, `simopt`); docstrings reference the Protocol classes.
- `functions.py` — Added `Examples` section to `vector_product_combinations`; `Raises` sections added to `tensor_rotation`, `get_multiplicity`, `get_generalized_Pascal`, `rescale_array`, `MHz_2_T`.
- `classes.py` — `Raises` sections added to `Core.__init__`; class-level docstrings improved with `Attributes` sections.
- `_wrappers.py` — Module docstring added; all decorator docstrings improved.
- `api_reference.rst` updated to include `radpair._types` automodule directive.
- Docs build passes with zero warnings.

### Issue: ~~Remove dead / placeholder code~~ ✅ Done

- `core.py` — `if True:` block with redundant `.copy()` calls removed. Variables (`res_fields`, `width`, `intensity`, `transition`) renamed directly to their final names (`fields`, `widths`, `intensities`, `transitions`) at the point of assignment, eliminating the unnecessary copy.
- `core.py` — `transition = np.zeros(...)` (now `transitions`) is retained as a placeholder required by `spectra.Spectra()` and `interp.Interpolator()`; it is never populated with meaningful transition data but is structurally necessary for the API contract.

### Issue: ~~Add `LICENSE` file~~ ✅ Done

- `LICENSE` file added to repo root.
- Referenced in `README.md` via link to `LICENSE`.

---

## Milestone 4 — Future Release: Complete Restructuring

> Depends on Milestones 1–3. Should be a separate major/minor version bump.

### Issue: ~~Replace dynamic-attribute interface with typed dataclasses~~ ✅ Done (partially — original motivation resolved earlier)

**Original motivation**: `do_simulation` accessed `Sys.g1`, `Sys.A1`, `vars(Sys)["A1"]` dynamically — no static analysis, no IDE support, no validation.

**Achieved in earlier issues**:
- `getattr`/`setattr`/`vars()` fully eliminated from `src/` — all access is direct attribute access (`Sys.A_tensors[i]`, `Sys.nuclei_n[i]`).
- `@runtime_checkable Protocol` classes in `_types.py` documented all fields with type annotations.
- The hardcoded 5-nuclei limit was removed, simplifying the attribute surface.

**Achieved in this issue**:
- Replaced `Protocol` classes with concrete `@dataclass` definitions (`Spinsystem`, `Experiment`, `SimulationOptions`) that can be instantiated directly.
- Added `__post_init__` validation for `Spinsystem`: length consistency (`A_tensors`/`nuclei_n`/`nuclei_I`/`A_frames`), shape checks, positive g-values, non-negative integer nuclei counts, half-integer spins, positive linewidth, valid donor/acceptor indices, no donor/acceptor overlap.
- Added sensible defaults: `D=0`, `E=0`, `J_ex=0`, zero frames, empty donor/acceptor lists, `Experiment.magnetic_field` auto-copies `B_z` when `None`.
- Eliminated `SimpleNamespace` from all test fixtures, example scripts, benchmark systems, reference-spectrum generator, and documentation.
- Added `spinsystem_field_names()` helper for introspecting dataclass fields.
- 27 new validation tests in `tests/test_types.py` covering all `__post_init__` checks and default-value behavior.
- All 189 tests pass (162 existing + 27 new); ruff format + check clean; docs build with zero warnings.
- **Backward compatibility**: `SimpleNamespace` objects with the right attributes still work at runtime (Python duck-typing), but type checkers will flag them — intentional to encourage migration.

### Issue: ~~Remove hardcoded 5-nuclei limit~~ ✅ Done

- **Breaking API change**: the fixed `A1`–`A5`, `n1`–`n5`, `I1`–`I5`, `A1_frame`–`A5_frame` attributes are replaced by list-based attributes:
  - `A_tensors` (list of 3-element arrays, variable length)
  - `nuclei_n` (list of int, same length as `A_tensors`)
  - `nuclei_I` (list of float, same length as `A_tensors`)
  - `A_frames` (list of 3-element arrays, same length as `A_tensors`)
  - `donor_list` / `acceptor_list` are now **0-indexed** (was 1-indexed)
- The simulation now accepts an arbitrary number of nuclei groups.
- Updated `Spinsystem` Protocol in `_types.py` to document the new list-based interface.
- Updated all 4 pipeline functions in `functions.py` (`prepare_spinsystem`, `build_tensors`, `rotate_tensors`, `compute_hyperfine_combinations`) to iterate over `Sys.A_tensors` instead of `range(1, 6)`.
- `build_tensors` now returns shape `(3 + n_nuclei, 3, 3)` instead of fixed `(8, 3, 3)`.
- Updated all fixtures in `tests/conftest.py`, `tests/test_smoke.py`, `tests/test_simulation_regression.py`, `tests/test_stages.py`, `tests/generate_reference_spectra.py`, `examples/_systems.py`, `benchmarks/_common.py`, `docs/source/examples.rst`, and `examples/full_five_nuclei.py`.
- Regenerated all 7 reference `.npz` files with the new API.
- All 162 tests pass; ruff format + check clean; docs build with zero warnings.

### Issue: ~~Break `do_simulation` into composable functions~~ ✅ Done

- `do_simulation` was ~280 lines doing unit conversion, tensor setup, rotation, resonance-field calculation, interpolation, and spectrum assembly in one function.
- Extracted 8 composable functions into `functions.py`:
  1. `prepare_spinsystem` — deep-copies the spin system and converts all parameters to internal angular-frequency units.
  2. `setup_orientation_grid` — creates coarse/fine orientation grids and integration weights.
  3. `build_tensors` — builds diagonal tensors and extracts Euler frame angles.
  4. `rotate_tensors` — tilts tensors into reference frames and rotates for each orientation (replaces `cl.Matrix` with direct `tensor_rotation` calls).
  5. `compute_hyperfine_combinations` — computes sum/difference hyperfine matrices and Pascal-triangle weights (replaces `cl.Core` with direct helper calls).
  6. `compute_resonance_fields` — solves the analytic Hamiltonian for 4 transitions, producing resonance fields, quantum beats, and widths.
  7. `compute_intensities` — calculates phase-angle-weighted line intensities with `[+1, -1, +1, -1]` pattern.
  8. `assemble_spectrum` — converts fields back to mT, optionally interpolates, applies weights/linewidth, and performs spectral summation.
- Constants `_GAMMA_E_REF` and `_GAUSSIAN_FWHM_TO_SIGMA` moved from `core.py` to `functions.py`.
- `core.py` reduced to a ~30-line orchestrator; imports of `deepcopy`, `product`, `scipy.constants`, `eprbase`, and `radpair.classes` removed.
- `classes.py` unchanged — `Matrix` and `Core` remain for existing tests and potential future use.
- 38 new unit tests in `tests/test_stages.py` covering each stage independently (shapes, values, edge cases) plus 2 end-to-end pipeline parity tests.
- All 162 tests pass (124 existing + 38 new); ruff format + check clean; docs build with zero warnings.

### Issue: ~~Remove deprecated Fibonacci-sphere code~~ ✅ Done

- Removed `sphere_fibonacci_grid_points`, `cartesian2spherical`, and `get_fibonacci_sphere` from `functions.py`.
- These functions were deprecated: the orientation grid is now constructed by the `eprbase` module (`grid.Grid`), used by `setup_orientation_grid`.
- Removed 11 corresponding tests (3 test classes) from `tests/test_functions.py`.
- Updated module docstring and `setup_orientation_grid` docstring to remove Fibonacci references.
- Updated TODO.md references.
- All 178 tests pass; ruff format + check clean; docs build with zero warnings.

### Issue: ~~Improve parallelization~~ ✅ Done

- Old `multicore` wrapper split the field axis across processes, duplicating the entire analytic pipeline per core. Now deprecated in `_wrappers.py`.
- New `do_simulation_multicore` in `core.py` runs the analytic pipeline (stages 1–7) once on the main process, then distributes only the Gaussian summation across worker processes via `multiprocessing.Pool`.
- Each worker handles a chunk of orientation peaks, bounded by `max_chunk_mb`.
- `SimulationOptions.max_chunk_mb` controls per-chunk memory: `None` = auto-detect from available RAM (25% cap), `0` = no limit, positive int = explicit MB limit.
- Added `_get_available_ram()` utility (psutil → /proc/meminfo → vm_stat → 1 GB fallback).
- Added `gaussian_summation()` and `_compute_chunk_size()` in `functions.py`.
- `assemble_spectrum` now calls `gaussian_summation` instead of `Spectra.by_summation` directly.
- Documented `max_chunk_mb` behavior in `examples.rst` with a dedicated "Chunked Gaussian summation" section.
- Old `multicore` decorator kept for backward compatibility but marked deprecated.
- 12 new tests: `_get_available_ram` (2), `_compute_chunk_size` (6), `gaussian_summation` (4).
- All 190 tests pass; ruff format + check clean; docs build with zero warnings.

### Issue: ~~Performance audit~~ ✅ Done

- Profiled `do_simulation` with cProfile on S7 (5 nuclei) at knots=12, 20, 25.
- **Result**: 98% of runtime is in `eprbase.Spectra._get_gaussian` (46%) and `Spectra.by_summation` (36%), which build a dense `(n_peaks, n_field)` float32 array. All `radpair` code combined is <2%.
- The original TODO concerns are all non-issues after refactoring:
  - ~~"5-deep nested loop builds Python lists"~~ — `compute_hyperfine_combinations` takes 0.003s (0.06%).
  - ~~"tensor_rotation allocates eulermatrix with per-element assignment"~~ — already vectorized; 27 calls take 0.001s.
  - ~~"`sphere_fibonacci_grid_points` uses Python loops"~~ — removed entirely.
- The chunked Gaussian summation (see "Improve parallelization" issue above) addresses the memory bottleneck by splitting the peak array into RAM-bounded chunks.
- Further performance gains require optimizing `eprbase.Spectra._get_gaussian` (e.g., chunked numexpr evaluation), which is outside the radpair scope.
