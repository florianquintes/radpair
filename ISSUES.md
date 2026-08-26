# radpair — Restructuring Issue List

Generated from a full codebase review. Every issue targets **code reduction**
or **complexity reduction**. Items are ordered by estimated lines removed.

Current sizes: `functions.py` 1078, `core.py` 274, `_types.py` 188,
`classes.py` 158, `_wrappers.py` 159, tests 2533, benchmarks 698,
examples 722 — **~5 764 total**.

---

## Issue 1 — Remove dead `classes.py` module (`Matrix` + `Core`)

**Files**: `src/radpair/classes.py` (158 lines), `tests/test_classes.py` (193 lines)

**Problem**: `Matrix` and `Core` are **not used anywhere in the simulation
pipeline**. After the refactoring into composable functions:

- `rotate_tensors` (`functions.py:543`) calls `tensor_rotation` directly and
  computes hyperfine projections inline — it does not instantiate `Matrix`.
- `compute_hyperfine_combinations` (`functions.py:600`) calls
  `get_multiplicity`, `get_normalized_Pascal`, and
  `vector_product_combinations` directly — it does not instantiate `Core`.
- `Matrix.get_hyperfine_projection` duplicates the projection code at
  `functions.py:594`.
- `Core.__init__` validation duplicates `Spinsystem.__post_init__` validation
  (`_types.py:81`).
- `classes.py` imports `radpair.functions`, creating an unnecessary
  inter-module dependency.

The **only** consumers are `tests/test_classes.py` (26 tests testing dead
code) and `docs/source/api_reference.rst` (automodule directive).

**Action**:
1. Delete `src/radpair/classes.py`.
2. Delete `tests/test_classes.py`.
3. Remove the `automodule:: radpair.classes` directive from
   `docs/source/api_reference.rst`.
4. Remove the `radpair.classes` row from the module overview table in
   `docs/source/introduction.rst:44-45`.
5. Remove `import radpair.classes as cl` from any remaining references
   (none in current source, but check git history if external scripts
   import it).

**Estimated removal**: 351 lines (158 source + 193 tests).

---

## Issue 2 — Remove dead `_wrappers.py` module

**Files**: `src/radpair/_wrappers.py` (159 lines), `tests/test_wrappers.py`
(141 lines)

**Problem**: All three decorators are unused by the simulation pipeline:

- **`multicore`** — deprecated (docstring says so). The new
  `do_simulation_multicore` in `core.py` replaced it. Not imported by any
  source module. Tests use a trivial `_identity_simulation` that doesn't
  even exercise radpair.
- **`timer`** — generic print-based timing decorator. Not used by any
  radpair code. The benchmark scripts (`benchmarks/_common.py`) use their
  own `time.perf_counter()` approach.
- **`function_benchmark`** — generic benchmark decorator. Not used by any
  radpair code. The benchmark scripts use their own `bench()` function.

**Action**:
1. Delete `src/radpair/_wrappers.py`.
2. Delete `tests/test_wrappers.py`.
3. Remove the `automodule:: radpair._wrappers` directive from
   `docs/source/api_reference.rst:18-20`.
4. Remove the `radpair._wrappers` row from the module overview table in
   `docs/source/introduction.rst:50-51`.

**Estimated removal**: 300 lines (159 source + 141 tests).

---

## Issue 3 — Remove `Experiment.magnetic_field` field

**Files**: `src/radpair/_types.py:149`, `tests/test_types.py:178-187`,
`tests/test_smoke.py:17`, `tests/conftest.py`

**Problem**: `Experiment.magnetic_field` exists solely for the deprecated
`multicore` decorator in `_wrappers.py`, which splits the field axis across
processes. After removing `_wrappers.py` (Issue 2), no code reads this
field. `do_simulation` uses `experiment.B_z`; `do_simulation_multicore`
also uses `experiment.B_z`. The `__post_init__` copy logic
(`_types.py:152-153`) is wasted work on every simulation.

**Action**:
1. Remove `magnetic_field` field and its `__post_init__` logic from
   `Experiment` in `_types.py`.
2. Remove `EXP_REQUIRED_ATTRS` or drop `"magnetic_field"` from it in
   `test_smoke.py:17`.
3. Remove `TestExperiment::test_magnetic_field_defaults_to_bz_copy` and
   `TestExperiment::test_explicit_magnetic_field` from `test_types.py`.
4. Update `docs/source/examples.rst` unit-conventions table (remove
   `magnetic_field` row if present).

**Estimated removal**: ~30 lines.

---

## Issue 4 — Eliminate pipeline duplication in `core.py`

**Files**: `src/radpair/core.py:97-145` (`_run_pipeline_through_intensities`),
`src/radpair/core.py:148-188` (`_gaussian_summation_worker`)

**Problem**: `_run_pipeline_through_intensities` **re-implements stages 1–7**
of `do_simulation` plus the flattening/interpolation logic from
`assemble_spectrum` (`functions.py:989`). This is ~50 lines of duplicated
pipeline code. Similarly, `_gaussian_summation_worker` duplicates the
per-orientation list construction from `gaussian_summation`
(`functions.py:930-946`).

The duplication exists because `do_simulation` doesn't expose its
intermediate state — it goes all the way to a spectrum in one call.

**Action**: Refactor `do_simulation` to use a shared `_run_pipeline` helper
that returns the flattened `(fields, intensities, widths, weights,
width_gauss, field_axis, max_chunk_mb)` tuple. Both `do_simulation` and
`do_simulation_multicore` call it:

```python
def _run_pipeline(spinsystem, experiment, simopt) -> tuple:
    """Run stages 1–8 and return (fields_flat, intensities_flat,
    widths_flat, weights, width_gauss, field_axis, max_chunk_mb)."""
    # ... stages 1-7 (same as do_simulation) ...
    # ... flatten + interpolate (same as assemble_spectrum) ...
    return fields_flat, intensities_flat, widths_flat, weights, ...

def do_simulation(...) -> np.ndarray:
    data = _run_pipeline(spinsystem, experiment, simopt)
    return np.nan_to_num(gaussian_summation(*data))

def do_simulation_multicore(...) -> np.ndarray:
    data = _run_pipeline(spinsystem, experiment, simopt)
    # ... distribute chunks across Pool ...
```

This makes `assemble_spectrum` unnecessary as a separate function — its
flattening/interpolation logic moves into `_run_pipeline`, and the
Gaussian summation is called directly.

**Estimated removal**: ~90 lines (eliminate `_run_pipeline_through_intensities`
and `_gaussian_summation_worker`, simplify `assemble_spectrum` or fold it
into `_run_pipeline`).

---

## Issue 5 — Consolidate S1–S7 system definitions

**Files**: `tests/generate_reference_spectra.py:64-286` (7 `_make_S*`
functions, ~220 lines), `examples/_systems.py:54-262` (7 `make_S*`
functions, ~210 lines)

**Problem**: The S1–S7 spinsystem definitions are **duplicated verbatim**
between these two files. Any parameter change requires updating both files
or they drift out of sync.

**Action**:
1. `generate_reference_spectra.py` should import `SYSTEMS`,
   `make_experiment`, `make_simopt` from `examples/_systems.py` (the
   canonical source).
2. Delete all `_make_S*` functions, `_make_experiment`, `_make_simopt`,
   `_zero_A`, `_zero_frame`, and constants (`FREQ_MW`, `N_POINTS`, etc.)
   from `generate_reference_spectra.py`.
3. The script becomes ~40 lines: import, loop over `SYSTEMS`, run
   `do_simulation`, save `.npz`.

**Estimated removal**: ~230 lines from `generate_reference_spectra.py`.

---

## Issue 6 — Inline trivial one-liner wrapper functions

**Files**: `src/radpair/functions.py`

**Problem**: Several functions are trivial wrappers around a single
expression, adding indirection without abstraction value:

| Function | Lines | Body | Used by |
|----------|-------|------|---------|
| `vector_product_combinations` | 148-176 | `a[:, None] * b` = `np.outer(a, b)` | `compute_hyperfine_combinations` only |
| `rescale_array` | 267-292 | `arr / arr.sum() * norm` | `get_normalized_Pascal` only |
| `get_D_diag` | 295-321 | `np.array([D-E, D+E, -2*D])` | `build_tensors` only |
| `spinsystem_field_names` (`_types.py:186`) | 186-188 | `[f.name for f in fields(Spinsystem)]` | `test_smoke.py`, `test_simulation_regression.py` only |

Each has its own test class (4, 3, 3 tests respectively) testing trivial
behavior.

**Action**:
1. Inline `vector_product_combinations` → `np.outer(mI_vector,
   a_projections[i])` in `compute_hyperfine_combinations`.
2. Inline `rescale_array` → `pascal_line / pascal_line.sum()` in
   `get_normalized_Pascal`.
3. Inline `get_D_diag` → `np.array([Sys.D - Sys.E, Sys.D + Sys.E, -2 *
   Sys.D])` in `build_tensors`.
4. Replace `spinsystem_field_names()` calls with
   `[f.name for f in dataclasses.fields(Spinsystem)]` or remove
   entirely (see Issue 8).
5. Remove the corresponding test classes:
   `TestVectorProductCombinations`, `TestRescaleArray`, `TestGetDDiag`
   from `test_functions.py`.

**Estimated removal**: ~120 lines (functions + docstrings + tests).

---

## Issue 7 — Reorganize `functions.py` (1078 lines) into focused modules

**Files**: `src/radpair/functions.py`

**Problem**: `functions.py` is a catch-all that mixes:
- **Physics math**: `tensor_rotation`, `compute_resonance_fields`,
  `compute_intensities`, `get_D_diag` (if not inlined)
- **Combinatorics**: `get_multiplicity`, `get_generalized_Pascal`,
  `get_normalized_Pascal`
- **Unit conversion**: `MHz_2_T`, `prepare_spinsystem`
- **Pipeline stages**: `setup_orientation_grid`, `build_tensors`,
  `rotate_tensors`, `compute_hyperfine_combinations`, `assemble_spectrum`
- **Infrastructure**: `_get_available_ram`, `_compute_chunk_size`,
  `gaussian_summation`

This makes it hard to navigate and violates the single-responsibility
principle.

**Action**: Split into focused modules (only if Issues 1-6 are done first,
since they reduce the content significantly):

```
src/radpair/
  _types.py        — dataclasses (unchanged)
  core.py          — do_simulation, do_simulation_multicore, _run_pipeline
  hamiltonian.py   — tensor_rotation, compute_resonance_fields,
                     compute_intensities (physics)
  hyperfine.py     — get_multiplicity, get_generalized_Pascal,
                     get_normalized_Pascal, compute_hyperfine_combinations
  pipeline.py      — prepare_spinsystem, setup_orientation_grid,
                     build_tensors, rotate_tensors, assemble_spectrum
                     (or _run_pipeline)
  summation.py     — gaussian_summation, _compute_chunk_size,
                     _get_available_ram, MHz_2_T
```

Alternatively, keep a single `functions.py` but reorganize sections with
clear headers and move the infrastructure code (`_get_available_ram`,
`_compute_chunk_size`) into `core.py` since they're only used by the
multicore path.

**Note**: This is the most disruptive issue. It should be done last and
only if the module is still too large after Issues 1-6. The functional
style is correct — no OOP needed. The question is purely file
organization.

**Estimated removal**: 0 lines (reorganization, not removal), but
significantly improves navigability.

---

## Issue 8 — Remove redundant tests

**Files**: `tests/test_smoke.py`, `tests/test_simulation_regression.py`,
`tests/test_stages.py`

**Problem**: Several test classes test things that are guaranteed by
construction or duplicate other tests:

1. **`test_smoke.py:30-51`** — `test_spinsystem_has_required_attrs`,
   `test_experiment_has_required_attrs`, `test_simopt_has_required_attrs`:
   These check `hasattr` on dataclass instances. A dataclass always has
   its declared fields — these tests can never fail unless the dataclass
   definition itself is broken (which would break every other test too).

2. **`test_simulation_regression.py:84-119`** —
   `TestDoSimulationInvariants`: Tests shape, real-valued, no-NaNs, no-Infs.
   These overlap with `test_smoke.py:59-83` which tests the same
   properties on the same fixtures.

3. **`test_stages.py:515-590`** — `TestPipelineMatchesDoSimulation`:
   Manually calls all 8 stages and compares to `do_simulation`. This is
   valuable as an integration test, but it duplicates ~40 lines of
   pipeline code in the test. After Issue 4 (shared `_run_pipeline`), this
   test could be simplified to comparing `_run_pipeline` output to
   `do_simulation` output, or removed entirely if stage-level tests
   already cover the contract.

**Action**:
1. Delete `test_spinsystem_has_required_attrs`,
   `test_experiment_has_required_attrs`, `test_simopt_has_required_attrs`
   from `test_smoke.py`.
2. Delete `TestDoSimulationInvariants` from
   `test_simulation_regression.py` (covered by `test_smoke.py` and the
   reference spectrum tests).
3. Simplify or remove `TestPipelineMatchesDoSimulation` from
   `test_stages.py` after Issue 4.

**Estimated removal**: ~100 lines.

---

## Issue 9 — Simplify `setup_orientation_grid` defensive code

**Files**: `src/radpair/functions.py:499-503`

**Problem**: The function has a `try/except IndexError` block to handle
the case where `theta_angles` is a scalar (0-d array):

```python
try:
    theta_angles.shape[0]
except IndexError:
    theta_angles = np.array([theta_angles])
    phi_angles = np.array([phi_angles])
```

This is fragile — it relies on `IndexError` from indexing a 0-d array.
`eprbase.grid.Grid.get_grid()` returns a 2-D array, so this branch is
likely never triggered in practice.

**Action**: Replace with `np.atleast_1d`:

```python
theta_angles = np.atleast_1d(theta_angles)
phi_angles = np.atleast_1d(phi_angles)
```

Or remove the block entirely if `get_grid` always returns a 2-D array
(verify with eprbase source).

**Estimated removal**: 2 lines saved, but eliminates a confusing pattern.

---

## Issue 10 — Fix outdated documentation

**Files**: `docs/source/introduction.rst:24`,
`docs/source/introduction.rst:44-51`

**Problem**:
1. Line 24: "up to five anisotropic nuclei groups" — the 5-nuclei limit was
   removed; the pipeline now supports an arbitrary number.
2. Lines 44-51: Module overview table lists `radpair.classes` and
   `radpair._wrappers`, both of which are slated for removal (Issues 1-2).
3. The description of `radpair.classes` says "Matrix class for tensor
   rotation and Core class representing a group of chemically equivalent
   nuclei" — these are no longer part of the pipeline.

**Action**:
1. Change "up to five" to "an arbitrary number of".
2. Remove rows for `radpair.classes` and `radpair._wrappers` after
   Issues 1-2.
3. Add row for any new modules created in Issue 7.

---

## Issue 11 — Clean up `SimpleNamespace` remnants in benchmarks

**Files**: `benchmarks/_common.py:11`, `benchmarks/_common.py:57-66`

**Problem**: `_common.py` imports `SimpleNamespace` and uses it in type
hints for `count_active_nuclei`, `bench_call`, `bench_call_multicore`,
and `bench`. The actual objects passed are `Spinsystem`, `Experiment`,
and `SimulationOptions` dataclasses. This is misleading.

**Action**:
1. Remove `from types import SimpleNamespace`.
2. Change type hints from `SimpleNamespace` to the actual types
   (`Spinsystem`, `Experiment`, `SimulationOptions`).

**Estimated removal**: 0 lines, but eliminates dead import and misleading
types.

---

## Summary

| Issue | Lines removed | Breaking? |
|-------|--------------|-----------|
| 1. Remove `classes.py` | ~351 | Yes (removes public classes) |
| 2. Remove `_wrappers.py` | ~300 | Yes (removes public decorators) |
| 3. Remove `Experiment.magnetic_field` | ~30 | Yes (API change) |
| 4. Eliminate pipeline duplication | ~90 | No (internal refactor) |
| 5. Consolidate S1–S7 definitions | ~230 | No (test script change) |
| 6. Inline trivial wrappers | ~120 | Yes (removes public functions) |
| 7. Reorganize `functions.py` | 0 | Yes (module paths change) |
| 8. Remove redundant tests | ~100 | No |
| 9. Simplify `setup_orientation_grid` | ~2 | No |
| 10. Fix outdated docs | 0 | No |
| 11. Clean up benchmark types | 0 | No |
| **Total** | **~1 223** | |

### Recommended order

1. **Issues 1+2** (remove dead modules) — do together, biggest win
2. **Issue 3** (remove `magnetic_field`) — follows from Issue 2
3. **Issue 5** (consolidate S1–S7) — independent, easy
4. **Issue 6** (inline trivial wrappers) — independent, easy
5. **Issue 4** (eliminate pipeline duplication) — after 1-3 settle
6. **Issue 8** (remove redundant tests) — after 1-6, since tests for
   removed code are already gone
7. **Issues 9-11** (cleanup) — anytime
8. **Issue 7** (reorganize modules) — last, only if needed
