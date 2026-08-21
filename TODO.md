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

### Issue: Tests for `radpair/functions.py`

Cover every function:

- `tensor_rotation` — 2D and 3D tensor input, identity rotation, known-angle rotation, invalid dimensions.
- `get_multiplicity` — spin 0, 0.5, 1, 1.5; negative spin error; non-0.5-multiple error.
- `vector_product_combinations` — correct shape and values for small vectors.
- `get_generalized_Pascal` — n=0 returns `[1]`, known cases (e.g., n=2 spin=0.5 → `[1,2,1]`), n=1 general spin, error cases.
- `get_normalized_Pascal` — sums to 1.0, delegates to `get_generalized_Pascal`.
- `rescale_array` — rescaling correctness, zero-sum error.
- `get_D_diag` — known D/E values produce expected diagonal.
- `MHz_2_T` — known conversion, all-positive g-tensor check, negative-g error.
- `sphere_fibonacci_grid_points` — correct shape `(ng, 3)`, points on unit sphere.
- `cartesian2spherical` — round-trip or known conversions, output shape `(3, n)`.
- `get_fibonacci_sphere` — returns `(theta, phi)` of correct length, cached behavior.

### Issue: Tests for `radpair/classes.py`

Cover every class and method:

- `Matrix.__init__` — stores matrix, `matrix_rot` is `None`.
- `Matrix.matrot` — rotation produces correct shape; identity rotation preserves matrix.
- `Matrix.get_hyperfine_projection` — known tensor produces expected projection.
- `Core.__init__` — stores number/spin, computes `total_spin`, validates negative/non-int/invalid-spin errors.
- `Core.set_hyperfine_matrix` — correct shape and values for small input.
- `Core.get_magnetic_spin_vector` — correct linspace for various spins.

### Issue: Tests for `radpair/_wrappers.py`

Cover every decorator:

- `timer` — returns original result, prints runtime line (cap-sys).
- `function_benchmark` — runs niter times, prints statistics, does not return the original result.
- `multicore` — splits field axis correctly, `cpu_cores=0` resolves to `cpu_count()`, output matches single-core concatenation.

### Issue: Tests for `radpair/core.py` — `do_simulation`

- Test with a minimal 1-nucleus system (only `A1`, others zeroed out via `Core(0, 0)`).
- Test with a 5-nucleus system exercising all cores.
- Verify output shape matches `Exp.B_z` (or `Exp.magnetic_field`).
- Verify output is real-valued and contains no NaNs (`np.nan_to_num` is called, but assert).
- Test `interpolation_mode` path (`SimOpt.refinement > 1`).

### Issue: Tests for `radpair/core.py` — `do_simulation_multicore`

- Verify multicore output matches single-core output (same inputs).
- Test with `SimOpt.cpu_cores = 0` (auto-detect).
- Test with `SimOpt.cpu_cores = 1` (degenerate case).

### Issue: Create reference test spectra and end-to-end comparison tests

- Create `tests/data/` directory.
- Simulate several representative systems using the current code and save the output spectra as `.npy` or `.npz` files in `tests/data/`.
- Suggested reference cases:
  1. Minimal system — 1 nucleus, 1 donor, no ZFS.
  2. Full system — 5 nuclei, mixed donor/acceptor, nonzero D/E/J.
  3. System with interpolation enabled (`refinement > 1`).
  4. System with only donor nuclei.
  5. System with only acceptor nuclei.
- Add a test module `tests/test_simulation_regression.py` that loads each reference spectrum and asserts `np.allclose(result, reference, rtol=...)`.
- Document the procedure to regenerate reference data (script or fixture with `--regenerate` flag).

---

## Milestone 3 — Codebase Refactoring (Pre-structure)

### Issue: ~~Add type annotations to all public and private functions~~ ✅ Done

- Created `src/radpair/_types.py` with `@runtime_checkable` `Protocol` classes (`Spinsystem`, `Experiment`, `SimulationOptions`) documenting all required attributes.
- `core.py` — `do_simulation` and `do_simulation_multicore` now use the Protocol types instead of `object`; parameters renamed to lowercase (`spinsystem`, `experiment`, `simopt`).
- `classes.py` — `Matrix.__init__` parameter typed as `np.ndarray`; `matrot` returns `-> None`; `get_hyperfine_projection` returns `-> np.ndarray`; `Core.set_hyperfine_matrix` returns `-> None`; `Core.get_magnetic_spin_vector` returns `-> None`; `matrix_rot` attribute typed as `np.ndarray | None`.
- `functions.py` — All `np.array` type hints replaced with `np.ndarray`; `psi` parameter typed as `np.ndarray | None`; return types added to all functions; `MHz_2_T` uses `float | np.ndarray` for both parameter and return.
- `_wrappers.py` — `timer` uses PEP 695 type parameters (`[**P, R]`); `function_benchmark` returns `Callable[..., None]`; `multicore` returns `Callable[..., np.ndarray]`; inner `multicore_wrapper` uses Protocol types for parameters.

### Issue: Eliminate anti-patterns

- `core.py:387` — `if True:` block is a no-op; remove it.
- `core.py:66-68` — Loop over `range(1, 6)` with hardcoded `A1`–`A5` access via `vars(Sys)`. This should be data-driven, not index-hardcoded.
- `core.py:70-98` — Repeated per-attribute scaling (`Sys.A1 *= …` through `Sys.A5 *= …`). Replace with a loop.
- `core.py:124-159` — Three near-identical `tilt_alpha`/`tilt_beta`/`tilt_gamma` arrays built manually. Generate programmatically.
- `core.py:169-191` — Repeated `np.diag` and `cl.Matrix` calls for `a1`–`a5`. Loop over a list.
- `core.py:208-217` — Five identical `if core_type["N"] == 0:` blocks. Loop.
- `core.py:221-236` — Five identical `.matrot()` calls and five `get_hyperfine_projection()` calls. Loop.
- `core.py:244-248` — Five identical `set_hyperfine_matrix` calls. Loop.
- `core.py:264-295` — Five-deep nested `for` loop over `mI_1`…`mI_5`. Use `itertools.product`.
- `core.py:366-369` — Four width variables assigned identically. Use a list/array.
- `classes.py:81` — `get_hyperfine_projection` uses magic index `2` for z-column; document or make explicit.

### Issue: ~~Replace `float or np.array` type annotation~~ ✅ Done

- `functions.py` — `nu: float or np.array` replaced with `nu: float | np.ndarray`; return type updated to `float | np.ndarray`.

### Issue: Consolidate unit-conversion magic numbers

- `core.py:85` — `tesang = 1.75880474e8` is a hardcoded gyromagnetic ratio. Extract as a named constant (or use `scipy.constants`).
- `core.py:98` — `4 * np.log(2)` is the FWHM-to-sigma factor for a Gaussian; name it.

### Issue: ~~Improve docstrings to numpydoc / Sphinx-compatible style~~ ✅ Done

- All functions and classes now have complete numpydoc docstrings with `Parameters`, `Returns`, and where applicable `Raises`, `Notes`, `Examples` sections.
- `.. math::` directives fixed to use proper LaTeX (e.g. `O^{\mathsf{T}}` instead of `O^{-1}`).
- `classes.py` — Docstring parameter mismatch fixed (`diag` → `mat`).
- `core.py` — Parameters standardized to lowercase (`spinsystem`, `experiment`, `simopt`); docstrings reference the Protocol classes.
- `functions.py` — Added `Examples` section to `vector_product_combinations`; `Raises` sections added to `tensor_rotation`, `get_multiplicity`, `get_generalized_Pascal`, `rescale_array`, `MHz_2_T`; `Notes` section added to `sphere_fibonacci_grid_points`.
- `classes.py` — `Raises` sections added to `Core.__init__`; class-level docstrings improved with `Attributes` sections.
- `_wrappers.py` — Module docstring added; all decorator docstrings improved.
- `api_reference.rst` updated to include `radpair._types` automodule directive.
- Docs build passes with zero warnings.

### Issue: Remove dead / placeholder code

- `core.py:387-391` — `if True:` block with redundant `.copy()` calls.
- `core.py:385` — `transition = np.zeros(...)` is created, copied, reshaped, but never populated with meaningful data; verify intent.

### Issue: ~~Add `LICENSE` file~~ ✅ Done

- `LICENSE` file added to repo root.
- Referenced in `README.md` via link to `LICENSE`.

---

## Milestone 4 — Future Release: Complete Restructuring

> Depends on Milestones 1–3. Should be a separate major/minor version bump.

### Issue: Replace dynamic-attribute interface with typed dataclasses

- `do_simulation` currently accesses `Sys.g1`, `Sys.A1`, `vars(Sys)["A1"]`, etc. dynamically.
- Introduce `@dataclass` definitions for `Spinsystem`, `Exp`, `SimOpt` with explicit fields.
- This enables IDE support, static analysis, and self-documenting code.

### Issue: Remove hardcoded 5-nuclei limit

- `core.py` assumes exactly 5 hyperfine tensors (`A1`–`A5`) and 5 `Core` objects.
- Refactor to accept a variable-length list of nuclei groups.
- Update `Matrix`/`Core` usage and the nested hyperfine summation loop accordingly.
- This is a breaking API change — plan for major version bump.

### Issue: Break `do_simulation` into composable functions

- `do_simulation` is ~360 lines doing unit conversion, tensor setup, rotation, resonance-field calculation, interpolation, and spectrum assembly in one function.
- Extract stages: `prepare_spinsystem`, `build_tensors`, `rotate_tensors`, `compute_resonance_fields`, `compute_intensities`, `assemble_spectrum`.
- Each stage should be independently testable.

### Issue: Improve parallelization

- Current `multicore` wrapper splits the field axis and uses `multiprocessing.Pool`.
- Evaluate `concurrent.futures.ProcessPoolExecutor` or vectorized batch approaches.
- Ensure `deepcopy` overhead is minimized (Sys is deep-copied per core but is read-only).
- Consider shared-memory for large arrays.

### Issue: Performance audit

- Profile `do_simulation` with representative inputs.
- The 5-deep nested loop (`core.py:264-295`) builds Python lists then converts to arrays — vectorize with `np.meshgrid` or `itertools.product` + array conversion.
- `tensor_rotation` (`functions.py:13`) allocates `eulermatrix` with per-element assignment; consider vectorized construction.
- `sphere_fibonacci_grid_points` (`functions.py:318`) uses Python loops; vectorize.
