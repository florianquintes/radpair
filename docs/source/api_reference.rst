API Reference
=============

The API reference documents every importable module in ``radpair``.
The package is organised into six modules: two public entry points
(:mod:`radpair.core`, :mod:`radpair._types`) and four internal modules
that implement the simulation pipeline stages.

.. contents:: Sections
   :local:
   :depth: 1

Core simulation functions
-------------------------

The :mod:`radpair.core` module provides the two entry points that users
call directly.  :func:`~radpair.core.do_simulation` runs the full
single-core pipeline; :func:`~radpair.core.do_simulation_multicore`
distributes the Gaussian summation stage across worker processes.

.. automodule:: radpair.core
   :members:
   :no-index:

Data types
----------

The :mod:`radpair._types` module defines three dataclasses that the user
instantiates and passes to the simulation functions.  Each dataclass
validates its fields in ``__post_init__`` and raises ``ValueError`` or
``TypeError`` for invalid inputs.

.. automodule:: radpair._types
   :members:
   :no-index:

Hamiltonian
-----------

The :mod:`radpair.hamiltonian` module implements the analytic solution
of the spin Hamiltonian.  It provides:

* **Tensor rotation** — Euler-angle similarity transformation of
  diagonal tensors into the lab and orientation frames.
* **Unit conversion** — converting MHz to Tesla using the isotropic
  g-value and Bohr magneton.
* **Resonance fields** — solving the four allowed transitions of the
  spin-correlated radical pair for each orientation and hyperfine
  combination.
* **Line intensities** — computing the absorptive/emissive intensity
  pattern from the phase angles of the quantum beat frequencies.

.. automodule:: radpair.hamiltonian
   :members:
   :no-index:

Hyperfine combinatorics
-----------------------

The :mod:`radpair.hyperfine` module provides the combinatorial
machinery for hyperfine line patterns:

* **Multiplicity** — :math:`2S+1` for a given nuclear spin *S*.
* **Generalised Pascal triangles** — relative intensities for groups
  of chemically equivalent nuclei with arbitrary spin.
* **Hyperfine combinations** — outer-product enumeration of magnetic
  spin projections across all nuclei groups, producing the sum and
  difference hyperfine matrices used by the Hamiltonian solver.

.. automodule:: radpair.hyperfine
   :members:
   :no-index:

Pipeline stages
---------------

The :mod:`radpair.pipeline` module implements the first four stages of
the simulation pipeline:

1. **Unit conversion** (:func:`~radpair.pipeline.prepare_spinsystem`) —
   deep-copy the spin system and convert all parameters from physical
   units (MHz, mT, Hz) to internal angular-frequency units.
2. **Orientation grid** (:func:`~radpair.pipeline.setup_orientation_grid`)
   — create a coarse (and optionally fine) orientation grid for
   spherical integration, using ``eprbase.grid.Grid``.
3. **Tensor construction** (:func:`~radpair.pipeline.build_tensors`) —
   build diagonal 3×3 tensors from the spin-system parameters and
   extract the Euler frame angles.
4. **Tensor rotation** (:func:`~radpair.pipeline.rotate_tensors`) —
   tilt each tensor from its eigenframe into the lab frame, then rotate
   for every orientation on the integration grid.  Returns the
   zz-projections used by the Hamiltonian solver.

.. automodule:: radpair.pipeline
   :members:
   :no-index:

Gaussian summation
------------------

The :mod:`radpair.summation` module implements the final stage of the
pipeline: summing Gaussian line shapes via chunked evaluation.  Chunking
bounds peak memory usage by splitting the peaks into batches, each of
which produces a dense ``float32`` array of shape
``(n_peaks_in_chunk, n_field)`` via ``eprbase.spectra.Spectra``.

.. automodule:: radpair.summation
   :members:
   :no-index:
