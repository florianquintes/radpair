Examples
========

This page provides worked examples showing how to use ``radpair`` to
simulate cw-EPR spectra of spin-correlated radical pairs.  All examples
use typed dataclass objects (``Spinsystem``, ``Experiment``,
``SimulationOptions`` from :mod:`radpair._types`) to define the spin
system, experiment, and simulation options.

Every code block on this page is self-contained — you can copy and paste
it into a ``.py`` file and run it directly.

.. note::

   The examples require `Matplotlib <https://pypi.org/project/matplotlib/>`_
   for plotting, which is included in the dev dependencies.  Install with
   ``uv sync --dev`` or ``pip install matplotlib``.

.. contents:: On this page
   :local:
   :depth: 2

Common setup
------------

All seven example spectra share the same X-band experiment and simulation
options.  The following code is needed for every example below:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from radpair._types import Experiment, SimulationOptions, Spinsystem
   from radpair.core import do_simulation

   # Experiment: X-band EPR, 9.75 GHz, field sweep 344–350 mT
   field_axis = np.linspace(344.0, 350.0, 500)

   experiment = Experiment(
       B_z=field_axis,
       freq_mw=9.75e9,            # Hz
   )

   # Simulation options: 12 orientation grid knots, single core
   simopt = SimulationOptions(
       knots=12,
       refinement=1,
       cpu_cores=1,
       max_chunk_mb=None,  # auto-detect chunk size from available RAM
   )

   _zero = np.array([0.0, 0.0, 0.0])

The :func:`~radpair.core.do_simulation` function takes three arguments
— ``spinsystem``, ``experiment``, ``simopt`` — and returns a
real-valued intensity array matching the shape of ``experiment.B_z``::

   intensity = do_simulation(spinsystem, experiment, simopt)

Unit conventions
~~~~~~~~~~~~~~~~

+------------------------------+----------------------------------------------+
| Attribute                    | Unit                                         |
+==============================+==============================================+
| ``Experiment.B_z``           | milliTesla                                   |
+------------------------------+----------------------------------------------+
| ``Experiment.freq_mw``       | Hz                                           |
+------------------------------+----------------------------------------------+
| ``Spinsystem.A_tensors``     | MHz (list of 3-element arrays)               |
+------------------------------+----------------------------------------------+
| ``Spinsystem.D``, ``E``      | MHz                                          |
+------------------------------+----------------------------------------------+
| ``Spinsystem.J_ex``          | MHz                                          |
+------------------------------+----------------------------------------------+
| ``Spinsystem.width_gauss``   | milliTesla (despite the name)                |
+------------------------------+----------------------------------------------+
| ``Spinsystem.*_frame``       | radians (Euler angles [alpha, beta, gamma])  |
+------------------------------+----------------------------------------------+
| ``Spinsystem.g1``, ``g2``    | dimensionless (3-element diagonal)           |
+------------------------------+----------------------------------------------+
| ``Spinsystem.nuclei_n``      | list of equivalent-nuclei counts (int >= 0)  |
+------------------------------+----------------------------------------------+
| ``Spinsystem.nuclei_I``      | list of nuclear spins (float, 1/2 multiples) |
+------------------------------+----------------------------------------------+
| ``Spinsystem.donor_list``    | 0-indexed list of donor nuclei positions     |
+------------------------------+----------------------------------------------+
| ``Spinsystem.acceptor_list`` | 0-indexed list of acceptor nuclei positions  |
+------------------------------+----------------------------------------------+

Example 1: Bare radical pair (S1)
---------------------------------

The simplest anisotropic radical pair: two radicals with anisotropic
g-tensors, nonzero ZFS (*D* = 8 MHz, *E* = 1.5 MHz) and exchange
(*J* = 2 MHz), but no resolved hyperfine couplings.

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0023, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[_zero, _zero, _zero, _zero, _zero],
       nuclei_n=[0, 0, 0, 0, 0],
       nuclei_I=[0.0, 0.0, 0.0, 0.0, 0.0],
       D=8.0, E=1.5, J_ex=2.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[_zero, _zero, _zero, _zero, _zero],
       donor_list=[], acceptor_list=[],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C0")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S1 — Bare radical pair (0 nuclei)")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s1.png
   :alt: S1 — Bare radical pair spectrum
   :width: 100%

Example 2: Single donor nucleus, isotropic (S2)
------------------------------------------------

One donor proton with isotropic g-tensors and isotropic hyperfine
coupling (*A* = 1.5 MHz).  The acceptor is "silent" (no nuclei,
isotropic g).  A small exchange (*J* = 0.1 MHz) is present; ZFS is zero.

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0030, 2.0030, 2.0030]),
       g2=np.array([2.0090, 2.0090, 2.0090]),
       A_tensors=[np.array([1.5, 1.5, 1.5]), _zero, _zero, _zero, _zero],
       nuclei_n=[1, 0, 0, 0, 0],
       nuclei_I=[0.5, 0.0, 0.0, 0.0, 0.0],
       D=0.0, E=0.0, J_ex=0.1,
       width_gauss=0.05,
       g1_frame=_zero, g2_frame=_zero, D_frame=_zero,
       A_frames=[_zero, _zero, _zero, _zero, _zero],
       donor_list=[0], acceptor_list=[],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C1")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S2 — Single donor ¹H, isotropic")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s2.png
   :alt: S2 — Single donor ¹H spectrum
   :width: 100%

Example 3: Two anisotropic nuclei, mixed donor/acceptor (S3)
-------------------------------------------------------------

A more realistic system: the donor carries one ¹H with anisotropic
hyperfine coupling (*A* = [5, 3, 4] MHz), and the acceptor carries two
equivalent ¹⁴N nuclei (*I* = 1, *n* = 2, *A* = [2.5, 1.8, 3.2] MHz).
Both g-tensors are anisotropic, and ZFS (*D* = 8, *E* = 1.5 MHz) and
exchange (*J* = 3 MHz) are nonzero.  Several Euler frames are nonzero,
so tensors are rotated relative to the lab frame.

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([5.0, 3.0, 4.0]),
           np.array([2.5, 1.8, 3.2]),
           _zero, _zero, _zero,
       ],
       nuclei_n=[1, 2, 0, 0, 0],
       nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
       D=8.0, E=1.5, J_ex=3.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.0, 0.0]),
           _zero, _zero, _zero,
       ],
       donor_list=[0], acceptor_list=[1],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C2")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S3 — 2 nuclei (donor ¹H + acceptor 2×¹⁴N), anisotropic")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s3.png
   :alt: S3 — Two anisotropic nuclei spectrum
   :width: 100%

Example 4: Donor/acceptor swap (S3 vs S4)
------------------------------------------

Systems S3 and S4 are identical except that the donor and acceptor nuclei
assignments are swapped.  This demonstrates how the same nuclei produce
different spectra depending on which radical they belong to — the
MHz-to-Tesla conversion uses different g-tensors for donor vs. acceptor
nuclei.

.. code-block:: python

   # S3: donor ¹H + acceptor 2×¹⁴N
   spinsystem_S3 = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([5.0, 3.0, 4.0]),
           np.array([2.5, 1.8, 3.2]),
           _zero, _zero, _zero,
       ],
       nuclei_n=[1, 2, 0, 0, 0],
       nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
       D=8.0, E=1.5, J_ex=3.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.0, 0.0]),
           _zero, _zero, _zero,
       ],
       donor_list=[0], acceptor_list=[1],
   )

   # S4: same tensors, swapped assignments
   spinsystem_S4 = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([5.0, 3.0, 4.0]),
           np.array([2.5, 1.8, 3.2]),
           _zero, _zero, _zero,
       ],
       nuclei_n=[1, 2, 0, 0, 0],
       nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
       D=8.0, E=1.5, J_ex=3.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.0, 0.0]),
           _zero, _zero, _zero,
       ],
       donor_list=[1], acceptor_list=[0],
   )

   intensity_S3 = do_simulation(spinsystem_S3, experiment, simopt)
   intensity_S4 = do_simulation(spinsystem_S4, experiment, simopt)

   print(f"S3 sum: {intensity_S3.sum():.10e}")
   print(f"S4 sum: {intensity_S4.sum():.10e}")
   print(f"Max abs diff: {np.max(np.abs(intensity_S3 - intensity_S4)):.2e}")

   fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
   axes[0].plot(field_axis, intensity_S3, lw=0.8, color="C2")
   axes[0].set_ylabel("Intensity")
   axes[0].set_title("S3 — donor ¹H + acceptor 2×¹⁴N")
   axes[0].axhline(0, color="gray", lw=0.5, ls="--")
   axes[1].plot(field_axis, intensity_S4, lw=0.8, color="C5")
   axes[1].set_ylabel("Intensity")
   axes[1].set_title("S4 — acceptor ¹H + donor 2×¹⁴N (swap)")
   axes[1].axhline(0, color="gray", lw=0.5, ls="--")
   axes[2].plot(field_axis, intensity_S3 - intensity_S4, lw=0.8, color="C7")
   axes[2].set_ylabel("S3 − S4")
   axes[2].set_xlabel("Magnetic field $B_z$ (mT)")
   axes[2].set_title("Difference")
   axes[2].axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s4.png
   :alt: S4 — Donor/acceptor swap spectrum
   :width: 100%

Example 5: Three nuclei with methyl group (S5)
----------------------------------------------

Donor with a methyl group (3 equivalent ¹H, *n* = 3, *A* = [4.5, 3.0, 5.5] MHz)
plus a single ¹H (*A* = [1.2, 1.5, 0.8] MHz), acceptor with 2 equivalent ¹⁴N
(*A* = [2.5, 1.8, 3.2] MHz).  All tensors are anisotropic, with full ZFS
(*D* = 10, *E* = 2 MHz) and exchange (*J* = 5 MHz).

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([4.5, 3.0, 5.5]),
           np.array([1.2, 1.5, 0.8]),
           np.array([2.5, 1.8, 3.2]),
           _zero, _zero,
       ],
       nuclei_n=[3, 1, 2, 0, 0],
       nuclei_I=[0.5, 0.5, 1.0, 0.0, 0.0],
       D=10.0, E=2.0, J_ex=5.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.1, 0.0, 0.0]),
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.1, 0.0]),
           _zero, _zero,
       ],
       donor_list=[0, 1], acceptor_list=[2],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C4")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S5 — 3 nuclei (methyl 3×¹H + ¹H + 2×¹⁴N)")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s5.png
   :alt: S5 — Three nuclei with methyl group
   :width: 100%

Example 6: Four nuclei with ³⁵Cl (S6)
-------------------------------------

Donor with one ¹H (*A* = [1.5, 1.5, 1.5] MHz, isotropic), acceptor with
¹⁴N (*A* = [3.0, 2.0, 4.0] MHz) + ³⁵Cl (*I* = 3/2, *A* = [8.0, 5.0, 10.0] MHz)
+ 2×¹H (*A* = [0.8, 1.2, 0.6] MHz).  The donor g-tensor is isotropic while
the acceptor g-tensor is anisotropic.  ZFS (*D* = 6, *E* = 1 MHz) and
exchange (*J* = 1.5 MHz).

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0030, 2.0030, 2.0030]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([1.5, 1.5, 1.5]),
           np.array([3.0, 2.0, 4.0]),
           np.array([8.0, 5.0, 10.0]),
           np.array([0.8, 1.2, 0.6]),
           _zero,
       ],
       nuclei_n=[1, 1, 1, 2, 0],
       nuclei_I=[0.5, 1.0, 1.5, 0.5, 0.0],
       D=6.0, E=1.0, J_ex=1.5,
       width_gauss=0.05,
       g1_frame=_zero,
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           _zero,
           np.array([0.1, 0.0, 0.0]),
           np.array([0.0, 0.2, 0.0]),
           _zero, _zero,
       ],
       donor_list=[0], acceptor_list=[1, 2, 3],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C5")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S6 — 4 nuclei (¹H + ¹⁴N + ³⁵Cl + 2×¹H)")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s6.png
   :alt: S6 — Four nuclei with ³⁵Cl
   :width: 100%

Example 7: Five nuclei, maximum complexity (S7)
-----------------------------------------------

All five nuclei groups are active (3 donor + 2 acceptor), with anisotropic
g-tensors, full ZFS (*D* = 10, *E* = 2 MHz), exchange (*J* = 5 MHz), and
multiple nuclear spins (*I* = 1/2, 1, 3/2) and multiplicities (*n* = 1, 2).

.. code-block:: python

   spinsystem = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([4.5, 3.0, 5.5]),
           np.array([2.0, 1.5, 2.8]),
           np.array([6.0, 4.0, 8.0]),
           np.array([1.5, 1.0, 2.0]),
           np.array([3.0, 2.0, 4.0]),
       ],
       nuclei_n=[1, 2, 1, 1, 2],
       nuclei_I=[0.5, 1.0, 1.5, 0.5, 0.5],
       D=10.0, E=2.0, J_ex=5.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.1, 0.0, 0.0]),
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.1, 0.0]),
           _zero, _zero,
       ],
       donor_list=[0, 1, 2], acceptor_list=[3, 4],
   )

   intensity = do_simulation(spinsystem, experiment, simopt)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity, lw=0.8, color="C3")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S7 — 5 nuclei (3 donor + 2 acceptor), maximum complexity")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   fig.tight_layout()
   plt.show()

.. image:: _static/spectrum_s7.png
   :alt: S7 — Five nuclei, maximum complexity
   :width: 100%

Multi-core simulation
---------------------

For computationally expensive systems, :func:`~radpair.core.do_simulation_multicore`
runs the analytic pipeline once on the main process, then distributes
the Gaussian peak summation across ``cpu_cores`` worker processes.
Each worker handles a chunk of peaks, keeping per-process memory
bounded by ``max_chunk_mb``.  The result is numerically identical to
the single-core call.

.. code-block:: python

   from radpair._types import SimulationOptions
   from radpair.core import do_simulation, do_simulation_multicore

   # Reuse the S3 spinsystem from Example 3
   spinsystem = Spinsystem(
       g1=np.array([2.0020, 2.0040, 2.0060]),
       g2=np.array([2.0080, 2.0100, 2.0120]),
       A_tensors=[
           np.array([5.0, 3.0, 4.0]),
           np.array([2.5, 1.8, 3.2]),
           _zero, _zero, _zero,
       ],
       nuclei_n=[1, 2, 0, 0, 0],
       nuclei_I=[0.5, 1.0, 0.0, 0.0, 0.0],
       D=8.0, E=1.5, J_ex=3.0,
       width_gauss=0.05,
       g1_frame=np.array([0.1, 0.2, 0.0]),
       g2_frame=np.array([0.0, 0.3, 0.1]),
       D_frame=np.array([0.2, 0.1, 0.0]),
       A_frames=[
           np.array([0.0, 0.1, 0.0]),
           np.array([0.1, 0.0, 0.0]),
           _zero, _zero, _zero,
       ],
       donor_list=[0], acceptor_list=[1],
   )

   simopt_multicore = SimulationOptions(
       knots=12,
       refinement=1,
       cpu_cores=4,        # use 4 worker processes (0 = auto-detect)
       max_chunk_mb=None,  # auto-detect chunk size from available RAM
   )

   intensity_sc = do_simulation(spinsystem, experiment, simopt)
   intensity_mc = do_simulation_multicore(spinsystem, experiment, simopt_multicore)

   print(f"Single-core sum: {intensity_sc.sum():.10e}")
   print(f"Multi-core  sum: {intensity_mc.sum():.10e}")
   print(f"Max abs diff:    {np.max(np.abs(intensity_sc - intensity_mc)):.2e}")

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(field_axis, intensity_mc, lw=0.8, color="C4", label="4 cores")
   ax.set_xlabel("Magnetic field $B_z$ (mT)")
   ax.set_ylabel("Intensity (arb. u.)")
   ax.set_title("S3 — Multi-core simulation (cpu_cores = 4)")
   ax.axhline(0, color="gray", lw=0.5, ls="--")
   ax.legend()
   fig.tight_layout()
   plt.show()

.. _chunked-summation:

Chunked Gaussian summation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The EPR spectrum is assembled by summing Gaussian line shapes for every
(orientation, hyperfine combination, transition) triplet.  The total
number of peaks grows as ``n_orientations × n_combinations × 4`` and can
easily reach hundreds of thousands or millions.  Internally,
``eprbase`` builds a dense ``float32`` array of shape
``(n_peaks, n_field)`` for the Gaussian evaluation — at 500 field points
this consumes ``n_peaks × 2 KB`` of memory.

The ``max_chunk_mb`` option controls how much memory a single chunk may
use:

- **``None`` (default)** — auto-detect from available RAM (targets 25%
  of free memory per chunk).  This works well on most systems.
- **``0`` or negative** — disable chunking entirely.  All peaks are
  processed in one pass.  Fastest for small systems but may cause
  out-of-memory errors for large ones.
- **Positive integer** — maximum memory in megabytes per chunk.
  Smaller values reduce peak memory at the cost of more iterations.
  Useful when running multiple simulations in parallel or when the
  auto-detection is too aggressive.

.. code-block:: python

   # Explicit 512 MB per chunk
   simopt = SimulationOptions(
       knots=25,
       max_chunk_mb=512,
   )

   # No limit (use with caution on large systems)
   simopt = SimulationOptions(
       knots=25,
       max_chunk_mb=0,
   )

The same option applies to both :func:`~radpair.core.do_simulation` and
:func:`~radpair.core.do_simulation_multicore`.  In multicore mode each
worker respects the limit independently, so the total memory
requirement is approximately ``cpu_cores × max_chunk_mb``.

.. figure:: _static/spectrum_overview.png
   :alt: Overview of all 7 spectra
   :width: 80%

   Overview of all seven example spectra (S1–S7), showing the increasing
   spectral complexity as more nuclei groups are added.

Available example scripts
-------------------------

All scripts are in the ``examples/`` directory and can be run individually
with ``uv run python examples/<script_name>.py``.  They use shared system
definitions from ``examples/_systems.py``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Description
   * - ``minimal_radical_pair.py``
     - S1: bare radical pair, 0 nuclei
   * - ``single_nucleus.py``
     - S2: single donor ¹H, isotropic
   * - ``anisotropic_two_nuclei.py``
     - S3: 2 nuclei, mixed donor/acceptor, anisotropic
   * - ``donor_acceptor_swap.py``
     - S3 vs S4: donor/acceptor swap comparison
   * - ``full_five_nuclei.py``
     - S7: 5 nuclei, maximum complexity
   * - ``multicore.py``
     - Multi-core simulation demo
   * - ``generate_all_plots.py``
     - Generate all 7 spectra plots for the docs
