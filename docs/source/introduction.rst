Introduction
============

``radpair`` is a Python-based simulation routine for continuous-wave
electron paramagnetic resonance (cw-EPR) spectra of singlet-born
spin-correlated radical pairs. It uses an analytic solution of the
Hamiltonian and a pseudo-secular approximation for the hyperfine couplings,
enabling fast computation without numerical diagonalisation of the spin
Hamiltonian.

The implementation is based on a Fortran 77 program originally developed for
a phd thesis. The Python port preserves the physical model while
modernising the code structure and leveraging NumPy and SciPy for vectorised
array operations.

.. note::

   A reference to the original thesis and relevant publications will be added
   here in a future update.

Capabilities
------------

* Simulation of cw-EPR spectra for radical pairs with an arbitrary number
  of anisotropic nuclei groups.
* Each nuclei group can be assigned as donor or acceptor.
* Support for the zero-field splitting (ZFS) tensor with parameters *D* and
  *E*, exchange interaction *J*, and anisotropic *g*-tensors.
* Optional interpolation for refined orientation grids.
* Single-core and multi-core execution via ``multiprocessing.Pool``.

Module Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``radpair.core``
     - Entry points for single-core (``do_simulation``) and multi-core
       (``do_simulation_multicore``) simulation.
   * - ``radpair.functions``
     - Math helpers: unit conversion, Pascal triangles, tensor rotation,
       and spherical grid generation.
