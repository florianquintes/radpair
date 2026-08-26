Contributing
============

Thank you for contributing to ``radpair``. The project is a Python package
for analytic simulation of cw-EPR spectra of singlet-born spin-correlated
radical pairs.

Project Structure
-----------------

The package uses a ``src`` layout. All package code lives under
``src/radpair``:

* ``core.py`` — entry points ``do_simulation`` and
  ``do_simulation_multicore``.
* ``functions.py`` — math helpers (unit conversion, Pascal triangles, tensor
  rotation, grid generation).

``__init__.py`` is empty; there are no public re-exports. Import directly
from submodules. The package depends on `eprbase <https://pypi.org/project/eprbase/>`_
for ``grid.Grid``, ``spectra.Spectra``, and ``interpolation.Interpolator``.

Development Setup
-----------------

The project requires Python 3.13 or newer. The repository uses ``uv`` for
dependency and environment management.

Clone the repository and change into its directory:

.. code-block:: console

   git clone https://github.com/florianquintes/radpair.git
   cd radpair

Install the locked development environment from the repository root:

.. code-block:: console

   uv sync --dev

The package uses a ``src`` layout. Application code belongs under
``src/radpair`` and tests belong under ``tests``.

Branches
--------

Create a focused branch for each change and keep unrelated changes separate.
The repository does not currently document a required branch naming scheme.

Pull Requests
-------------

Pull requests should explain the change and include the relevant validation
results. Before opening a pull request, run the checks that apply to the
change:

.. code-block:: console

   uv run ruff format --check .
   uv run ruff check .
   uv run pytest
   uv build

If documentation is changed, also build the documentation as described in
the :doc:`development` page. Mention known baseline failures or warnings
rather than presenting them as regressions.

Code Style
----------

Follow the existing Python and reStructuredText style in the surrounding
files. Keep importable Python modules under ``src/radpair`` and add tests
under ``tests``.

Ruff is the configured formatting and code-quality tool, using its default
rules:

.. code-block:: console

   uv run ruff format --check .
   uv run ruff check .

Docstrings should follow the `numpydoc <https://numpydoc.readthedocs.io/>`_
style with LaTeX math where applicable. Type annotations are required on all
functions.

Documentation
-------------

Documentation source files are under ``docs/source``. The Sphinx
configuration reads the package version dynamically via
``importlib.metadata.version("radpair")``.

Build the documentation from the repository root:

.. code-block:: console

   uv run sphinx-build -b html docs/source docs/build/html

The generated HTML is written to ``docs/build/html``. Add a title to every
document included by the root ``index.rst`` so Sphinx can include it in the
table of contents.
