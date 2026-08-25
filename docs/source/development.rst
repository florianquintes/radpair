Development
===========

Development Setup
-----------------

The project requires Python 3.13 or newer and uses ``uv`` for dependency and
environment management. From the repository root, install the development
environment with:

.. code-block:: console

   uv sync --dev

Running Tests
-------------

Run the test suite from the repository root:

.. code-block:: console

   uv run pytest

To run a single test:

.. code-block:: console

   uv run pytest tests/test_foo.py::test_name

Code Formatting and Linting
---------------------------

The project uses Ruff for both formatting and linting (no separate formatter
is configured). Check formatting and linting with:

.. code-block:: console

   uv run ruff format --check .
   uv run ruff check .

To auto-fix formatting and linting issues:

.. code-block:: console

   uv run ruff format .
   uv run ruff check . --fix

Build the Documentation
-----------------------

Build the HTML documentation from the repository root:

.. code-block:: console

   uv run sphinx-build -b html docs/source docs/build/html

The generated documentation is written to ``docs/build/html``.

Build the Package
-----------------

Build the distribution packages with:

.. code-block:: console

   uv build

Continuous Integration
----------------------

CI runs three workflows on every push and pull request:

* **CI** (``ci.yml``) — runs ``uv run pytest``.
* **Format and lint** (``format.yml``) — runs ``ruff format --check`` and
  ``ruff check``.
* **Documentation** (``docs.yml``) — builds the Sphinx docs and deploys to
  GitHub Pages on ``main`` and ``develop``.

Release
-------

Tags matching ``v*.*.*`` trigger the **Publish to PyPI** workflow
(``publish.yml``), which builds the package with ``uv build`` and publishes
to PyPI via trusted publishing.
