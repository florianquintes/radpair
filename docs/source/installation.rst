Installation
============

Requirements
------------

``radpair`` requires Python 3.13 or newer. ``uv`` or ``pip`` is required to
install the package.

The package depends on `eprbase <https://pypi.org/project/eprbase/>`_,
`NumPy <https://pypi.org/project/numpy/>`_, and
`SciPy <https://pypi.org/project/scipy/>`_.

Installation Commands
---------------------

The following commands cover the common installation use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 
     - uv
     - pip
   * - User
     - ``uv add radpair``
     - ``pip install radpair``
   * - Developer
     - ``uv sync --dev``
     - ``pip install -e .``
