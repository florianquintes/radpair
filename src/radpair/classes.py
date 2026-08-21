"""Helper classes for the radpair package.

Provides :class:`Matrix` (tensor rotation) and :class:`Core` (nuclei
group) used by :func:`radpair.core.do_simulation`.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

import numpy as np

import radpair.functions as fun


class Matrix:
    """Represent an ``n×n`` matrix and provide rotation operations.

    Attributes
    ----------
    matrix : np.ndarray
        Tensor in its initial (eigenbasis) state.
    matrix_rot : np.ndarray or None
        Rotated matrices (``None`` until :meth:`matrot` is called).
    """

    def __init__(self, mat: np.ndarray) -> None:
        """Initialize a matrix in its eigenbasis.

        Parameters
        ----------
        mat : np.ndarray
            The matrix (typically a 3×3 diagonal tensor) in its
            eigenbasis.
        """
        self.matrix = mat
        self.matrix_rot: np.ndarray | None = None

    def matrot(self, phi: np.ndarray, theta: np.ndarray, psi: np.ndarray = 0.0) -> None:
        """Rotate the matrix using Euler transformation (y-convention).

        The result is stored in ``self.matrix_rot`` without modifying
        ``self.matrix``.

        Parameters
        ----------
        phi : np.ndarray
            Phi angles in radians for the transformation.
        theta : np.ndarray
            Theta angles in radians for the transformation.
        psi : np.ndarray, optional
            Psi angles in radians for the transformation (default 0.0).
        """
        self.matrix_rot = fun.tensor_rotation(self.matrix, phi, theta, psi)

    def get_hyperfine_projection(self) -> np.ndarray:
        r"""Compute the effective hyperfine coupling for each orientation.

        Only applicable to hyperfine tensors.  The effective coupling is
        the Euclidean norm of the third column of the rotated matrix:

        .. math::

            A_{\mathrm{eff}} = \sqrt{A_{xz}^2 + A_{yz}^2 + A_{zz}^2}

        Returns
        -------
        np.ndarray
            Effective hyperfine couplings, shape ``(N,)``, where *N* is
            the number of orientations.  The *i*-th element corresponds
            to the *i*-th orientation.
        """
        return np.sqrt((self.matrix_rot[:, :, 2] ** 2).sum(axis=1))


class Core:
    """Represent a group of chemically equivalent nuclei.

    Attributes
    ----------
    number : int
        Number of coupling nuclei.
    spin : float
        Magnetic spin of one nucleus.
    total_spin : float
        Total magnetic spin of all nuclei (``number * spin``).
    pascal : np.ndarray
        Intensity distribution from a (normalized) Pascal triangle.
    mI_len : int
        Length of the ``mI_vector``.
    mI_vector : np.ndarray
        Array of all magnetic spin projection values.
    hyperfine_matrix : np.ndarray
        Matrix of hyperfine couplings for every ``m_I`` value.
    """

    def __init__(self, number: int, spin: float) -> None:
        """Initialize a group of chemically equivalent nuclei.

        Parameters
        ----------
        number : int
            Number of coupling nuclei (>= 0).
        spin : float
            Magnetic spin quantum number (>= 0, multiple of 0.5).

        Raises
        ------
        ValueError
            If ``number`` or ``spin`` is negative, or ``spin`` is not a
            multiple of 0.5.
        TypeError
            If ``number`` is not an integer.
        """
        if number < 0.0:
            raise ValueError("Number can't be negativ!")
        if spin < 0.0:
            raise ValueError("Spin can't be negativ!")
        if not isinstance(number, int):
            raise TypeError("Number must be a natural number!")
        if (spin % 0.5) > 1e-3:
            raise ValueError("Spin must be divisible by 0.5!")

        self.number = number
        self.spin = spin
        self.total_spin = self.number * self.spin
        self.get_magnetic_spin_vector()
        self.pascal = fun.get_normalized_Pascal(self.number, self.spin)
        self.mI_len = self.mI_vector.size

    def set_hyperfine_matrix(self, hyperfine_arr: np.ndarray) -> None:
        """Set up the hyperfine matrix from coupling constants.

        Computes the outer product of the magnetic spin vector with the
        hyperfine coupling array and stores the result in
        ``self.hyperfine_matrix``.

        Parameters
        ----------
        hyperfine_arr : np.ndarray
            Array of hyperfine coupling constants for each orientation.
        """
        self.hyperfine_matrix = fun.vector_product_combinations(
            self.mI_vector, hyperfine_arr
        )

    def get_magnetic_spin_vector(self) -> None:
        """Compute and store the magnetic spin projection vector.

        Populates ``self.mI_vector`` with evenly spaced values from
        ``-total_spin`` to ``+total_spin``.
        """
        multi = fun.get_multiplicity(self.total_spin)
        m_I = self.total_spin
        self.mI_vector = np.linspace(-m_I, m_I, multi).astype(np.float32)
