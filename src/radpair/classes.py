#!/usr/bin/env python3
"""
(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""
import numpy as np

import radpair.functions as fun


class Matrix:
    """
    Get an object representing a nxn-matrix in its initial state.

    Attributes
    ----------
    matrix : np.array
        Tensor in its initial state.
    matrix_rot : np.array
        Rotated matrices.

    Methods
    -------
    matrot(phi, theta, psi=0.)
        Rotate the matrix using Euler transformation (y-convention).
    """

    def __init__(self, mat: np.array) -> None:
        """
        Initialize a matrix in his eigenbasis.

        Parameters
        ----------
        diag : np.array
            Diagonal elements of the matrix.

        Returns
        -------
        None.

        """
        self.matrix = mat
        self.matrix_rot = None

    def matrot(self, phi: float, theta: float, psi: float = 0):
        """
        Rotate the matrix using Euler transformation (y-convention) without
        changing self.matrix.

        Parameters
        ----------
        phi : float
            Phi angle in radian for transformation.
        theta : float
            Theta angle in radian for transformation.
        psi : float, optional
            Psi angle in radian for transformation. The default is 0..

        Returns
        -------
        None.

        """
        self.matrix_rot = fun.tensor_rotation(self.matrix, phi, theta, psi)

    def get_hyperfine_projection(self) -> np.array:
        r"""
        Get the hyperfine projection. Only needed for hyperfine tensors.

        .. math::
            A_{\mathrm{eff}} = \sqrt{A_{xz}^2 + A_{yz}^2 + A_{zz}^2}

        Returns
        -------
        np.array (N,)
            All effective hyperfine couplings. i'th hyperfine coupling
            corresponds to the i'th orientation.

        """
        return np.sqrt((self.matrix_rot[:, :, 2] ** 2).sum(axis=1))


class Core:
    """
    Get an object representing a given number of chemically equivalent nuclei.

    Attributes
    ----------
    number : int
        Number of coupling nuclei.
    spin : float
        Magnetic spin of one nuclei.
    total_spin : float
        Magnetic spin of all nuclei.
    pascal : np.array
        Intensity distribution given by a Pascal triangle.
    mI_len : int
        length of the mI_vector.
    mI_vector : np.array
        Object of class 'Vector' with all m_I values.
    hyperfine_matrix :
        Matrix with all hyperfine couplings for every m_I value.

    Methods
    -------
    set_hyperfine_matrix(hyperfine_arr)
        Set up the hyperfine matrix using the hyperfine couplings and
        self.mI_vector.
    get_magnetic_spin_vector()
        Get the mI_vector.

    """

    def __init__(self, number: int, spin: float) -> None:
        """
        Initialize an object representing a given number of chemically
        equivalent nuclei.

        Parameters
        ----------
        number : int
            Number of coupling atoms.
        spin : float
            Magnetic spin of a quantum particle.

        Returns
        -------
        None.

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

    def set_hyperfine_matrix(self, hyperfine_arr: np.array):
        """
        Set up the hyperfine matrix using the hyperfine array and the magnet
        spin vector.

        Parameters
        ----------
        hyperfine_arr : np.array
            Array with the hyperfine coupling constants.

        Returns
        -------
        None.

        """
        self.hyperfine_matrix = fun.vector_product_combinations(
            self.mI_vector, hyperfine_arr
        )

    def get_magnetic_spin_vector(self):
        """Get the magnetic spin vector as an object of class 'Vector'."""
        multi = fun.get_multiplicity(self.total_spin)
        m_I = self.total_spin
        self.mI_vector = np.linspace(-m_I, m_I, multi).astype(np.float32)
