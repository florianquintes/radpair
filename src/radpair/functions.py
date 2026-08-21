"""
(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from functools import lru_cache

import numpy as np
import scipy.constants as constant


def tensor_rotation(
    tensor: np.array, phi: np.array, theta: np.array, psi: np.array = None
) -> np.array:
    r"""
    Algorithm:
    Euler transformation using y-convention. The euler matrix is set up
    with the given angles. Phi and theta is necessary, psi is optional.
    The euler matrix O of the SO(3) Group in y-convention is set up in already
    multiplicated form. Than the orthogonal similarity transformation of the
    tensor T is carried out:

    .. math::

        T' = O^{-1}\cdot T \cdot O

    with:

    .. math::

        O^{-1} = O^T


    Parameters
    ----------
    tensor : np.array
        Tensor which should be rotated using Euler transformation
        (y-convention).
    phi : float
        Phi angle in radian for transformation.
    theta : float
        Theta angle in radian for transformation.
    psi : float, optional
        Psi angle in radian for transformation. The default is None.

    Returns
    -------
    rotatedTensor : np.array
        Rotated tensor.

    """
    if psi is None:
        psi = np.zeros(phi.size)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    costhet = np.cos(theta)
    sinthet = np.sin(theta)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    eulermatrix = np.zeros((phi.size, 3, 3))
    eulermatrix[:, 0, 0] = cosphi * costhet * cospsi - sinphi * sinpsi
    eulermatrix[:, 0, 1] = -cosphi * costhet * sinpsi - sinphi * cospsi
    eulermatrix[:, 0, 2] = cosphi * sinthet
    eulermatrix[:, 1, 0] = sinphi * costhet * cospsi + cosphi * sinpsi
    eulermatrix[:, 1, 1] = -sinphi * costhet * sinpsi + cosphi * cospsi
    eulermatrix[:, 1, 2] = sinphi * sinthet
    eulermatrix[:, 2, 0] = -sinthet * cospsi
    eulermatrix[:, 2, 1] = sinthet * sinpsi
    eulermatrix[:, 2, 2] = costhet

    if tensor.ndim == 2:
        rot_1 = np.einsum("ij, ajk -> aik", tensor, eulermatrix)
    elif tensor.ndim == 3:
        rot_1 = np.einsum("aij, ajk -> aik", tensor, eulermatrix)
    else:
        raise ValueError("Tensor has wrong dimensions!")
    rotatedTensor = np.einsum("aji, ajk -> aik", eulermatrix, rot_1)

    return rotatedTensor


@lru_cache
def get_multiplicity(spin: float) -> int:
    """
    Get the multiplicity of a core with spin I:

    .. math::
        M = 2S + 1.

    Parameters
    ----------
    spin : float
        Magnetic spin of a quantum particle.

    Returns
    -------
    multiplicity : int
        Multiplicity of a quantum particle with a given spin in an external
        magnetic field due to Zeeman splitting.

    """
    if spin < 0.0:
        raise ValueError("Spin can't be negative!")
    if (spin % 0.5) > 1e-3:
        raise ValueError("Spin must be divisible by 0.5!")

    multiplicity = int(2 * spin + 1)

    return multiplicity


def vector_product_combinations(a: np.array, b: np.array) -> np.array:
    """
    Multiply each value of a given numpy.array with each value of another one.

    ::

        [1 2 3] * [4 5 6]

    will give you:
    ::

            [ 4  5  6]
        M = [ 8 10 12]
            [12 15 18]

    Parameters
    ----------
    a : np.array
        First array. Used as column vector.
    b : np.array
        Second array. Used as line vector.

    Returns
    -------
    matrix : np.array
        Matrix containing the products of all permutations of all values from
        one vector with the other.

    """
    matrix = a.reshape((1, a.shape[0])).T * b

    return matrix


@lru_cache
def get_generalized_Pascal(number: int, spin: float) -> np.array:
    """
    Get a generalized Pascal triangle for a given number of atoms and spin.

    (c) Stephan Rein
    Modified by: Florian Quintes

    Parameters
    ----------
    number : int
        Number of coupling atoms.
    spin : float
        Magnetic spin of a quantum particle.

    Returns
    -------
    np.array
        Array containing the relativ intensities of 'number' coupling atoms
        with magnetic spin 'spin'.

    """
    if number < 0.0:
        raise ValueError("Number can't be negativ!")
    if spin < 0.0:
        raise ValueError("Spin can't be negativ!")
    if not isinstance(number, int):
        raise TypeError("Number must be a natural number!")
    if (spin % 0.5) > 1e-3:
        raise ValueError("Spin must be divisible by 0.5!")

    n = int(number)
    if n == 0:
        return np.ones(1)
    else:
        s0 = int(2 * spin * n + 1)
        A = np.zeros((n, s0))
        A[0, 0 : int(2 * spin + 1)] = 1
        I2 = 2 * spin
        for i in range(1, n):
            for j in range(s0):
                if j + I2 >= s0:
                    ub = int(s0 - 1)
                else:
                    ub = int(j + I2)
                if j - I2 < 0:
                    lb = 0
                else:
                    lb = int(j - I2)
                A[i, j] = np.sum(A[i - 1, lb : ub - int(2 * spin) + 1])
        return A[n - 1, :]


@lru_cache
def get_normalized_Pascal(number: int, spin: float) -> np.array:
    """
    Get a generalized Pascal triangle for a given number of atoms and spin. The
    sum of all intensities is 1..

    Parameters
    ----------
    number : int
        Number of coupling atoms.
    spin : float
        Magnetic spin of a quantum particle.

    Returns
    -------
    pascal_line : np.array
        Array containing the relativ intensities of 'number' coupling atoms
        with magnetic spin 'spin'.

    """
    pascal_line = get_generalized_Pascal(number, spin)
    pascal_line = rescale_array(pascal_line)

    return pascal_line


def rescale_array(arr: np.array, norm: float = 1.0) -> np.array:
    """
    Rescale a given 1d array to a given norm. Default is 1.

    ::

        [1. 3. 4.] -> [0.125 0.375, 0.5]

    Parameters
    ----------
    arr : np.array
        Numpy 1d-array which will be rescaled.
    norm : float, optional
        New sum of all array values. The default is 1..

    Returns
    -------
    scaled_arr : np.array
        Rescaled array.

    """
    if arr.sum() == 0:
        raise ZeroDivisionError("Can't rescale an array with sum 0!")

    scaled_arr = arr / arr.sum() * norm

    return scaled_arr


def get_D_diag(D: float, E: float) -> np.array:
    """
    Get the diagonal elements of the resulting D-Tensor.

    ::

            [D-E    0   0]
        D = [   0 D+E   0]
            [   0    0 -2*D]

    Parameters
    ----------
    D : float
        ZFS parameter D.
    E : float
        ZFS parameter E.

    Returns
    -------
    D_diag : np.array
        Array containing the diagonal elements of the D-Tensor.

    """
    D_diag = np.array([D - E, D + E, -2 * D])

    return D_diag


def MHz_2_T(nu: float or np.array, g_tensor: np.array) -> float:
    r"""
    Convert a given MHz value to a corresponding tesla value with respect to
    the given g-Tensor.

    .. math::

        \\nu_{\mathrm{Tesla}} = \\frac{\\nu_{\mathrm{MHz}}}
        {g_{\mathrm{iso}}\cdot\\mu_{B}\cdot10^{-3}}

    Parameters
    ----------
    nu : float or np.array
        Frequency in MHz.
    g_tensor : np.array
        g-Tensor of the electron.

    Returns
    -------
    nu_tesla : float
        Frequency in Tesla.

    """
    if not (g_tensor > 0).all():
        raise ValueError("All values of the g-Tensor need to be higher than 0!")

    mu_b = constant.value("Bohr magneton in Hz/T")
    g_iso = g_tensor.sum() / 3
    nu_tesla = 1e6 * nu / (g_iso * mu_b)

    return nu_tesla


def sphere_fibonacci_grid_points(ng: int) -> np.ndarray:
    """
    Calculate Fibonacci spiral gridpoints on a sphere.

    Parameters
    ----------
    ng : int
        Number of points that shall be calculated.

    Returns
    -------
    xg : np.ndarray
        Coordinates of the desired number of grid points. The three cartesian
        coordinates are given. The shape of the array is 3xng.

    Licensing
    ---------

      This code is distributed under the GNU LGPL license.
      https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid.py

    Modified
    --------

      15 May 2015

    Author
    ------

      John Burkardt

    Reference
    ---------

      Richard Swinbank, James Purser,
      Fibonacci grids: A novel approach to global modelling,
      Quarterly Journal of the Royal Meteorological Society,
      Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    theta = np.zeros(ng)
    sphi = np.zeros(ng)
    cphi = np.zeros(ng)

    for i in range(ng):
        i2 = 2 * i - (ng - 1)
        theta[i] = 2.0 * np.pi * float(i2) / phi
        sphi[i] = float(i2) / float(ng)
        cphi[i] = np.sqrt(float(ng + i2) * float(ng - i2)) / float(ng)

    xg = np.zeros((ng, 3))

    for i in range(ng):
        xg[i, 0] = cphi[i] * np.sin(theta[i])
        xg[i, 1] = cphi[i] * np.cos(theta[i])
        xg[i, 2] = sphi[i]

    return xg


def cartesian2spherical(xyz: np.ndarray) -> np.ndarray:
    """
    Convert a set of three cartesian coordinates (x, y and z) to a set of
    three spherical coordinates (r, theta and phi).

    Parameters
    ----------
    xyz : np.ndarray
        This array contains n sets of cartesian coordinates x, y and z.
        Therefore, it describes n Points and its shape is 3xn.

    Returns
    -------
    rtp : np.ndarray
        This array contains the transformed sets of xyz. First argument is
        the radius r, second and third are the angles theta and phi.

    """
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    rtp = np.array([r, theta, phi])

    return rtp


@lru_cache
def get_fibonacci_sphere(points: int) -> tuple[np.array, np.array]:
    """
    Get a fibonacci sphere in spherical coordinates. Either by loading a saved
    one or by calculating a new one.

    Parameters
    ----------
    points : int
        Number of points of the fibonacci sphere.

    Returns
    -------
    theta : np.array
        Array with theta values of the fibonacci sphere.
    phi : np.array
        Array with phi values of the fibonacci sphere.

    """
    xyz = sphere_fibonacci_grid_points(points)
    _, theta, phi = cartesian2spherical(xyz)

    return theta, phi
