"""Math helper functions for the radpair package.

Provides unit conversion, tensor rotation, Pascal-triangle generation,
Fibonacci-sphere grid points, and spherical-coordinate conversion.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from functools import lru_cache

import numpy as np
import scipy.constants as constant


def tensor_rotation(
    tensor: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray | None = None,
) -> np.ndarray:
    r"""Rotate a tensor (or batch of tensors) via Euler transformation.

    The Euler matrix *O* of the SO(3) group is set up in y-convention
    (already in multiplied form) and the orthogonal similarity
    transformation is carried out:

    .. math::

        T' = O^{\mathsf{T}} \cdot T \cdot O

    where :math:`O^{-1} = O^{\mathsf{T}}` (orthogonality).

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to be rotated.  A single 2-D array of shape ``(3, 3)``
        or a batch of 2-D arrays of shape ``(N, 3, 3)``.
    phi : np.ndarray
        Phi (Euler) angles in radians, shape ``(N,)``.
    theta : np.ndarray
        Theta (Euler) angles in radians, shape ``(N,)``.
    psi : np.ndarray, optional
        Psi (Euler) angles in radians, shape ``(N,)``.  If ``None``,
        zeros are used (default).

    Returns
    -------
    np.ndarray
        Rotated tensor(s).  Shape matches the input ``tensor`` except
        that the leading dimension becomes ``N`` (the number of angles).

    Raises
    ------
    ValueError
        If ``tensor`` does not have 2 or 3 dimensions.
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
    rotated_tensor = np.einsum("aji, ajk -> aik", eulermatrix, rot_1)

    return rotated_tensor


@lru_cache
def get_multiplicity(spin: float) -> int:
    r"""Return the multiplicity of a particle with spin *S*.

    .. math::

        M = 2S + 1

    Parameters
    ----------
    spin : float
        Magnetic spin quantum number (must be non-negative and a
        multiple of 0.5).

    Returns
    -------
    int
        Multiplicity (number of Zeeman levels).

    Raises
    ------
    ValueError
        If ``spin`` is negative or not a multiple of 0.5.
    """
    if spin < 0.0:
        raise ValueError("Spin can't be negative!")
    if (spin % 0.5) > 1e-3:
        raise ValueError("Spin must be divisible by 0.5!")

    multiplicity = int(2 * spin + 1)

    return multiplicity


def vector_product_combinations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the outer product of two 1-D arrays.

    Given two vectors, returns a matrix containing all pairwise products.

    Parameters
    ----------
    a : np.ndarray
        First 1-D array (used as column vector).
    b : np.ndarray
        Second 1-D array (used as row vector).

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(a), len(b))`` containing the products
        of all permutations of values from ``a`` with values from ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> vector_product_combinations(np.array([1, 2, 3]), np.array([4, 5, 6]))
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])
    """
    matrix = a.reshape((1, a.shape[0])).T * b

    return matrix


@lru_cache
def get_generalized_Pascal(number: int, spin: float) -> np.ndarray:
    """Compute a generalized Pascal triangle for ``number`` nuclei of spin ``spin``.

    Returns the relative intensities of the hyperfine lines for a group
    of ``number`` chemically equivalent nuclei with magnetic spin
    ``spin``.

    (c) Stephan Rein
    Modified by: Florian Quintes

    Parameters
    ----------
    number : int
        Number of chemically equivalent nuclei (>= 0).
    spin : float
        Magnetic spin quantum number (>= 0, multiple of 0.5).

    Returns
    -------
    np.ndarray
        Array of relative intensities.  For ``number == 0`` returns
        ``[1]``.

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
def get_normalized_Pascal(number: int, spin: float) -> np.ndarray:
    """Compute a normalized generalized Pascal triangle (sum = 1).

    Wraps :func:`get_generalized_Pascal` and rescales the result so
    that all intensities sum to 1.

    Parameters
    ----------
    number : int
        Number of chemically equivalent nuclei (>= 0).
    spin : float
        Magnetic spin quantum number (>= 0, multiple of 0.5).

    Returns
    -------
    np.ndarray
        Normalized array of relative intensities summing to 1.
    """
    pascal_line = get_generalized_Pascal(number, spin)
    pascal_line = rescale_array(pascal_line)

    return pascal_line


def rescale_array(arr: np.ndarray, norm: float = 1.0) -> np.ndarray:
    """Rescale a 1-D array so that its elements sum to ``norm``.

    Parameters
    ----------
    arr : np.ndarray
        1-D array to be rescaled.
    norm : float, optional
        Target sum of all array values (default is 1.0).

    Returns
    -------
    np.ndarray
        Rescaled array.

    Raises
    ------
    ZeroDivisionError
        If the array sums to zero.
    """
    if arr.sum() == 0:
        raise ZeroDivisionError("Can't rescale an array with sum 0!")

    scaled_arr = arr / arr.sum() * norm

    return scaled_arr


def get_D_diag(D: float, E: float) -> np.ndarray:
    r"""Return the diagonal elements of the ZFS *D*-tensor.

    The diagonal tensor is constructed from the zero-field splitting
    parameters *D* and *E* as:

    .. math::

        \mathrm{diag}(D) = \begin{pmatrix} D - E & 0 & 0 \\
                0 & D + E & 0 \\
                0 & 0 & -2D \end{pmatrix}

    Parameters
    ----------
    D : float
        ZFS parameter *D*.
    E : float
        ZFS parameter *E*.

    Returns
    -------
    np.ndarray
        Array of shape ``(3,)`` containing the diagonal elements.
    """
    D_diag = np.array([D - E, D + E, -2 * D])

    return D_diag


def MHz_2_T(nu: float | np.ndarray, g_tensor: np.ndarray) -> float | np.ndarray:
    r"""Convert a frequency in MHz to the corresponding magnetic field in Tesla.

    The conversion uses the isotropic g-value and the Bohr magneton:

    .. math::

        B = \frac{\nu_{\mathrm{MHz}} \times 10^{6}}
        {g_{\mathrm{iso}} \cdot \mu_{B} \times 10^{-3}}

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency (or array of frequencies) in MHz.
    g_tensor : np.ndarray
        g-tensor diagonal elements, shape ``(3,)``.  All values must be
        positive.

    Returns
    -------
    float or np.ndarray
        Magnetic field in Tesla (scalar if ``nu`` is scalar, array
        otherwise).

    Raises
    ------
    ValueError
        If any element of ``g_tensor`` is not positive.
    """
    if not (g_tensor > 0).all():
        raise ValueError("All values of the g-Tensor need to be higher than 0!")

    mu_b = constant.value("Bohr magneton in Hz/T")
    g_iso = g_tensor.sum() / 3
    nu_tesla = 1e6 * nu / (g_iso * mu_b)

    return nu_tesla


def sphere_fibonacci_grid_points(ng: int) -> np.ndarray:
    """Calculate Fibonacci-spiral grid points on a unit sphere.

    Parameters
    ----------
    ng : int
        Number of grid points to generate.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the grid points, shape ``(ng, 3)``.

    Notes
    -----
    This code is distributed under the GNU LGPL license.
    Original source:
    https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid.py

    Modified 15 May 2015 by John Burkardt.

    Reference
    ---------
    Richard Swinbank, James Purser, "Fibonacci grids: A novel approach
    to global modelling", Quarterly Journal of the Royal Meteorological
    Society, Volume 132, Number 619, July 2006 Part B, pages 1769-1793.
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
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates ``(x, y, z)`` for *n* points, shape
        ``(n, 3)``.

    Returns
    -------
    np.ndarray
        Spherical coordinates ``(r, theta, phi)`` for each point,
        shape ``(3, n)``.  Here *theta* is the polar angle (from the
        z-axis) and *phi* is the azimuthal angle (from the x-axis).
    """
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    rtp = np.array([r, theta, phi])

    return rtp


@lru_cache
def get_fibonacci_sphere(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Fibonacci sphere in spherical coordinates.

    Parameters
    ----------
    points : int
        Number of grid points on the sphere.

    Returns
    -------
    theta : np.ndarray
        Polar angles (from the z-axis) of the grid points.
    phi : np.ndarray
        Azimuthal angles (from the x-axis) of the grid points.
    """
    xyz = sphere_fibonacci_grid_points(points)
    _, theta, phi = cartesian2spherical(xyz)

    return theta, phi
