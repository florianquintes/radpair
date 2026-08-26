"""Hyperfine combinatorics: multiplicities, Pascal triangles, and combinations.

Provides the combinatorial machinery for computing hyperfine line
patterns from groups of chemically equivalent nuclei, including
generalised Pascal triangles and the outer-product combinations of
magnetic spin projections.

(c) M. Sc. Theresia Quintes, M. Sc. Florian Quintes, 2019-2026

@author: Thresia Quintes, Florian Quintes
"""

from functools import lru_cache
from itertools import product

import numpy as np

from radpair._types import Spinsystem


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
    pascal_line = pascal_line / pascal_line.sum()

    return pascal_line


def compute_hyperfine_combinations(
    Sys: Spinsystem,
    a_projections: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Compute sum and difference hyperfine matrices and spectral weights.

    For each combination of magnetic spin projections across all nuclei
    groups, computes the sum hyperfine (A_1) and difference hyperfine
    (A_2) contributions, weighted by Pascal-triangle intensities.

    Parameters
    ----------
    Sys : Spinsystem
        Spin-system object with nuclei group parameters (nuclei_n,
        nuclei_I, donor_list, acceptor_list).
    a_projections : list[np.ndarray]
        Effective hyperfine couplings for each nuclei group,
        each of shape ``(N,)`` where N is the number of orientations.

    Returns
    -------
    A_1 : np.ndarray
        Sum hyperfine multiplied by 2, shape ``(N, n_comb, 1)``.
    A_2 : np.ndarray
        Difference hyperfine multiplied by 2, shape ``(N, n_comb, 1)``.
    spec_weights : list[float]
        Spectral weights (Pascal-triangle products) for each combination.
    """
    core_data = []
    core_types = []

    for i in range(len(Sys.A_tensors)):
        if i in Sys.acceptor_list:
            ct = 1
            n = Sys.nuclei_n[i]
            I = Sys.nuclei_I[i]
        elif i in Sys.donor_list:
            ct = -1
            n = Sys.nuclei_n[i]
            I = Sys.nuclei_I[i]
        else:
            ct = 0
            n = 0
            I = 0.0

        total_spin = n * I
        multi = get_multiplicity(total_spin)
        mI_vector = np.linspace(-total_spin, total_spin, multi).astype(np.float32)
        pascal = get_normalized_Pascal(n, I)
        mI_len = mI_vector.size

        hyperfine_matrix = np.outer(mI_vector, a_projections[i - 1])

        core_data.append((hyperfine_matrix, pascal, mI_len))
        core_types.append(ct)

    spec_weights = []
    sums_hyperfine = []
    diffs_hyperfine = []

    for indices in product(*[range(cd[2]) for cd in core_data]):
        sum_hf = sum(cd[0][idx] for cd, idx in zip(core_data, indices))
        diff_hf = sum(
            ct * cd[0][idx] for ct, cd, idx in zip(core_types, core_data, indices)
        )
        weight = 1.0
        for cd, idx in zip(core_data, indices):
            weight *= cd[1][idx]

        spec_weights.append(weight)
        sums_hyperfine.append(sum_hf)
        diffs_hyperfine.append(diff_hf)

    sum_hyperfine = np.array(sums_hyperfine).T
    diff_hyperfine = np.array(diffs_hyperfine).T

    A_1 = sum_hyperfine * 2
    A_2 = diff_hyperfine * 2

    A_1 = A_1[:, :, np.newaxis]
    A_2 = A_2[:, :, np.newaxis]

    return A_1, A_2, spec_weights
