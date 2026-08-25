"""Tests for :mod:`radpair.functions`."""

import numpy as np
import pytest
import scipy.constants as constant

from radpair.functions import (
    MHz_2_T,
    get_D_diag,
    get_generalized_Pascal,
    get_multiplicity,
    get_normalized_Pascal,
    rescale_array,
    tensor_rotation,
    vector_product_combinations,
)

# ---------------------------------------------------------------------------
# tensor_rotation
# ---------------------------------------------------------------------------


class TestTensorRotation:
    """Tests for :func:`tensor_rotation`."""

    def test_2d_identity_rotation(self):
        """Identity rotation (all angles zero) preserves the tensor."""
        tensor = np.diag(np.array([1.0, 2.0, 3.0]))
        angles = np.zeros(1)
        result = tensor_rotation(tensor, angles, angles, angles)
        assert result.shape == (1, 3, 3)
        np.testing.assert_allclose(result[0], tensor)

    def test_3d_identity_rotation(self):
        """Identity rotation preserves a batch of tensors."""
        tensors = np.array([np.diag([1, 2, 3]), np.diag([4, 5, 6])], dtype=float)
        angles = np.zeros(2)
        result = tensor_rotation(tensors, angles, angles, angles)
        assert result.shape == (2, 3, 3)
        np.testing.assert_allclose(result[0], tensors[0])
        np.testing.assert_allclose(result[1], tensors[1])

    def test_2d_known_rotation(self):
        """180° rotation about z swaps x and y diagonal elements."""
        tensor = np.diag(np.array([1.0, 2.0, 3.0]))
        phi = np.array([np.pi])
        theta = np.array([0.0])
        psi = np.array([0.0])
        result = tensor_rotation(tensor, phi, theta, psi)
        # Rotation by 180° about z: x→-x, y→-y, z→z
        # Diagonal tensor stays diagonal with same values
        np.testing.assert_allclose(result[0], tensor, atol=1e-10)

    def test_3d_known_rotation(self):
        """90° rotation about z maps x→y, y→-x for a batch of tensors."""
        tensors = np.array([np.diag([1.0, 0.0, 0.0])])
        phi = np.array([np.pi / 2])
        theta = np.array([0.0])
        psi = np.array([0.0])
        result = tensor_rotation(tensors, phi, theta, psi)
        # A tensor with only xx=1, rotated 90° about z, should have yy=1
        np.testing.assert_allclose(result[0, 1, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 0, 0], 0.0, atol=1e-10)

    def test_psi_none_defaults_to_zero(self):
        """When psi is None, the result matches psi=zeros."""
        tensor = np.diag(np.array([1.0, 2.0, 3.0]))
        phi = np.array([0.1, 0.2])
        theta = np.array([0.3, 0.4])
        result_no_psi = tensor_rotation(tensor, phi, theta)
        result_zero_psi = tensor_rotation(tensor, phi, theta, np.zeros(2))
        np.testing.assert_allclose(result_no_psi, result_zero_psi)

    def test_invalid_dimensions_raises(self):
        """A 1-D tensor raises ValueError."""
        tensor = np.array([1.0, 2.0, 3.0])
        angles = np.zeros(1)
        with pytest.raises(ValueError, match="wrong dimensions"):
            tensor_rotation(tensor, angles, angles, angles)

    def test_multiple_angles(self):
        """Rotation with N angles returns N rotated tensors."""
        tensor = np.diag(np.array([1.0, 2.0, 3.0]))
        n = 5
        phi = np.linspace(0, np.pi, n)
        theta = np.linspace(0, np.pi / 2, n)
        result = tensor_rotation(tensor, phi, theta)
        assert result.shape == (n, 3, 3)

    def test_rotation_is_orthogonal(self):
        """Rotated tensor has the same eigenvalues as the original."""
        tensor = np.diag(np.array([3.0, 1.0, 2.0]))
        phi = np.array([0.5])
        theta = np.array([1.0])
        psi = np.array([0.3])
        result = tensor_rotation(tensor, phi, theta, psi)
        orig_eig = np.sort(np.linalg.eigvalsh(tensor))
        rot_eig = np.sort(np.linalg.eigvalsh(result[0]))
        np.testing.assert_allclose(rot_eig, orig_eig)


# ---------------------------------------------------------------------------
# get_multiplicity
# ---------------------------------------------------------------------------


class TestGetMultiplicity:
    """Tests for :func:`get_multiplicity`."""

    @pytest.mark.parametrize("spin,expected", [(0, 1), (0.5, 2), (1, 3), (1.5, 4)])
    def test_valid_spins(self, spin, expected):
        assert get_multiplicity(spin) == expected

    def test_negative_spin_raises(self):
        with pytest.raises(ValueError, match="negative"):
            get_multiplicity(-0.5)

    def test_non_half_integer_spin_raises(self):
        with pytest.raises(ValueError, match="divisible by 0.5"):
            get_multiplicity(0.3)


# ---------------------------------------------------------------------------
# vector_product_combinations
# ---------------------------------------------------------------------------


class TestVectorProductCombinations:
    """Tests for :func:`vector_product_combinations`."""

    def test_shape(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = vector_product_combinations(a, b)
        assert result.shape == (3, 2)

    def test_values(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected = np.outer(a, b)
        np.testing.assert_array_equal(vector_product_combinations(a, b), expected)


# ---------------------------------------------------------------------------
# get_generalized_Pascal
# ---------------------------------------------------------------------------


class TestGetGeneralizedPascal:
    """Tests for :func:`get_generalized_Pascal`."""

    def test_n_zero(self):
        result = get_generalized_Pascal(0, 0.5)
        assert result.shape == (1,)
        np.testing.assert_array_equal(result, [1])

    def test_known_case_spin_half_n2(self):
        """Two spin-1/2 nuclei → [1, 2, 1]."""
        result = get_generalized_Pascal(2, 0.5)
        np.testing.assert_array_equal(result, [1, 2, 1])

    def test_known_case_spin_half_n1(self):
        """One spin-1/2 nucleus → [1, 1]."""
        result = get_generalized_Pascal(1, 0.5)
        np.testing.assert_array_equal(result, [1, 1])

    def test_known_case_spin_1_n1(self):
        """One spin-1 nucleus → [1, 1, 1]."""
        result = get_generalized_Pascal(1, 1.0)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_negative_number_raises(self):
        with pytest.raises(ValueError, match="negativ"):
            get_generalized_Pascal(-1, 0.5)

    def test_negative_spin_raises(self):
        with pytest.raises(ValueError, match="negativ"):
            get_generalized_Pascal(1, -0.5)

    def test_non_integer_number_raises(self):
        with pytest.raises(TypeError, match="natural number"):
            get_generalized_Pascal(1.5, 0.5)

    def test_non_half_integer_spin_raises(self):
        with pytest.raises(ValueError, match="divisible by 0.5"):
            get_generalized_Pascal(1, 0.3)


# ---------------------------------------------------------------------------
# get_normalized_Pascal
# ---------------------------------------------------------------------------


class TestGetNormalizedPascal:
    """Tests for :func:`get_normalized_Pascal`."""

    def test_sums_to_one(self):
        result = get_normalized_Pascal(3, 0.5)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_n_zero(self):
        result = get_normalized_Pascal(0, 0.5)
        np.testing.assert_allclose(result, [1.0])

    def test_matches_generalized_rescaled(self):
        n, s = 2, 1.0
        generalized = get_generalized_Pascal(n, s)
        expected = generalized / generalized.sum()
        result = get_normalized_Pascal(n, s)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# rescale_array
# ---------------------------------------------------------------------------


class TestRescaleArray:
    """Tests for :func:`rescale_array`."""

    def test_default_norm(self):
        arr = np.array([1.0, 3.0, 4.0])
        result = rescale_array(arr)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_custom_norm(self):
        arr = np.array([1.0, 3.0, 4.0])
        result = rescale_array(arr, norm=8.0)
        np.testing.assert_allclose(result.sum(), 8.0)

    def test_values_correct(self):
        arr = np.array([1.0, 3.0, 4.0])
        result = rescale_array(arr)
        expected = np.array([0.125, 0.375, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_zero_sum_raises(self):
        arr = np.zeros(5)
        with pytest.raises(ZeroDivisionError, match="sum 0"):
            rescale_array(arr)


# ---------------------------------------------------------------------------
# get_D_diag
# ---------------------------------------------------------------------------


class TestGetDDiag:
    """Tests for :func:`get_D_diag`."""

    def test_known_values(self):
        D, E = 10.0, 2.0
        result = get_D_diag(D, E)
        expected = np.array([D - E, D + E, -2 * D])
        np.testing.assert_allclose(result, expected)

    def test_zero_E(self):
        D = 5.0
        result = get_D_diag(D, 0.0)
        expected = np.array([5.0, 5.0, -10.0])
        np.testing.assert_allclose(result, expected)

    def test_shape(self):
        result = get_D_diag(1.0, 0.5)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# MHz_2_T
# ---------------------------------------------------------------------------


class TestMHz2T:
    """Tests for :func:`MHz_2_T`."""

    def test_scalar_conversion(self):
        g = np.array([2.0, 2.0, 2.0])
        mu_b = constant.value("Bohr magneton in Hz/T")
        nu = 1.0
        expected = 1e6 * nu / (2.0 * mu_b)
        result = MHz_2_T(nu, g)
        np.testing.assert_allclose(result, expected)

    def test_array_conversion(self):
        g = np.array([2.0, 2.0, 2.0])
        mu_b = constant.value("Bohr magneton in Hz/T")
        nu = np.array([1.0, 2.0, 3.0])
        expected = 1e6 * nu / (2.0 * mu_b)
        result = MHz_2_T(nu, g)
        np.testing.assert_allclose(result, expected)

    def test_anisotropic_g(self):
        g = np.array([2.001, 2.005, 2.010])
        g_iso = g.sum() / 3
        mu_b = constant.value("Bohr magneton in Hz/T")
        nu = 5.0
        expected = 1e6 * nu / (g_iso * mu_b)
        result = MHz_2_T(nu, g)
        np.testing.assert_allclose(result, expected)

    def test_negative_g_raises(self):
        g = np.array([2.0, -1.0, 2.0])
        with pytest.raises(ValueError, match="higher than 0"):
            MHz_2_T(1.0, g)
