"""Tests for :mod:`radpair.classes`."""

import numpy as np
import pytest

from radpair.classes import Core, Matrix
from radpair.functions import get_normalized_Pascal

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


class TestMatrixInit:
    """Tests for :class:`Matrix` initialization."""

    def test_stores_matrix(self):
        mat = np.diag(np.array([1.0, 2.0, 3.0]))
        m = Matrix(mat)
        np.testing.assert_array_equal(m.matrix, mat)

    def test_matrix_rot_is_none(self):
        m = Matrix(np.diag(np.array([1.0, 2.0, 3.0])))
        assert m.matrix_rot is None


class TestMatrixMatrot:
    """Tests for :meth:`Matrix.matrot`."""

    def test_identity_rotation_preserves_matrix(self):
        mat = np.diag(np.array([1.0, 2.0, 3.0]))
        m = Matrix(mat)
        angles = np.zeros(3)
        m.matrot(phi=angles, theta=angles)
        assert m.matrix_rot is not None
        assert m.matrix_rot.shape == (3, 3, 3)
        for i in range(3):
            np.testing.assert_allclose(m.matrix_rot[i], mat)

    def test_original_matrix_unchanged(self):
        mat = np.diag(np.array([1.0, 2.0, 3.0]))
        m = Matrix(mat)
        angles = np.array([0.5, 0.1, 0.3])
        m.matrot(phi=angles, theta=angles)
        np.testing.assert_array_equal(m.matrix, mat)

    def test_rotated_shape(self):
        m = Matrix(np.diag(np.array([1.0, 2.0, 3.0])))
        n = 5
        phi = np.linspace(0, np.pi, n)
        theta = np.linspace(0, np.pi / 2, n)
        m.matrot(phi=phi, theta=theta)
        assert m.matrix_rot.shape == (n, 3, 3)


class TestGetHyperfineProjection:
    """Tests for :meth:`Matrix.get_hyperfine_projection`."""

    def test_diagonal_isotropic(self):
        """Isotropic diagonal tensor → projection equals the diagonal value."""
        mat = np.diag(np.array([5.0, 5.0, 5.0]))
        m = Matrix(mat)
        angles = np.zeros(3)
        m.matrot(phi=angles, theta=angles)
        result = m.get_hyperfine_projection()
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_shape(self):
        mat = np.diag(np.array([1.0, 2.0, 3.0]))
        m = Matrix(mat)
        n = 10
        phi = np.linspace(0, np.pi, n)
        theta = np.linspace(0, np.pi / 2, n)
        m.matrot(phi=phi, theta=theta)
        result = m.get_hyperfine_projection()
        assert result.shape == (n,)

    def test_all_nonneg(self):
        """Effective couplings are Euclidean norms, hence non-negative."""
        mat = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.5], [2.0, 0.5, 5.0]])
        m = Matrix(mat)
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, np.pi / 2, 20)
        m.matrot(phi=phi, theta=theta)
        result = m.get_hyperfine_projection()
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class TestCoreInit:
    """Tests for :class:`Core` initialization."""

    def test_stores_number_and_spin(self):
        c = Core(2, 0.5)
        assert c.number == 2
        assert c.spin == 0.5

    def test_total_spin(self):
        c = Core(3, 0.5)
        assert c.total_spin == 1.5

    def test_total_spin_zero(self):
        c = Core(0, 0.0)
        assert c.total_spin == 0.0

    def test_pascal_correct(self):
        c = Core(2, 0.5)
        expected = get_normalized_Pascal(2, 0.5)
        np.testing.assert_allclose(c.pascal, expected)

    def test_mI_len(self):
        c = Core(2, 0.5)
        # 2 nuclei with spin 1/2 → total_spin=1 → multiplicity=3
        assert c.mI_len == 3

    def test_negative_number_raises(self):
        with pytest.raises(ValueError, match="negativ"):
            Core(-1, 0.5)

    def test_negative_spin_raises(self):
        with pytest.raises(ValueError, match="negativ"):
            Core(1, -0.5)

    def test_non_integer_number_raises(self):
        with pytest.raises(TypeError, match="natural number"):
            Core(1.5, 0.5)

    def test_non_half_integer_spin_raises(self):
        with pytest.raises(ValueError, match="divisible by 0.5"):
            Core(1, 0.3)

    def test_zero_core(self):
        """Core(0, 0) represents an empty nuclei group."""
        c = Core(0, 0)
        assert c.number == 0
        assert c.spin == 0
        assert c.total_spin == 0.0
        assert c.mI_len == 1
        assert c.mI_vector.shape == (1,)


class TestCoreSetHyperfineMatrix:
    """Tests for :meth:`Core.set_hyperfine_matrix`."""

    def test_shape(self):
        c = Core(1, 0.5)
        hf = np.array([1.0, 2.0, 3.0])
        c.set_hyperfine_matrix(hf)
        # mI_vector has length 2, hf has length 3 → (2, 3)
        assert c.hyperfine_matrix.shape == (2, 3)

    def test_values(self):
        c = Core(1, 0.5)
        hf = np.array([10.0, 20.0, 30.0])
        c.set_hyperfine_matrix(hf)
        # mI_vector = [-0.5, 0.5]
        expected = np.outer(np.array([-0.5, 0.5]), hf)
        np.testing.assert_allclose(c.hyperfine_matrix, expected)

    def test_zero_core(self):
        c = Core(0, 0)
        hf = np.array([1.0, 2.0, 3.0])
        c.set_hyperfine_matrix(hf)
        assert c.hyperfine_matrix.shape == (1, 3)


class TestCoreGetMagneticSpinVector:
    """Tests for :meth:`Core.get_magnetic_spin_vector`."""

    def test_spin_half_one_nucleus(self):
        c = Core(1, 0.5)
        np.testing.assert_allclose(c.mI_vector, [-0.5, 0.5])

    def test_spin_half_two_nuclei(self):
        c = Core(2, 0.5)
        np.testing.assert_allclose(c.mI_vector, [-1.0, 0.0, 1.0])

    def test_spin_one_one_nucleus(self):
        c = Core(1, 1.0)
        np.testing.assert_allclose(c.mI_vector, [-1.0, 0.0, 1.0])

    def test_dtype(self):
        c = Core(1, 0.5)
        assert c.mI_vector.dtype == np.float32

    def test_symmetric(self):
        """mI vector is symmetric about zero."""
        c = Core(3, 1.5)
        np.testing.assert_allclose(c.mI_vector, -c.mI_vector[::-1])
