"""Tests for the dataclass types and ``__post_init__`` validation in
:mod:`radpair._types`.
"""

import numpy as np
import pytest

from radpair._types import Experiment, SimulationOptions, Spinsystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_G1 = np.array([2.0, 2.0, 2.0])
_G2 = np.array([2.0, 2.0, 2.0])
_ZERO = np.array([0.0, 0.0, 0.0])


def _valid_kwargs(**overrides) -> dict:
    """Return valid Spinsystem constructor kwargs, optionally overridden."""
    kwargs: dict = {
        "g1": _G1.copy(),
        "g2": _G2.copy(),
        "A_tensors": [np.array([1.5, 1.5, 1.5]), _ZERO.copy(), _ZERO.copy()],
        "nuclei_n": [1, 0, 0],
        "nuclei_I": [0.5, 0.0, 0.0],
        "A_frames": [_ZERO.copy(), _ZERO.copy(), _ZERO.copy()],
        "width_gauss": 0.05,
        "donor_list": [0],
        "acceptor_list": [],
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Spinsystem — valid construction
# ---------------------------------------------------------------------------


class TestSpinsystemValid:
    """Tests for valid Spinsystem construction."""

    def test_minimal_valid(self):
        """A minimal valid Spinsystem should construct without error."""
        sys = Spinsystem(**_valid_kwargs())
        assert sys.g1.shape == (3,)

    def test_defaults_applied(self):
        """Optional fields should get their default values."""
        sys = Spinsystem(**_valid_kwargs())
        assert sys.D == 0.0
        assert sys.E == 0.0
        assert sys.J_ex == 0.0
        np.testing.assert_array_equal(sys.g1_frame, _ZERO)
        np.testing.assert_array_equal(sys.g2_frame, _ZERO)
        np.testing.assert_array_equal(sys.D_frame, _ZERO)
        assert sys.donor_list == [0]
        assert sys.acceptor_list == []

    def test_defaults_when_omitted(self):
        """D, E, J_ex, frames should default when omitted entirely."""
        sys = Spinsystem(
            g1=_G1.copy(),
            g2=_G2.copy(),
            A_tensors=[_ZERO.copy()],
            nuclei_n=[0],
            nuclei_I=[0.0],
            A_frames=[_ZERO.copy()],
            width_gauss=0.1,
        )
        assert sys.D == 0.0
        assert sys.E == 0.0
        assert sys.J_ex == 0.0
        assert sys.g1_frame.shape == (3,)

    def test_is_dataclass_instance(self):
        """Spinsystem should be a dataclass instance."""
        from dataclasses import is_dataclass

        assert is_dataclass(Spinsystem)
        assert is_dataclass(Spinsystem(**_valid_kwargs()))


# ---------------------------------------------------------------------------
# Spinsystem — validation errors
# ---------------------------------------------------------------------------


class TestSpinsystemValidation:
    """Tests for Spinsystem ``__post_init__`` validation."""

    def test_nuclei_n_length_mismatch(self):
        with pytest.raises(ValueError, match="nuclei_n has 2 elements"):
            Spinsystem(**_valid_kwargs(nuclei_n=[1, 0]))

    def test_nuclei_I_length_mismatch(self):
        with pytest.raises(ValueError, match="nuclei_I has 2 elements"):
            Spinsystem(**_valid_kwargs(nuclei_I=[0.5, 0.0]))

    def test_A_frames_length_mismatch(self):
        with pytest.raises(ValueError, match="A_frames has 2 elements"):
            Spinsystem(**_valid_kwargs(A_frames=[_ZERO.copy(), _ZERO.copy()]))

    def test_A_tensors_wrong_shape(self):
        with pytest.raises(ValueError, match="A_tensors\\[0\\] has shape"):
            Spinsystem(
                **_valid_kwargs(
                    A_tensors=[np.array([1.0, 2.0]), _ZERO.copy(), _ZERO.copy()]
                )
            )

    def test_A_frames_wrong_shape(self):
        with pytest.raises(ValueError, match="A_frames\\[0\\] has shape"):
            Spinsystem(
                **_valid_kwargs(
                    A_frames=[np.array([0.0, 0.0]), _ZERO.copy(), _ZERO.copy()]
                )
            )

    def test_g1_not_positive(self):
        with pytest.raises(ValueError, match="g1 values must be positive"):
            Spinsystem(**_valid_kwargs(g1=np.array([2.0, -1.0, 2.0])))

    def test_g2_not_positive(self):
        with pytest.raises(ValueError, match="g2 values must be positive"):
            Spinsystem(**_valid_kwargs(g2=np.array([2.0, 0.0, 2.0])))

    def test_nuclei_n_negative(self):
        with pytest.raises(ValueError, match="nuclei_n\\[0\\] = -1"):
            Spinsystem(**_valid_kwargs(nuclei_n=[-1, 0, 0]))

    def test_nuclei_n_not_int(self):
        with pytest.raises(TypeError, match="nuclei_n\\[0\\] is float"):
            Spinsystem(**_valid_kwargs(nuclei_n=[1.0, 0, 0]))

    def test_nuclei_I_negative(self):
        with pytest.raises(ValueError, match="nuclei_I\\[0\\] = -0.5"):
            Spinsystem(**_valid_kwargs(nuclei_I=[-0.5, 0.0, 0.0]))

    def test_nuclei_I_not_half_integer(self):
        with pytest.raises(ValueError, match="multiple of 0.5"):
            Spinsystem(**_valid_kwargs(nuclei_I=[0.3, 0.0, 0.0]))

    def test_width_gauss_zero(self):
        with pytest.raises(ValueError, match="width_gauss = 0.0"):
            Spinsystem(**_valid_kwargs(width_gauss=0.0))

    def test_width_gauss_negative(self):
        with pytest.raises(ValueError, match="width_gauss = -0.1"):
            Spinsystem(**_valid_kwargs(width_gauss=-0.1))

    def test_donor_index_out_of_range(self):
        with pytest.raises(IndexError, match="out of range"):
            Spinsystem(**_valid_kwargs(donor_list=[5]))

    def test_acceptor_index_out_of_range(self):
        with pytest.raises(IndexError, match="out of range"):
            Spinsystem(**_valid_kwargs(acceptor_list=[3]))

    def test_donor_acceptor_overlap(self):
        with pytest.raises(ValueError, match="appear in both"):
            Spinsystem(**_valid_kwargs(donor_list=[0], acceptor_list=[0]))

    def test_negative_index(self):
        with pytest.raises(IndexError, match="out of range"):
            Spinsystem(**_valid_kwargs(donor_list=[-1]))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class TestExperiment:
    """Tests for Experiment dataclass."""

    def test_magnetic_field_defaults_to_bz_copy(self):
        exp = Experiment(B_z=np.linspace(340, 350, 10), freq_mw=9.5e9)
        np.testing.assert_array_equal(exp.magnetic_field, exp.B_z)
        assert exp.magnetic_field is not exp.B_z

    def test_explicit_magnetic_field(self):
        bz = np.linspace(340, 350, 10)
        mf = np.linspace(340, 350, 20)
        exp = Experiment(B_z=bz, freq_mw=9.5e9, magnetic_field=mf)
        assert exp.magnetic_field is mf

    def test_is_dataclass(self):
        from dataclasses import is_dataclass

        assert is_dataclass(Experiment)


# ---------------------------------------------------------------------------
# SimulationOptions
# ---------------------------------------------------------------------------


class TestSimulationOptions:
    """Tests for SimulationOptions dataclass."""

    def test_defaults(self):
        opt = SimulationOptions()
        assert opt.grid_points == 12
        assert opt.refinement == 1
        assert opt.cpu_cores == 1

    def test_custom(self):
        opt = SimulationOptions(grid_points=24, refinement=2, cpu_cores=4)
        assert opt.grid_points == 24
        assert opt.refinement == 2
        assert opt.cpu_cores == 4

    def test_is_dataclass(self):
        from dataclasses import is_dataclass

        assert is_dataclass(SimulationOptions)
