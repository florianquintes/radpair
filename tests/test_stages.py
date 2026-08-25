"""Unit tests for the composable simulation stages in :mod:`radpair.functions`.

Each of the 8 functions extracted from ``do_simulation`` is tested
independently for shape, dtype, value-range, and known-value cases.
The fixtures from :mod:`tests.conftest` provide ready-to-use spin
systems, experiments, and simulation options.
"""

import numpy as np
import pytest

from radpair.functions import (
    _GAMMA_E_REF,
    _GAUSSIAN_FWHM_TO_SIGMA,
    assemble_spectrum,
    build_tensors,
    compute_hyperfine_combinations,
    compute_intensities,
    compute_resonance_fields,
    prepare_spinsystem,
    rotate_tensors,
    setup_orientation_grid,
)

# ---------------------------------------------------------------------------
# prepare_spinsystem
# ---------------------------------------------------------------------------


class TestPrepareSpinsystem:
    """Tests for :func:`prepare_spinsystem`."""

    def test_original_not_modified(self, minimal_spinsystem, experiment):
        """The input spinsystem must not be modified (deep copy)."""
        g1_before = minimal_spinsystem.g1.copy()
        width_before = minimal_spinsystem.width_gauss
        prepare_spinsystem(minimal_spinsystem, experiment.freq_mw, experiment.B_z)
        np.testing.assert_array_equal(minimal_spinsystem.g1, g1_before)
        assert minimal_spinsystem.width_gauss == width_before

    def test_g_tensors_halved(self, minimal_spinsystem, experiment):
        """g1 and g2 should be multiplied by 0.5."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        np.testing.assert_allclose(Sys.g1, minimal_spinsystem.g1 * 0.5)
        np.testing.assert_allclose(Sys.g2, minimal_spinsystem.g2 * 0.5)

    def test_a_tensors_converted(self, minimal_spinsystem, experiment):
        """A_tensors[0] (donor) should be converted from MHz to angular frequency with 0.5 factor."""
        from radpair.functions import MHz_2_T

        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        expected_a0 = MHz_2_T(minimal_spinsystem.A_tensors[0], minimal_spinsystem.g2)
        expected_a0 = expected_a0 * 0.5 * _GAMMA_E_REF
        np.testing.assert_allclose(Sys.A_tensors[0], expected_a0)

    def test_d_e_j_ex_converted(self, full_spinsystem, experiment):
        """D, E, J_ex should be converted from MHz to angular frequency."""
        from radpair.functions import MHz_2_T

        Sys, _, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        g_12 = (full_spinsystem.g1 + full_spinsystem.g2) / 2
        expected_d = MHz_2_T(full_spinsystem.D, g_12)
        expected_d = expected_d / 3 * 0.5 * _GAMMA_E_REF
        np.testing.assert_allclose(Sys.D, expected_d)

        expected_e = MHz_2_T(full_spinsystem.E, g_12)
        expected_e = expected_e * 0.5 * _GAMMA_E_REF
        np.testing.assert_allclose(Sys.E, expected_e)

        expected_j = MHz_2_T(full_spinsystem.J_ex, g_12)
        expected_j = expected_j * _GAMMA_E_REF
        np.testing.assert_allclose(Sys.J_ex, expected_j)

    def test_freq_mw_converted(self, minimal_spinsystem, experiment):
        """freq_mw should be converted to angular frequency."""
        mu_b = 1.399624617e10  # Bohr magneton in Hz/T (approx)
        _Sys, freq_mw, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        expected = experiment.freq_mw / (2 * mu_b) * _GAMMA_E_REF
        np.testing.assert_allclose(freq_mw, expected, rtol=1e-5)

    def test_bz_converted(self, minimal_spinsystem, experiment):
        """B_z should be converted from mT to angular frequency."""
        _, _, B_z = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        expected = experiment.B_z * 1e-3 * _GAMMA_E_REF
        np.testing.assert_allclose(B_z, expected)

    def test_width_gauss_converted(self, minimal_spinsystem, experiment):
        """width_gauss should be converted to sigma squared."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        expected = (
            minimal_spinsystem.width_gauss * 1e-3 * _GAMMA_E_REF
        ) ** 2 / _GAUSSIAN_FWHM_TO_SIGMA
        np.testing.assert_allclose(Sys.width_gauss, expected)

    def test_int_dtype_promoted(self):
        """A tensors with integer dtype should be promoted to float64."""
        from radpair._types import Spinsystem

        sys_int = Spinsystem(
            g1=np.array([2.0, 2.0, 2.0]),
            g2=np.array([2.0, 2.0, 2.0]),
            A_tensors=[
                np.array([1, 1, 1], dtype=np.int64),
                np.array([0, 0, 0], dtype=np.int64),
                np.array([0, 0, 0], dtype=np.int64),
                np.array([0, 0, 0], dtype=np.int64),
                np.array([0, 0, 0], dtype=np.int64),
            ],
            nuclei_n=[1, 0, 0, 0, 0],
            nuclei_I=[0.5, 0.0, 0.0, 0.0, 0.0],
            A_frames=[
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            ],
            width_gauss=0.5,
            donor_list=[0],
            acceptor_list=[],
        )
        Sys, _, _ = prepare_spinsystem(sys_int, 9.5e9, np.linspace(340, 350, 10))
        assert Sys.A_tensors[0].dtype == np.float64


# ---------------------------------------------------------------------------
# setup_orientation_grid
# ---------------------------------------------------------------------------


class TestSetupOrientationGrid:
    """Tests for :func:`setup_orientation_grid`."""

    def test_no_interpolation(self):
        """refinement=1 should produce no fine grid."""
        theta, phi, theta_fine, phi_fine, weights, mode = setup_orientation_grid(12, 1)
        assert theta.shape[0] == phi.shape[0]
        assert theta_fine is None
        assert phi_fine is None
        assert mode is False
        assert weights.ndim == 2

    def test_with_interpolation(self):
        """refinement>1 should produce a fine grid with more points."""
        theta, phi, theta_fine, phi_fine, weights, mode = setup_orientation_grid(12, 2)
        assert theta.shape[0] == phi.shape[0]
        assert theta_fine.shape[0] > theta.shape[0]
        assert phi_fine.shape[0] > phi.shape[0]
        assert mode is True
        assert weights.ndim == 2

    def test_theta_range(self):
        """Theta should be in [0, pi]."""
        theta, _phi, _, _, _, _ = setup_orientation_grid(12, 1)
        assert theta.min() >= 0
        assert theta.max() <= np.pi

    def test_phi_range(self):
        """Phi should be in [0, 2*pi)."""
        _, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        assert phi.min() >= 0
        assert phi.max() < 2 * np.pi + 1e-10


# ---------------------------------------------------------------------------
# build_tensors
# ---------------------------------------------------------------------------


class TestBuildTensors:
    """Tests for :func:`build_tensors`."""

    def test_all_tensors_shape(self, minimal_spinsystem, experiment):
        """all_tensors should have shape (3 + n_nuclei, 3, 3)."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, _frame_angles = build_tensors(Sys)
        n_nuclei = len(Sys.A_tensors)
        assert all_tensors.shape == (3 + n_nuclei, 3, 3)

    def test_frame_angles_shape(self, minimal_spinsystem, experiment):
        """frame_angles should have shape (3 + n_nuclei, 3)."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        _, frame_angles = build_tensors(Sys)
        n_nuclei = len(Sys.A_tensors)
        assert frame_angles.shape == (3 + n_nuclei, 3)

    def test_g_tensors_diagonal(self, minimal_spinsystem, experiment):
        """g1 and g2 tensors should be diagonal with halved g-values."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, _ = build_tensors(Sys)
        np.testing.assert_allclose(np.diag(all_tensors[0]), Sys.g1)
        np.testing.assert_allclose(np.diag(all_tensors[1]), Sys.g2)

    def test_d_tensor_diag_matches_get_D_diag(self, full_spinsystem, experiment):
        """D tensor diagonal should match get_D_diag(Sys.D, Sys.E)."""
        from radpair.functions import get_D_diag

        Sys, _, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, _ = build_tensors(Sys)
        expected = get_D_diag(Sys.D, Sys.E)
        np.testing.assert_allclose(np.diag(all_tensors[2]), expected)


# ---------------------------------------------------------------------------
# rotate_tensors
# ---------------------------------------------------------------------------


class TestRotateTensors:
    """Tests for :func:`rotate_tensors`."""

    def test_output_shapes(self, minimal_spinsystem, experiment):
        """g1, g2, D should be 1-D; a_projections should have 5 elements."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        n = theta.shape[0]
        assert g1.shape == (n,)
        assert g2.shape == (n,)
        assert D.shape == (n,)
        assert len(a_projections) == 5
        for a in a_projections:
            assert a.shape == (n,)

    def test_isotropic_g_no_frame(self, minimal_spinsystem, experiment):
        """With isotropic g and zero frames, g1 should equal Sys.g1[2] for all orientations."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        g1, _, _, _ = rotate_tensors(all_tensors, frame_angles, theta, phi)
        expected = minimal_spinsystem.g1[2] * 0.5
        np.testing.assert_allclose(g1, expected)

    def test_isotropic_a_projection_zero_for_inactive(
        self, minimal_spinsystem, experiment
    ):
        """Inactive nuclei groups (A=0) should produce zero projections."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        _, _, _, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        # Groups 2-5 are inactive in minimal_spinsystem
        for i in range(1, 5):
            np.testing.assert_allclose(a_projections[i], 0.0)

    def test_nonzero_frame_changes_g(self, full_spinsystem, experiment):
        """With nonzero frame angles, g1 should vary across orientations."""
        Sys, _, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        g1, _, _, _ = rotate_tensors(all_tensors, frame_angles, theta, phi)
        assert np.std(g1) > 0


# ---------------------------------------------------------------------------
# compute_hyperfine_combinations
# ---------------------------------------------------------------------------


class TestComputeHyperfineCombinations:
    """Tests for :func:`compute_hyperfine_combinations`."""

    def test_output_shapes(self, minimal_spinsystem, experiment):
        """A_1 and A_2 should have shape (N, n_comb, 1)."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        _, _, _, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        A_1, A_2, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        n = theta.shape[0]
        n_comb = len(spec_weights)
        assert A_1.shape == (n, n_comb, 1)
        assert A_2.shape == (n, n_comb, 1)

    def test_n_combinations_minimal(self, minimal_spinsystem, experiment):
        """Minimal system has 1 active group (n=1, I=0.5) → 2 combinations."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        _, _, _, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        _, _, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        assert len(spec_weights) == 2

    def test_weights_sum_to_one(self, full_spinsystem, experiment):
        """Spectral weights should sum to 1 (normalized Pascal triangles)."""
        Sys, _, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        _, _, _, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        _, _, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        np.testing.assert_allclose(sum(spec_weights), 1.0, atol=1e-10)

    def test_zero_projections_give_zero_hyperfine(self, minimal_spinsystem, experiment):
        """With all-zero a_projections, A_1 and A_2 should be zero."""
        Sys, _, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        n = 12
        a_projections = [np.zeros(n) for _ in range(len(Sys.A_tensors))]
        A_1, A_2, _ = compute_hyperfine_combinations(Sys, a_projections)
        np.testing.assert_allclose(A_1, 0.0)
        np.testing.assert_allclose(A_2, 0.0)


# ---------------------------------------------------------------------------
# compute_resonance_fields
# ---------------------------------------------------------------------------


class TestComputeResonanceFields:
    """Tests for :func:`compute_resonance_fields`."""

    @pytest.fixture
    def _resonance_inputs(self, full_spinsystem, experiment):
        """Prepare inputs for compute_resonance_fields."""
        Sys, freq_mw, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        all_tensors, frame_angles = build_tensors(Sys)
        theta, phi, _, _, _, _ = setup_orientation_grid(12, 1)
        g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        A_1, A_2, _spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        return Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2

    def test_res_fields_shape(self, _resonance_inputs):
        """res_fields should have 4 transitions in the last dimension."""
        J_ex, freq_mw, g1, g2, D, A_1, A_2 = _resonance_inputs
        res_fields, _, _, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        assert res_fields.shape[-1] == 4

    def test_delta_omega_shape(self, _resonance_inputs):
        """delta_omega should match res_fields shape."""
        J_ex, freq_mw, g1, g2, D, A_1, A_2 = _resonance_inputs
        _, delta_omega, _, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        res_fields, _, _, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        assert delta_omega.shape == res_fields.shape

    def test_quantum_beat_shape(self, _resonance_inputs):
        """quantum_beat should match res_fields shape."""
        J_ex, freq_mw, g1, g2, D, A_1, A_2 = _resonance_inputs
        _, _, quantum_beat, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        res_fields, _, _, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        assert quantum_beat.shape == res_fields.shape

    def test_widths_shape(self, _resonance_inputs):
        """widths should have shape (N, 4, n_comb)."""
        J_ex, freq_mw, g1, g2, D, A_1, A_2 = _resonance_inputs
        _, _, _, widths = compute_resonance_fields(J_ex, freq_mw, g1, g2, D, A_1, A_2)
        n = g1.shape[0]
        n_comb = A_1.shape[1]
        assert widths.shape == (n, 4, n_comb)

    def test_no_nans_in_res_fields(self, _resonance_inputs):
        """res_fields should not contain NaNs for reasonable inputs."""
        J_ex, freq_mw, g1, g2, D, A_1, A_2 = _resonance_inputs
        res_fields, _, _, _ = compute_resonance_fields(
            J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        assert not np.any(np.isnan(res_fields))


# ---------------------------------------------------------------------------
# compute_intensities
# ---------------------------------------------------------------------------


class TestComputeIntensities:
    """Tests for :func:`compute_intensities`."""

    def test_output_shape(self):
        """Intensities should match delta_omega shape."""
        delta_omega = np.random.rand(12, 3, 4) * 0.1
        quantum_beat = np.ones((12, 3, 4))
        intensities = compute_intensities(delta_omega, quantum_beat)
        assert intensities.shape == (12, 3, 4)

    def test_sign_pattern(self):
        """The sign pattern should be [+1, -1, +1, -1] along the last axis."""
        delta_omega = np.zeros((1, 1, 4))
        quantum_beat = np.ones((1, 1, 4))
        intensities = compute_intensities(delta_omega, quantum_beat)
        # With delta_omega=0, phase_angle=0, sin^2=0, so all intensities are 0.
        # Use nonzero delta_omega to see the sign pattern.
        delta_omega = np.full((1, 1, 4), 0.1)
        quantum_beat = np.full((1, 1, 4), 1.0)
        intensities = compute_intensities(delta_omega, quantum_beat)
        signs = np.sign(intensities[0, 0, :])
        assert signs[0] >= 0
        assert signs[1] <= 0
        assert signs[2] >= 0
        assert signs[3] <= 0

    def test_zero_delta_omega_gives_zero_intensity(self):
        """With delta_omega=0, phase angle is 0, so intensity is 0."""
        delta_omega = np.zeros((5, 2, 4))
        quantum_beat = np.ones((5, 2, 4))
        intensities = compute_intensities(delta_omega, quantum_beat)
        np.testing.assert_allclose(intensities, 0.0)


# ---------------------------------------------------------------------------
# assemble_spectrum
# ---------------------------------------------------------------------------


class TestAssembleSpectrum:
    """Tests for :func:`assemble_spectrum`."""

    @pytest.fixture
    def _assembly_inputs(self, full_spinsystem, experiment, simopt_basic):
        """Prepare all inputs for assemble_spectrum by running the pipeline stages."""
        Sys, freq_mw, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        theta, phi, theta_fine, phi_fine, weights, interp_mode = setup_orientation_grid(
            simopt_basic.knots, simopt_basic.refinement
        )
        all_tensors, frame_angles = build_tensors(Sys)
        g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        A_1, A_2, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        res_fields, delta_omega, quantum_beat, widths = compute_resonance_fields(
            Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        intensities = compute_intensities(delta_omega, quantum_beat)
        return {
            "res_fields": res_fields,
            "intensities": intensities,
            "widths": widths,
            "spec_weights": spec_weights,
            "original_width_gauss": full_spinsystem.width_gauss,
            "original_B_z": experiment.B_z,
            "theta_angles": theta,
            "phi_angles": phi,
            "theta_fine": theta_fine,
            "phi_fine": phi_fine,
            "weights": weights,
            "interpolation_mode": interp_mode,
        }

    def test_output_shape(self, _assembly_inputs, experiment):
        """Output shape should match the original B_z."""
        intensity = assemble_spectrum(**_assembly_inputs)
        assert intensity.shape == experiment.B_z.shape

    def test_no_nans(self, _assembly_inputs):
        """Output should not contain NaNs."""
        intensity = assemble_spectrum(**_assembly_inputs)
        assert not np.any(np.isnan(intensity))

    def test_real_valued(self, _assembly_inputs):
        """Output should be real-valued."""
        intensity = assemble_spectrum(**_assembly_inputs)
        assert not np.iscomplexobj(intensity)

    def test_has_positive_and_negative(self, _assembly_inputs):
        """Spin-correlated RP should have both absorptive and emissive lines."""
        intensity = assemble_spectrum(**_assembly_inputs)
        assert np.any(intensity > 0)
        assert np.any(intensity < 0)


# ---------------------------------------------------------------------------
# End-to-end: pipeline matches do_simulation
# ---------------------------------------------------------------------------


class TestPipelineMatchesDoSimulation:
    """Verify that calling the 8 stages manually produces the same result as do_simulation."""

    def test_pipeline_matches_single_call(
        self, full_spinsystem, experiment, simopt_basic
    ):
        from radpair.core import do_simulation

        expected = do_simulation(full_spinsystem, experiment, simopt_basic)

        Sys, freq_mw, _ = prepare_spinsystem(
            full_spinsystem, experiment.freq_mw, experiment.B_z
        )
        theta, phi, theta_fine, phi_fine, weights, interp_mode = setup_orientation_grid(
            simopt_basic.knots, simopt_basic.refinement
        )
        all_tensors, frame_angles = build_tensors(Sys)
        g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        A_1, A_2, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        res_fields, delta_omega, quantum_beat, widths = compute_resonance_fields(
            Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        intensities = compute_intensities(delta_omega, quantum_beat)
        result = assemble_spectrum(
            res_fields,
            intensities,
            widths,
            spec_weights,
            full_spinsystem.width_gauss,
            experiment.B_z,
            theta,
            phi,
            theta_fine,
            phi_fine,
            weights,
            interp_mode,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_pipeline_matches_with_interpolation(
        self, minimal_spinsystem, experiment, simopt_interpolation
    ):
        from radpair.core import do_simulation

        expected = do_simulation(minimal_spinsystem, experiment, simopt_interpolation)

        Sys, freq_mw, _ = prepare_spinsystem(
            minimal_spinsystem, experiment.freq_mw, experiment.B_z
        )
        theta, phi, theta_fine, phi_fine, weights, interp_mode = setup_orientation_grid(
            simopt_interpolation.knots, simopt_interpolation.refinement
        )
        all_tensors, frame_angles = build_tensors(Sys)
        g1, g2, D, a_projections = rotate_tensors(all_tensors, frame_angles, theta, phi)
        A_1, A_2, spec_weights = compute_hyperfine_combinations(Sys, a_projections)
        res_fields, delta_omega, quantum_beat, widths = compute_resonance_fields(
            Sys.J_ex, freq_mw, g1, g2, D, A_1, A_2
        )
        intensities = compute_intensities(delta_omega, quantum_beat)
        result = assemble_spectrum(
            res_fields,
            intensities,
            widths,
            spec_weights,
            minimal_spinsystem.width_gauss,
            experiment.B_z,
            theta,
            phi,
            theta_fine,
            phi_fine,
            weights,
            interp_mode,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
