"""Tests for eigsep_sim.experiments (absorbed and generalized from bloom21cm).

The point of generalizing this module was to remove the hard dependency on
``bloom_sim``; these tests exercise it with small hand-rolled ``simulate_fn``
and ``design_fn`` callables to confirm the injection seam actually works.
"""

import numpy as np
import pytest

from eigsep_sim.experiments import (
    FrequencyRecoveryResult,
    monopole_uncertainty_from_fit,
    run_frequency_recovery,
    source_column_from_signal,
)
from eigsep_sim.recovery import build_surface_design_matrix, normal_solve

NOBS, NDIPOLE, NPIX = 24, 2, 12


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    masks = np.ones((NOBS, NPIX))
    beams = rng.uniform(0.5, 1.5, size=(NOBS, NDIPOLE, NPIX))
    omega_B = beams.sum(axis=2)
    return masks, beams, omega_B


def _design_fn(masks, beams, omega_B, sun_pixels, npix, *,
               include_t_rx=False, source_columns=None):
    """Minimal design builder matching the documented protocol."""
    weights = np.asarray(beams) / np.asarray(omega_B)[:, :, None]
    return build_surface_design_matrix(
        weights,
        masks,
        source_columns=source_columns,
        include_receiver_offsets=include_t_rx,
    )


def _make_simulate_fn(design_fn, noiseless=True):
    def simulate_fn(masks, beams, omega_B, sky_map, t_regolith, t_sun,
                    sun_pixels, sigma_noise, rng=None, t_rx=None,
                    additive_signal=None):
        A = design_fn(masks, beams, omega_B, sun_pixels, len(sky_map),
                      include_t_rx=False, source_columns=None)
        x = np.concatenate([sky_map, [t_regolith]])
        y = A @ x
        if additive_signal is not None:
            y = y + np.asarray(additive_signal).reshape(y.shape)
        if not noiseless and rng is not None:
            y = y + rng.normal(0.0, float(np.mean(sigma_noise)), size=y.shape)
        return y.copy(), y
    return simulate_fn


class TestRunFrequencyRecovery:
    def test_matched_truth_and_model_recovers_sky(self, arrays):
        masks, beams, omega_B = arrays
        rng = np.random.default_rng(1)
        sky = rng.uniform(100.0, 300.0, size=NPIX)

        result = run_frequency_recovery(
            simulate_fn=_make_simulate_fn(_design_fn),
            design_fn=_design_fn,
            truth_masks=masks,
            truth_beams=beams,
            truth_omega_B=omega_B,
            recovery_masks=masks,
            recovery_beams=beams,
            recovery_omega_B=omega_B,
            sky_map=sky,
            t_regolith=0.0,
            t_sun=0.0,
            sun_pixels=np.zeros(NOBS, dtype=int),
            sigma_noise=1.0,
            include_t_rx=False,
        )
        assert isinstance(result, FrequencyRecoveryResult)
        assert result.sky_mean_K == pytest.approx(sky.mean(), rel=1e-6)

    def test_mismatched_beams_bias_the_recovery(self, arrays):
        masks, beams, omega_B = arrays
        rng = np.random.default_rng(2)
        sky = rng.uniform(100.0, 300.0, size=NPIX)
        # recovery model uses a perturbed beam -> the A/B seam must show it
        bad_beams = beams * rng.uniform(0.8, 1.2, size=beams.shape)

        matched = run_frequency_recovery(
            simulate_fn=_make_simulate_fn(_design_fn),
            design_fn=_design_fn,
            truth_masks=masks, truth_beams=beams, truth_omega_B=omega_B,
            recovery_masks=masks, recovery_beams=beams,
            recovery_omega_B=omega_B,
            sky_map=sky, t_regolith=0.0, t_sun=0.0,
            sun_pixels=np.zeros(NOBS, dtype=int),
            sigma_noise=1.0, include_t_rx=False,
        )
        mismatched = run_frequency_recovery(
            simulate_fn=_make_simulate_fn(_design_fn),
            design_fn=_design_fn,
            truth_masks=masks, truth_beams=beams, truth_omega_B=omega_B,
            recovery_masks=masks, recovery_beams=bad_beams,
            recovery_omega_B=bad_beams.sum(axis=2),
            sky_map=sky, t_regolith=0.0, t_sun=0.0,
            sun_pixels=np.zeros(NOBS, dtype=int),
            sigma_noise=1.0, include_t_rx=False,
        )
        assert abs(mismatched.sky_mean_K - sky.mean()) > abs(
            matched.sky_mean_K - sky.mean()
        )

    def test_returns_design_matrix_and_data(self, arrays):
        masks, beams, omega_B = arrays
        sky = np.full(NPIX, 200.0)
        result = run_frequency_recovery(
            simulate_fn=_make_simulate_fn(_design_fn),
            design_fn=_design_fn,
            truth_masks=masks, truth_beams=beams, truth_omega_B=omega_B,
            recovery_masks=masks, recovery_beams=beams,
            recovery_omega_B=omega_B,
            sky_map=sky, t_regolith=0.0, t_sun=0.0,
            sun_pixels=np.zeros(NOBS, dtype=int),
            sigma_noise=1.0, include_t_rx=False,
        )
        assert result.design_matrix.shape[0] == NOBS * NDIPOLE
        assert result.design_matrix.shape[1] == NPIX + 1
        assert result.y.shape == (NOBS * NDIPOLE,)


class TestMonopoleUncertainty:
    def test_nan_when_nothing_observed(self, arrays):
        masks, beams, omega_B = arrays
        A = _design_fn(masks, beams, omega_B, None, NPIX)
        fit = normal_solve(A, np.zeros(A.shape[0]), NPIX)
        fit = dict(fit)
        fit["unobserved"] = np.ones(NPIX, dtype=bool)
        assert np.isnan(
            monopole_uncertainty_from_fit(fit, A, NPIX, 1.0)
        )

    def test_scales_linearly_with_noise(self, arrays):
        masks, beams, omega_B = arrays
        A = _design_fn(masks, beams, omega_B, None, NPIX)
        fit = normal_solve(A, np.zeros(A.shape[0]), NPIX)
        one = monopole_uncertainty_from_fit(fit, A, NPIX, 1.0)
        two = monopole_uncertainty_from_fit(fit, A, NPIX, 2.0)
        assert two == pytest.approx(2 * one)

    def test_positive_for_observed_sky(self, arrays):
        masks, beams, omega_B = arrays
        A = _design_fn(masks, beams, omega_B, None, NPIX)
        fit = normal_solve(A, np.zeros(A.shape[0]), NPIX)
        assert monopole_uncertainty_from_fit(fit, A, NPIX, 1.0) > 0


class TestSourceColumnFromSignal:
    def test_divides_by_scalar_temperature(self):
        signal = np.arange(NOBS * NDIPOLE, dtype=float).reshape(
            NOBS, NDIPOLE
        )
        got = source_column_from_signal(signal, 4.0)
        np.testing.assert_allclose(got, signal / 4.0)

    def test_squeezes_single_frequency_axis(self):
        signal = np.ones((NOBS, NDIPOLE, 1))
        got = source_column_from_signal(signal, 2.0)
        assert got.shape == (NOBS, NDIPOLE)
        np.testing.assert_allclose(got, 0.5)

    def test_multifrequency_raises(self):
        with pytest.raises(ValueError, match="one frequency channel"):
            source_column_from_signal(np.ones((NOBS, NDIPOLE, 3)), 1.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            source_column_from_signal(np.ones((NOBS, NDIPOLE)), 0.0)

    def test_vector_temperature_raises(self):
        with pytest.raises(ValueError, match="scalar or length 1"):
            source_column_from_signal(
                np.ones((NOBS, NDIPOLE)), np.array([1.0, 2.0])
            )
