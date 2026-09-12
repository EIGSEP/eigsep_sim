"""Tests for the systematic-injection helpers absorbed into
eigsep_sim.sources from bloom21cm/src/systematics.py.

Array layout throughout:
    masks   (nobs, npix)
    beams   (nobs, ndipole, npix)
    omega_B (nobs, ndipole)
"""

import numpy as np
import pytest

from eigsep_sim.sources import (
    earth_rfi_signal,
    earth_rfi_temperature_K,
    extended_source_coupling,
    extended_source_signal,
    k_to_v_per_m_root_hz,
    normalized_beam_weights,
    point_source_coupling,
    point_source_signal,
    quiet_sun_signal,
    surface_emission_signal,
    surface_residual_signal,
    synthetic_lunar_temperature_map,
)

NOBS, NDIPOLE, NPIX = 6, 2, 12


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    masks = np.ones((NOBS, NPIX))
    beams = rng.uniform(0.5, 1.5, size=(NOBS, NDIPOLE, NPIX))
    omega_B = beams.sum(axis=2)
    return masks, beams, omega_B


class TestNormalizedBeamWeights:
    def test_sums_to_one_over_sphere(self, arrays):
        _, beams, omega_B = arrays
        w = normalized_beam_weights(beams, omega_B)
        np.testing.assert_allclose(w.sum(axis=2), 1.0)

    def test_rejects_wrong_beam_ndim(self, arrays):
        _, _, omega_B = arrays
        with pytest.raises(ValueError, match="nobs, ndipole, npix"):
            normalized_beam_weights(np.ones((NOBS, NPIX)), omega_B)

    def test_rejects_mismatched_omega(self, arrays):
        _, beams, _ = arrays
        with pytest.raises(ValueError, match="omega_B must have shape"):
            normalized_beam_weights(beams, np.ones((NOBS, NDIPOLE + 1)))


class TestPointSource:
    def test_coupling_shape(self, arrays):
        masks, beams, omega_B = arrays
        c = point_source_coupling(masks, beams, omega_B, 3)
        assert c.shape == (NOBS, NDIPOLE)

    def test_fully_masked_pixel_gives_zero(self, arrays):
        masks, beams, omega_B = arrays
        masks = masks.copy()
        masks[:, 3] = 0.0
        c = point_source_coupling(masks, beams, omega_B, 3)
        np.testing.assert_allclose(c, 0.0)

    def test_matches_manual_weight(self, arrays):
        masks, beams, omega_B = arrays
        w = normalized_beam_weights(beams, omega_B)
        c = point_source_coupling(masks, beams, omega_B, 5)
        np.testing.assert_allclose(c, w[:, :, 5])

    def test_out_of_range_pixel_raises(self, arrays):
        masks, beams, omega_B = arrays
        with pytest.raises(ValueError, match="outside the map"):
            point_source_coupling(masks, beams, omega_B, NPIX)

    def test_signal_scales_with_temperature(self, arrays):
        masks, beams, omega_B = arrays
        s1 = point_source_signal(masks, beams, omega_B, 3, 100.0)
        s2 = point_source_signal(masks, beams, omega_B, 3, 200.0)
        np.testing.assert_allclose(s2, 2 * s1)

    def test_per_time_pixel_index(self, arrays):
        masks, beams, omega_B = arrays
        pix = np.arange(NOBS) % NPIX
        c = point_source_coupling(masks, beams, omega_B, pix)
        w = normalized_beam_weights(beams, omega_B)
        for t in range(NOBS):
            np.testing.assert_allclose(c[t], w[t, :, pix[t]])

    def test_source_time_cycling(self, arrays):
        masks, beams, omega_B = arrays
        n_source_times = 3
        pix = np.arange(n_source_times)
        c = point_source_coupling(
            masks, beams, omega_B, pix, n_source_times=n_source_times
        )
        assert c.shape == (NOBS, NDIPOLE)

    def test_bad_pixel_shape_raises(self, arrays):
        masks, beams, omega_B = arrays
        with pytest.raises(ValueError, match="pixel_index must be"):
            point_source_coupling(masks, beams, omega_B, np.arange(5))


class TestExtendedSource:
    def test_single_pixel_weight_matches_point_source(self, arrays):
        masks, beams, omega_B = arrays
        sw = np.zeros((NOBS, NPIX))
        sw[:, 4] = 1.0
        ext = extended_source_coupling(masks, beams, omega_B, sw)
        pt = point_source_coupling(masks, beams, omega_B, 4)
        np.testing.assert_allclose(ext, pt)

    def test_shape_mismatch_raises(self, arrays):
        masks, beams, omega_B = arrays
        with pytest.raises(ValueError, match="source_weights must have"):
            extended_source_coupling(
                masks, beams, omega_B, np.ones((NOBS, NPIX + 1))
            )

    def test_signal_shape(self, arrays):
        masks, beams, omega_B = arrays
        sw = np.full((NOBS, NPIX), 1.0 / NPIX)
        sig = extended_source_signal(
            masks, beams, omega_B, sw, np.array([100.0, 200.0])
        )
        assert sig.shape == (NOBS, NDIPOLE, 2)


class TestSurfaceEmission:
    def test_no_blocked_pixels_gives_zero(self, arrays):
        masks, beams, omega_B = arrays  # masks all ones -> nothing blocked
        out = surface_emission_signal(
            masks, beams, omega_B, np.full(NPIX, 300.0)
        )
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_fully_blocked_uniform_surface(self, arrays):
        _, beams, omega_B = arrays
        masks = np.zeros((NOBS, NPIX))
        out = surface_emission_signal(
            masks, beams, omega_B, np.full(NPIX, 300.0)
        )
        # weights sum to 1 over the sphere, all blocked -> full 300 K
        np.testing.assert_allclose(out[:, :, 0], 300.0)

    def test_residual_subtracts_scalar_model(self, arrays):
        _, beams, omega_B = arrays
        masks = np.zeros((NOBS, NPIX))
        surf = np.full(NPIX, 300.0)
        resid = surface_residual_signal(masks, beams, omega_B, surf, 300.0)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_bad_surface_shape_raises(self, arrays):
        masks, beams, omega_B = arrays
        with pytest.raises(ValueError, match="surface_temperature_K"):
            surface_emission_signal(
                masks, beams, omega_B, np.ones(NPIX + 3)
            )


class TestSyntheticLunarMap:
    def test_shape_and_mean(self):
        m = synthetic_lunar_temperature_map(4, base_K=300.0, dipole_K=20.0,
                                            quadrupole_K=0.0)
        import healpy

        assert m.shape == (healpy.nside2npix(4),)
        # dipole averages out over the sphere
        assert m.mean() == pytest.approx(300.0, abs=1.0)

    def test_deterministic(self):
        a = synthetic_lunar_temperature_map(4)
        b = synthetic_lunar_temperature_map(4)
        np.testing.assert_array_equal(a, b)


class TestGatedSourceSignals:
    def test_earth_rfi_is_zero_out_of_band(self, arrays):
        masks, beams, omega_B = arrays
        freqs = np.array([60e6, 95e6, 130e6])
        sig = earth_rfi_signal(
            masks, beams, omega_B, np.zeros(NOBS, dtype=int), freqs
        )
        assert sig.shape == (NOBS, NDIPOLE, 3)
        np.testing.assert_allclose(sig[:, :, 0], 0.0)
        np.testing.assert_allclose(sig[:, :, 2], 0.0)
        assert np.all(sig[:, :, 1] > 0)

    def test_earth_rfi_matches_temperature_model(self, arrays):
        masks, beams, omega_B = arrays
        freqs = np.array([95e6])
        sig = earth_rfi_signal(
            masks, beams, omega_B, np.zeros(NOBS, dtype=int), freqs
        )
        coupling = point_source_coupling(
            masks, beams, omega_B, np.zeros(NOBS, dtype=int)
        )
        expected = coupling * earth_rfi_temperature_K(freqs)[0]
        np.testing.assert_allclose(sig[:, :, 0], expected)

    def test_earth_rfi_tone_mode(self, arrays):
        masks, beams, omega_B = arrays
        freqs = np.array([90e6, 95e6, 100e6])
        sig = earth_rfi_signal(
            masks,
            beams,
            omega_B,
            np.zeros(NOBS, dtype=int),
            freqs,
            tone_freqs_hz=[95e6],
        )
        np.testing.assert_allclose(sig[:, :, 0], 0.0)
        assert np.all(sig[:, :, 1] > 0)

    def test_blocked_earth_gives_zero(self, arrays):
        _, beams, omega_B = arrays
        masks = np.zeros((NOBS, NPIX))
        sig = earth_rfi_signal(
            masks,
            beams,
            omega_B,
            np.zeros(NOBS, dtype=int),
            np.array([95e6]),
        )
        np.testing.assert_allclose(sig, 0.0)

    def test_quiet_sun_signal_shape(self, arrays):
        from astropy.time import Time

        masks, beams, omega_B = arrays
        times = Time("2030-01-01") + np.arange(NOBS) * 0.01
        freqs = np.array([50e6, 100e6])
        sig = quiet_sun_signal(
            masks, beams, omega_B, np.zeros(NOBS, dtype=int), freqs, times
        )
        assert sig.shape == (NOBS, NDIPOLE, 2)
        assert np.all(sig > 0)


class TestFieldStrengthConversion:
    def test_scales_as_sqrt_temperature(self):
        a = k_to_v_per_m_root_hz(100.0, 100e6)
        b = k_to_v_per_m_root_hz(400.0, 100e6)
        assert b == pytest.approx(2 * a)

    def test_scales_linearly_with_frequency(self):
        a = k_to_v_per_m_root_hz(100.0, 100e6)
        b = k_to_v_per_m_root_hz(100.0, 200e6)
        assert b == pytest.approx(2 * a)

    def test_positive(self):
        assert k_to_v_per_m_root_hz(300.0, 75e6) > 0
