"""Tests for eigsep_sim.reflectivity (absorbed from eigsep_terrain)."""

import warnings

import numpy as np
import pytest

from eigsep_sim.reflectivity import (
    TERRAIN_TYPES,
    complex_permittivity,
    complex_ref_index,
    conductivity_from_resistivity,
    eps_r_from_gpr,
    permittivity_from_conductivity,
    reflection_coefficient,
    terrain_reflection_coefficient,
)

FREQS = np.linspace(50e6, 250e6, 21)


class TestConductivity:
    def test_inverse_scaling(self):
        assert conductivity_from_resistivity(1e3) == pytest.approx(
            10 * conductivity_from_resistivity(1e4)
        )

    def test_positive(self):
        assert conductivity_from_resistivity(1e4) > 0


class TestPermittivity:
    def test_lossless_limit_is_real(self):
        eps = complex_permittivity(5.0, 0.0, FREQS)
        np.testing.assert_allclose(eps.imag, 0.0)
        np.testing.assert_allclose(eps.real, 5.0)

    def test_loss_term_is_negative_imaginary(self):
        eps = complex_permittivity(5.0, 1e3, FREQS)
        assert np.all(eps.imag < 0)

    def test_loss_falls_with_frequency(self):
        eps = complex_permittivity(5.0, 1e3, FREQS)
        assert np.all(np.diff(np.abs(eps.imag)) < 0)

    def test_ref_index_is_sqrt_of_permittivity(self):
        n = complex_ref_index(5.0, 1e3, FREQS)
        np.testing.assert_allclose(
            n**2, complex_permittivity(5.0, 1e3, FREQS)
        )


class TestReflectionCoefficient:
    def test_matched_media_gives_zero(self):
        assert reflection_coefficient(1.0, eta0=1.0) == 0

    def test_magnitude_bounded_for_passive_media(self):
        for name, t in TERRAIN_TYPES.items():
            r = terrain_reflection_coefficient(name, FREQS)
            assert np.all(np.abs(r) <= 1.0 + 1e-12), name

    def test_denser_medium_reflects_more(self):
        # wet soil (eps_r 22) should reflect more than dry sand (eps_r 4)
        wet = np.abs(terrain_reflection_coefficient("wet_soil", FREQS))
        dry = np.abs(terrain_reflection_coefficient("dry_sand", FREQS))
        assert np.all(wet > dry)

    def test_unknown_terrain_raises(self):
        with pytest.raises(ValueError, match="Unknown terrain type"):
            terrain_reflection_coefficient("cheese", FREQS)

    def test_lunar_regolith_is_low_loss(self):
        r = terrain_reflection_coefficient("lunar_regolith", FREQS)
        # eps_r 3.0, essentially lossless -> r ~ (1-sqrt(3))/(1+sqrt(3))
        expected = (1 - np.sqrt(3.0)) / (1 + np.sqrt(3.0))
        np.testing.assert_allclose(r.real, expected, rtol=1e-3)


class TestGPR:
    def test_roundtrip_against_known_permittivity(self):
        from eigsep_base.const import c

        eps_r = 9.0
        depth = 2.0
        v = c / np.sqrt(eps_r)
        travel_time = 2 * depth / v
        assert eps_r_from_gpr(depth, travel_time) == pytest.approx(
            eps_r, rel=1e-6
        )

    def test_vacuum_gives_unity(self):
        from eigsep_base.const import c

        assert eps_r_from_gpr(1.0, 2.0 / c) == pytest.approx(1.0, rel=1e-6)


class TestDeprecatedAlias:
    def test_permittivity_from_conductivity_warns(self):
        with pytest.warns(DeprecationWarning):
            permittivity_from_conductivity(1e3, FREQS)

    def test_matches_complex_ref_index_with_unit_eps(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            got = permittivity_from_conductivity(1e3, FREQS)
        np.testing.assert_allclose(got, complex_ref_index(1, 1e3, FREQS))
