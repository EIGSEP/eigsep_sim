"""Tests for the position-based moon geometry absorbed into
eigsep_sim.ephemeris (from bloom21cm/src/moon_geometry.py) and the ray
helpers absorbed from the former eigsep_sim.utils.
"""

import numpy as np
import pytest
from astropy.time import Time

from eigsep_base.const import R_MOON
from eigsep_sim.ephemeris import (
    body_direction_from_moon_gal,
    earth_illuminated_fraction,
    moon_limb_cos_angle,
    moon_reflect_vector,
    moon_surface_distance,
    occulted_by_moon,
    sample_disk,
)

ALT = 100e3  # 100 km orbit altitude
D = R_MOON + ALT


class TestMoonLimbCosAngle:
    def test_at_surface_is_zero(self):
        pos = np.array([R_MOON, 0.0, 0.0])
        assert moon_limb_cos_angle(pos) == pytest.approx(0.0, abs=1e-12)

    def test_tends_to_one_far_away(self):
        pos = np.array([1e12, 0.0, 0.0])
        assert moon_limb_cos_angle(pos) == pytest.approx(1.0)

    def test_matches_analytic(self):
        pos = np.array([D, 0.0, 0.0])
        expected = np.sqrt(1.0 - (R_MOON / D) ** 2)
        assert moon_limb_cos_angle(pos) == pytest.approx(expected)

    def test_vectorized(self):
        pos = np.array([[D, 0, 0], [0, D, 0], [0, 0, 2 * D]], dtype=float)
        got = moon_limb_cos_angle(pos)
        assert got.shape == (3,)
        assert got[2] > got[0]


class TestOccultedByMoon:
    def test_straight_up_is_not_occulted(self):
        pos = np.array([D, 0.0, 0.0])
        assert not occulted_by_moon(pos, np.array([1.0, 0.0, 0.0]))

    def test_straight_down_is_occulted(self):
        pos = np.array([D, 0.0, 0.0])
        assert occulted_by_moon(pos, np.array([-1.0, 0.0, 0.0]))

    def test_tangent_is_the_boundary(self):
        pos = np.array([D, 0.0, 0.0])
        cos_limb = moon_limb_cos_angle(pos)
        # just inside the limb -> occulted; just outside -> visible
        inside = np.array([-cos_limb - 1e-6, np.sqrt(1 - cos_limb**2), 0.0])
        inside /= np.linalg.norm(inside)
        outside = np.array([-cos_limb + 1e-3, np.sqrt(1 - cos_limb**2), 0.0])
        outside /= np.linalg.norm(outside)
        assert occulted_by_moon(pos, inside)
        assert not occulted_by_moon(pos, outside)

    def test_vectorized_over_positions(self):
        pos = np.array([[D, 0, 0], [-D, 0, 0]], dtype=float)
        direction = np.array([[1.0, 0, 0], [1.0, 0, 0]])
        got = occulted_by_moon(pos, direction)
        assert got.tolist() == [False, True]


class TestBodyDirectionFromMoonGal:
    def test_unit_norm_scalar_time(self):
        t = Time("2030-03-01T00:00:00")
        v = body_direction_from_moon_gal("earth", t)
        assert v.shape == (3,)
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_unit_norm_array_time(self):
        t = Time("2030-03-01T00:00:00") + np.arange(4) * 0.25
        v = body_direction_from_moon_gal("earth", t)
        assert v.shape == (4, 3)
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)

    def test_earth_direction_sweeps_over_a_month(self):
        t = Time("2030-03-01T00:00:00") + np.array([0.0, 14.0])
        v = body_direction_from_moon_gal("earth", t)
        # roughly antipodal half a sidereal month later
        assert np.dot(v[0], v[1]) < 0

    def test_agrees_with_body_directions_gal(self):
        from eigsep_sim.ephemeris import body_directions_gal

        t = Time("2030-03-01T00:00:00") + np.arange(3) * 0.5
        a = body_direction_from_moon_gal("sun", t)
        dirs, _ = body_directions_gal(t, bodies=("sun",))
        # same geometry via two different frame paths; ~arcmin agreement
        cos = np.sum(a * dirs["sun"], axis=1)
        np.testing.assert_allclose(cos, 1.0, atol=1e-4)


class TestEarthIlluminatedFraction:
    def test_within_unit_interval(self):
        t = Time("2030-01-01T00:00:00") + np.arange(30) * 1.0
        frac = earth_illuminated_fraction(t)
        assert np.all((frac >= 0.0) & (frac <= 1.0))

    def test_scalar_time_returns_float(self):
        frac = earth_illuminated_fraction(Time("2030-01-01T00:00:00"))
        assert isinstance(frac, float)

    def test_varies_over_a_synodic_month(self):
        t = Time("2030-01-01T00:00:00") + np.arange(30) * 1.0
        frac = earth_illuminated_fraction(t)
        assert frac.max() - frac.min() > 0.5


class TestMoonSurfaceDistance:
    def test_misses_return_nan(self):
        # looking directly away from the Moon never hits it
        angle = np.array([np.pi])
        assert np.all(np.isnan(moon_surface_distance(angle, D)))

    def test_head_on_ray_hits(self):
        angle = np.array([0.0])
        got = moon_surface_distance(angle, D)
        assert np.all(np.isfinite(got))
        assert np.all(got > 0)


class TestMoonReflectVector:
    def test_normal_incidence_reverses(self):
        moon_pos = np.zeros(3)
        # ray arriving along -x at the +x pole of the Moon
        vec = np.array([[R_MOON], [0.0], [0.0]])
        out = moon_reflect_vector(vec, moon_pos)
        np.testing.assert_allclose(out[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)

    def test_output_is_unit_length(self):
        moon_pos = np.zeros(3)
        rng = np.random.default_rng(0)
        v = rng.normal(size=(3, 5))
        v = R_MOON * v / np.linalg.norm(v, axis=0)
        out = moon_reflect_vector(v, moon_pos)
        np.testing.assert_allclose(np.linalg.norm(out, axis=0), 1.0)


class TestSampleDisk:
    def test_shape_and_unit_norm(self):
        s = sample_disk(np.array([0.0, 0.0, 1.0]) + 1e-9, 0.1, 50)
        assert s.shape == (3, 50)
        np.testing.assert_allclose(np.linalg.norm(s, axis=0), 1.0, atol=1e-12)

    def test_samples_lie_within_cone(self):
        axis = np.array([1.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        r_ang = 0.2
        s = sample_disk(axis, r_ang, 200)
        cos = axis @ s
        assert np.all(cos >= np.cos(r_ang) - 1e-9)

    def test_wider_cone_spreads_further(self):
        axis = np.array([1.0, 0.5, 0.0])
        narrow = sample_disk(axis, 0.05, 300)
        wide = sample_disk(axis, 0.5, 300)
        u = axis / np.linalg.norm(axis)
        assert (u @ wide).min() < (u @ narrow).min()
