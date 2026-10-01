"""Tests for the fixed-site sky / ground / offset design matrix."""

import healpy
import numpy as np
import pytest

from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import (
    HealpixBeam,
    HorizonProfile,
    build_design_matrix,
    fisher_summary,
    solve,
)

NSIDE_BEAM = 8


def _rz(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sky_rotations(ntime, tilt_deg=40.0):
    """Toy gal->top rotations: a tilted sky turning once per ``ntime`` rows."""
    tilt = np.deg2rad(tilt_deg)
    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(tilt), -np.sin(tilt)],
         [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    return np.stack(
        [rx @ _rz(2 * np.pi * t / ntime) for t in range(ntime)]
    )


def _isotropic_beam(nfreq=2):
    npix = healpy.nside2npix(NSIDE_BEAM)
    return HealpixBeam(np.ones((nfreq, npix)), np.linspace(50e6, 100e6, nfreq))


def _dipole_beam(nfreq=2):
    """Short-dipole power along body +x, mildly frequency dependent."""
    npix = healpy.nside2npix(NSIDE_BEAM)
    x = np.array(healpy.pix2vec(NSIDE_BEAM, np.arange(npix)))[0]
    maps = np.stack(
        [(1 - x**2) ** (1 + 0.2 * f) + 0.05 for f in range(nfreq)]
    )
    return HealpixBeam(maps, np.linspace(50e6, 100e6, nfreq))


def test_flat_horizon_matches_up_hemisphere():
    dirs = np.array(healpy.pix2vec(16, np.arange(healpy.nside2npix(16))))
    np.testing.assert_array_equal(
        HorizonProfile.flat().visible(dirs), dirs[2] > 0
    )


def test_horizon_profile_wraps_bearings():
    h = HorizonProfile([10.0, 350.0], [0.0, 0.2])
    # 0 deg sits halfway between 350 and 10 (through 360).
    np.testing.assert_allclose(h.elevation(0.0), 0.1)


def test_isotropic_beam_flat_horizon_sees_half_ground():
    dm = build_design_matrix(
        _isotropic_beam(), HorizonProfile.flat(), _sky_rotations(4),
        np.eye(3), nside_sky=4, nside_int=32,
    )
    # Grid pixels on the z = 0 ring count as ground.
    z = healpy.pix2vec(32, np.arange(healpy.nside2npix(32)))[2]
    np.testing.assert_allclose(dm.A[:, :, dm.ground], np.mean(z <= 0), atol=1e-12)
    np.testing.assert_allclose(dm.A[:, :, dm.sky].sum(-1), np.mean(z > 0), atol=1e-12)
    np.testing.assert_allclose(dm.A[:, :, dm.offset], 1.0)


def test_uniform_sky_ground_offset_degeneracy_is_exact():
    ntime = 6
    rots = mount_rotation(np.linspace(0, 150, ntime), 30.0, 142.164)
    dm = build_design_matrix(
        _dipole_beam(), HorizonProfile([0, 120, 240], [0.05, 0.2, -0.1]),
        _sky_rotations(ntime), rots, nside_sky=4, nside_int=32,
        offset_groups=np.array([0, 0, 0, 1, 1, 1]),
    )
    null = dm.pack(1.0, 1.0, 0.0)
    np.testing.assert_allclose(dm.predict(null), 1.0, atol=1e-12)


def test_repeated_pointings_share_configuration():
    ntime = 4
    rot_body = mount_rotation(np.array([0.0, 0.0, 90.0, 90.0]), 20.0, 0.0)
    kw = dict(nside_sky=4, nside_int=32)
    beam, horizon, sky = _dipole_beam(), HorizonProfile.flat(), _sky_rotations(ntime)
    dm = build_design_matrix(beam, horizon, sky, rot_body, **kw)
    for t in range(ntime):
        one = build_design_matrix(beam, horizon, sky[t:t + 1], rot_body[t], **kw)
        np.testing.assert_allclose(dm.A[:, t], one.A[:, 0], atol=1e-12)


def test_from_npz_reads_total_power(tmp_path):
    npix = healpy.nside2npix(NSIDE_BEAM)
    th = np.ones((3, npix))
    path = tmp_path / "beam.npz"
    np.savez(path, gain_th=th, gain_ph=2 * th, freqs=np.array([50.0, 60.0, 70.0]),
             nside=NSIDE_BEAM, empirical_frequency_mask=np.array([0, 1, 1], bool))
    beam = HealpixBeam.from_npz(path, drop_last=True)
    np.testing.assert_allclose(beam.freqs_hz, [50e6, 60e6])
    np.testing.assert_array_equal(beam.meta["empirical_frequency_mask"], [0, 1])
    vals = beam(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T)
    np.testing.assert_allclose(vals, 3.0)


def test_ground_recovered_once_offset_and_sky_prior_fixed():
    rng = np.random.default_rng(0)
    nper = 40
    az = np.repeat([0.0, 60.0, 120.0], nper)
    el = np.repeat([0.0, 45.0, 75.0], nper)
    dm = build_design_matrix(
        _dipole_beam(1), HorizonProfile([0, 90, 180, 270], [0.1, 0.3, 0.0, 0.2]),
        np.tile(_sky_rotations(nper), (3, 1, 1)),
        mount_rotation(az, el, 142.164), nside_sky=2, nside_int=32,
    )
    ntime = dm.A.shape[1]
    sky = 200 + 50 * rng.standard_normal(dm.npix)
    x_true = dm.pack(sky, 290.0, 0.0)[0]
    y = dm.A[0] @ x_true + 0.01 * rng.standard_normal(ntime)
    prior = dm.pack(1e3, np.inf, 1e-3)[0]
    fit = solve(dm.A[0], y, 0.01, prior_sigma=prior)
    g = dm.ground.start
    sigma_g = np.sqrt(fit["cov"][g, g])
    assert sigma_g < 2.0
    assert fit["x"][g] == pytest.approx(290.0, abs=4 * sigma_g)


def test_fisher_summary_reports_degenerate_direction():
    dm = build_design_matrix(
        _dipole_beam(1), HorizonProfile.flat(), _sky_rotations(8),
        np.eye(3), nside_sky=2, nside_int=32,
    )
    out = fisher_summary(dm, 0, noise_sigma=0.1)
    assert out["names"] == ["ground", "offset"]
    # Unconstrained: the uniform (sky + ground, -offset) direction is null.
    null = solve(dm.A[0], np.zeros(dm.A.shape[1]), 0.1)["null_vectors"]
    uniform = dm.pack(dm.sky_observed.astype(float), 1.0, -1.0)[0]
    coef, *_ = np.linalg.lstsq(null, uniform, rcond=None)
    np.testing.assert_allclose(null @ coef, uniform, atol=1e-6)
    assert np.all(np.isinf(out["sigma"]))
    assert np.isinf(out["sigma_sky_mean"])
    # Pinning the offset is not enough for one fixed pointing: the ground
    # column is constant, so it still trades against a uniform sky shift.
    pinned = fisher_summary(dm, 0, noise_sigma=0.1,
                            prior_sigma=dm.pack(np.inf, np.inf, 1.0)[0])
    assert np.isinf(pinned["sigma"][0])
    assert pinned["sigma"][1] == pytest.approx(1.0)


def _three_pointing_dm(nper=40, nside_sky=2):
    az = np.repeat([0.0, 60.0, 120.0], nper)
    el = np.repeat([0.0, 45.0, 75.0], nper)
    return build_design_matrix(
        _dipole_beam(1),
        HorizonProfile([0, 90, 180, 270], [0.1, 0.3, 0.0, 0.2]),
        np.tile(_sky_rotations(nper), (3, 1, 1)),
        mount_rotation(az, el, 142.164), nside_sky=nside_sky, nside_int=32,
    )


def test_sky_template_column_is_beam_weighted_map():
    dm = _three_pointing_dm()
    sky_map = np.arange(dm.npix, dtype=float)[None]
    dmt = dm.with_sky_templates(sky_map, names=["gsm"])
    assert dmt.column_names[-1] == "gsm"
    assert dmt.template == slice(dm.A.shape[2], dm.A.shape[2] + 1)
    np.testing.assert_allclose(
        dmt.A[0, :, dmt.template.start], dm.A[0, :, dm.sky] @ sky_map[0]
    )
    # Zero template amplitude reproduces the original prediction.
    x = dm.pack(sky_map, 290.0, 3.0)
    np.testing.assert_allclose(
        dmt.predict(dmt.pack(sky_map, 290.0, 3.0, template=0.0)),
        dm.predict(x),
    )


def test_sky_template_absorbs_scale_error_instead_of_ground():
    rng = np.random.default_rng(1)
    dm = _three_pointing_dm()
    gsm = 200 + 50 * rng.standard_normal(dm.npix)
    eta = 0.85  # the data see a scaled sky, e.g. through a lossy front end
    y = dm.A[0] @ dm.pack(eta * gsm, 290.0, 0.0)[0]
    g = dm.ground.start

    # Prior centred on the unscaled map: the scale error lands in ground.
    centred = dm.pack(0.02 * gsm, np.inf, 1e-3)[0]
    fit = solve(dm.A[0], y, 0.01, prior_sigma=centred,
                prior_mean=dm.pack(gsm, 0.0, 0.0)[0])
    bias_centred = abs(fit["x"][g] - 290.0)
    assert bias_centred > 5.0

    # Template amplitude free, prior only on the residual sky.
    dmt = dm.with_sky_templates(gsm[None])
    prior = dmt.pack(0.02 * gsm, np.inf, 1e-3, template=np.inf)[0]
    fit = solve(dmt.A[0], y, 0.01, prior_sigma=prior)
    assert fit["x"][dmt.template.start] == pytest.approx(eta, abs=0.01)
    assert abs(fit["x"][g] - 290.0) < 0.1 * bias_centred


def test_select_rows_keeps_columns_and_recomputes_observed():
    dm = _three_pointing_dm().with_sky_templates(np.ones((1, 1, 48)))
    sub = dm.select_rows(np.arange(10))
    assert sub.A.shape == (1, 10, dm.A.shape[2])
    assert sub.template_names == dm.template_names
    np.testing.assert_array_equal(
        sub.sky_observed, np.any(dm.A[:, :10, dm.sky] > 0, axis=(0, 1))
    )


def test_solve_is_independent_of_column_units():
    rng = np.random.default_rng(2)
    dm = _three_pointing_dm()
    gsm = 200 + 50 * rng.standard_normal(dm.npix)
    dmt = dm.with_sky_templates(gsm[None])
    delta = 0.1 * gsm * rng.standard_normal(dm.npix)
    y = dmt.A[0] @ dmt.pack(delta, 290.0, 5.0, template=0.9)[0]
    y = y + 0.01 * rng.standard_normal(len(y))
    prior = dmt.pack(0.1 * gsm, np.inf, np.inf, template=np.inf)[0]
    ref = solve(dmt.A[0], y, 0.01, prior_sigma=prior)

    scale = np.ones(dmt.A.shape[2])
    scale[dmt.template] = 1e6  # template column in wildly different units
    scaled = solve(dmt.A[0] * scale, y, 0.01, prior_sigma=prior / scale)
    np.testing.assert_array_equal(scaled["constrained"], ref["constrained"])
    np.testing.assert_allclose(scaled["x"] * scale, ref["x"], rtol=1e-4, atol=1e-4)
    assert ref["constrained"][dmt.ground.start]
    a = dmt.template.start
    assert ref["x"][a] == pytest.approx(0.9, abs=3 * np.sqrt(ref["cov"][a, a]))


def test_solve_flags_ground_offset_degeneracy_at_one_pointing():
    dm = build_design_matrix(
        _dipole_beam(1), HorizonProfile.flat(), _sky_rotations(30),
        np.eye(3), nside_sky=2, nside_int=32,
    )
    rng = np.random.default_rng(3)
    gsm = 200 + 50 * rng.standard_normal(dm.npix)
    dmt = dm.with_sky_templates(gsm[None])
    prior = dmt.pack(0.1 * gsm, np.inf, np.inf, template=np.inf)[0]
    y = dmt.A[0] @ dmt.pack(0.0, 290.0, 5.0, template=1.0)[0]
    sol = solve(dmt.A[0], y, 0.01, prior_sigma=prior)
    assert not sol["constrained"][dmt.ground.start]
    assert not sol["constrained"][dmt.offset.start]
    assert sol["constrained"][dmt.template.start]
    # The pedestal they share is still determined.
    f_gnd = dmt.A[0, 0, dmt.ground.start]
    pedestal = f_gnd * sol["x"][dmt.ground.start] + sol["x"][dmt.offset.start]
    assert pedestal == pytest.approx(f_gnd * 290.0 + 5.0, abs=0.5)
