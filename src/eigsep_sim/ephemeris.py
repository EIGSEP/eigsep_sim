"""Solar-system body ephemeris for external radio sources (Sun, Earth, ...).

Companion to :mod:`observer` — where ``Observer`` subclasses give the
spacecraft's *own* position/attitude, this module gives the position of
everything else (Sun, Earth) relative to the Moon, in the same inertial
galactic frame used throughout the rest of the package (``ICRS2GAL``).

Uses astropy's built-in low-precision analytical ephemeris
(``get_body_barycentric``) — no external kernel download required, and
accurate to well under a HEALPix pixel at ``nside <= 64`` (~1 deg), which is
all that's needed to place an extended source (Sun, Earth) on the sky and
gate it by lunar occultation.  Swap in a JPL DE44x kernel via
``astropy.coordinates.solar_system_ephemeris.set('jpl')`` (requires the
``jplephem`` package and a downloaded kernel) if higher precision is ever
needed — not expected for this use case.

Note on frames: the *inertial* (galactic) direction from the Moon to Earth
sweeps a full circle roughly once per sidereal month (~27.3 days, the
Moon's orbital period around Earth) — it is only in the Moon **body-fixed**
frame (MCMF) that Earth appears nearly stationary (that's what tidal
locking means).  The Sun's inertial direction from the Moon sweeps once per
synodic month (~29.5 days); in the MCMF frame this sweep *is* the lunar
day/night cycle relevant to surface heating.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric
import astropy.units as u

from eigsep_base.const import R_MOON, R_SUN, R_EARTH
from healjax.coord import rot_m

from .observer import ICRS2GAL, _moon_icrs2mcmf

_BODY_RADII_M = {"sun": R_SUN, "earth": R_EARTH}


def body_directions_gal(times, bodies=("sun", "earth")):
    """Unit vectors + distances from the Moon's centre to ``bodies``.

    Parameters
    ----------
    times : Time-like, shape (ntimes,)
    bodies : sequence of str
        Names understood by ``astropy.coordinates.get_body_barycentric``
        (e.g. ``"sun"``, ``"earth"``).

    Returns
    -------
    dirs : dict[str, ndarray (ntimes, 3)]
        Unit vectors, galactic frame, Moon centre -> body.
    dists : dict[str, ndarray (ntimes,)]
        Distances [m].
    """
    times = Time(times)
    moon_pos = get_body_barycentric("moon", times)
    dirs, dists = {}, {}
    for body in bodies:
        rel = get_body_barycentric(body, times) - moon_pos
        xyz = rel.xyz.to(u.m).value  # (3, ntimes)
        d = np.linalg.norm(xyz, axis=0)
        u_icrs = xyz / d
        u_gal = ICRS2GAL @ u_icrs  # (3, ntimes)
        dirs[body] = np.ascontiguousarray(u_gal.T, dtype=np.float64)  # (ntimes, 3)
        dists[body] = d
    return dirs, dists


def body_angular_radius(body, dist_m):
    """Angular radius [rad] of ``body`` at distance ``dist_m`` (small-angle)."""
    return _BODY_RADII_M[body] / np.asarray(dist_m)


def body_occulted_by_moon(orbit, times, body_dir_gal):
    """Boolean mask, shape (ntimes,): True where the Moon blocks the line of
    sight from ``orbit``'s spacecraft position to ``body_dir_gal[t]``.

    Same eclipse geometry as ``LunarOrbit.above_horizon_stack``, evaluated
    for one time-varying direction (an external body) instead of a fixed
    HEALPix grid.  Parallax between "from Moon centre" and "from spacecraft"
    is neglected, consistent with how the rest of the pipeline treats the
    sky as being at infinity (valid here too: orbital radius is ~1e-2x the
    Earth distance and ~1e-5x the Sun distance).

    Parameters
    ----------
    orbit : LunarOrbit
    times : Time-like, shape (ntimes,)
    body_dir_gal : ndarray, shape (ntimes, 3)
        Unit vectors, galactic frame (e.g. from ``body_directions_gal``).

    Returns
    -------
    occulted : ndarray of bool, shape (ntimes,)
    """
    pos = orbit.spacecraft_position_stack(times)  # (ntimes, 3) m
    d = np.linalg.norm(pos, axis=1)
    moon_dir = pos / d[:, None]
    dot = np.sum(moon_dir * body_dir_gal, axis=1)
    limb_dot = -np.sqrt(np.maximum(0.0, 1.0 - (R_MOON / d) ** 2))
    return dot <= limb_dot


def moon_surface_intersection_mcmf(orbit, times, sky_dirs_gal):
    """Where an occulted line of sight actually terminates on the Moon.

    For each (time, sky direction) finds the near intersection of the ray
    from the spacecraft along that direction with the Moon's sphere, and
    returns it as a body-fixed (MCMF) unit surface-normal vector -- i.e.
    the selenographic point a blocked sky pixel's brightness should
    physically be evaluated at (see :mod:`regolith`), rather than treating
    every occulted pixel as an undifferentiated single temperature.

    Parameters
    ----------
    orbit : LunarOrbit
    times : Time-like, shape (ntimes,)
    sky_dirs_gal : ndarray, shape (ntimes, npix, 3) or (npix, 3)
        Galactic-frame unit vectors for each sky pixel; a (npix, 3) array
        is broadcast across all times.

    Returns
    -------
    normals_mcmf : ndarray, shape (ntimes, npix, 3)
        Body-fixed unit surface-normal vectors at the intersection point.
        ``NaN`` where the ray does not actually intersect the sphere (not
        occulted) -- callers should already have an occultation mask (e.g.
        from ``LunarOrbit.above_horizon_stack``) and only use this where
        that mask is False.
    """
    times = Time(times)
    ntimes = len(times)
    pos = orbit.spacecraft_position_stack(times)  # (ntimes, 3) m
    d = np.linalg.norm(pos, axis=1)  # (ntimes,)

    sky_dirs_gal = np.asarray(sky_dirs_gal, dtype=np.float64)
    if sky_dirs_gal.ndim == 2:
        sky_dirs_gal = np.broadcast_to(
            sky_dirs_gal, (ntimes,) + sky_dirs_gal.shape
        )

    # Ray-sphere intersection: |pos + t*u|^2 = R_MOON^2, near root, t>0
    # (a mathematically valid root behind the spacecraft, e.g. for a ray
    # pointing away from the Moon, does not count as a hit).
    pos_dot_u = np.einsum("ti,tpi->tp", pos, sky_dirs_gal)  # (ntimes, npix)
    disc = pos_dot_u**2 - (d[:, None] ** 2 - R_MOON**2)
    t_near = -pos_dot_u - np.sqrt(np.maximum(disc, 0.0))  # (ntimes, npix)
    valid = (disc >= 0.0) & (t_near > 0.0)

    intersection_gal = (
        pos[:, None, :] + t_near[..., None] * sky_dirs_gal
    )  # (ntimes, npix, 3)
    normal_gal = intersection_gal / np.linalg.norm(
        intersection_gal, axis=-1, keepdims=True
    )
    normal_gal[~valid] = np.nan

    gal2icrs = ICRS2GAL.T
    out = np.empty_like(normal_gal)
    for i, t in enumerate(times):
        icrs2mcmf = _moon_icrs2mcmf(t)
        out[i] = (icrs2mcmf @ (gal2icrs @ normal_gal[i].T)).T
    return out


# ---------------------------------------------------------------------------
# Position-based geometry (absorbed from bloom21cm/src/moon_geometry.py)
#
# The functions above take a ``LunarOrbit`` and a ``Time`` array; the ones
# below take raw spacecraft position vectors instead, which is what the
# quiet-window analysis scripts and sky-coverage figures need when there is
# no orbit object in hand.  They implement the same ray-sphere test.
# ---------------------------------------------------------------------------


def body_direction_from_moon_gal(body, obs_times):
    """Unit vector(s) from the Moon toward ``body``, galactic Cartesian.

    Single-body, direction-only counterpart to :func:`body_directions_gal`
    (which returns dicts of directions *and* distances for several bodies at
    once).  Accepts a scalar ``Time`` and then returns a bare ``(3,)`` vector,
    which :func:`body_directions_gal` does not.

    Parameters
    ----------
    body : str
        Any body name accepted by
        ``astropy.coordinates.get_body_barycentric`` (e.g. ``"earth"``,
        ``"sun"``).
    obs_times : astropy.time.Time
        Scalar or array-valued.

    Returns
    -------
    ndarray, shape (3,) or (n_times, 3)
    """
    r_body = get_body_barycentric(body, obs_times)
    r_moon = get_body_barycentric("moon", obs_times)
    vec = np.column_stack([
        (r_body.x - r_moon.x).to_value(u.au),
        (r_body.y - r_moon.y).to_value(u.au),
        (r_body.z - r_moon.z).to_value(u.au),
    ])
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    sc = SkyCoord(x=vec[:, 0], y=vec[:, 1], z=vec[:, 2],
                  representation_type="cartesian", frame="icrs")
    gal = sc.galactic.cartesian
    out = np.column_stack([gal.x.value, gal.y.value, gal.z.value])
    return out[0] if obs_times.isscalar else out


def moon_limb_cos_angle(spacecraft_pos_m):
    """cos(theta_moon) at each position, theta_moon = arcsin(R_MOON / d).

    Parameters
    ----------
    spacecraft_pos_m : ndarray, shape (3,) or (n, 3)
        Spacecraft position(s) relative to the Moon centre [m].

    Returns
    -------
    float or ndarray, shape (n,)
    """
    pos = np.asarray(spacecraft_pos_m, dtype=float)
    d = np.linalg.norm(pos, axis=-1)
    return np.sqrt(np.maximum(0.0, 1.0 - (R_MOON / d) ** 2))


def earth_illuminated_fraction(obs_times):
    """Earth's Sun-illuminated fraction as seen from the Moon, per time.

    Phase angle is the Sun-Earth-Moon angle (Moon as observer); illuminated
    fraction follows the standard ``(1 + cos(phase)) / 2`` convention used
    for planetary phase (full at phase 0, new at phase 180 deg).

    Parameters
    ----------
    obs_times : astropy.time.Time
        Scalar or array-valued.

    Returns
    -------
    float or ndarray, shape (n_times,)
        Illuminated fraction in [0, 1].
    """
    sun_dir = np.atleast_2d(body_direction_from_moon_gal("sun", obs_times))
    earth_dir = np.atleast_2d(body_direction_from_moon_gal("earth", obs_times))
    cos_phase = np.sum(sun_dir * earth_dir, axis=-1)
    frac = 0.5 * (1.0 + cos_phase)
    return float(frac[0]) if obs_times.isscalar else frac


def occulted_by_moon(spacecraft_pos_m, direction_gal):
    """Whether ``direction_gal`` is blocked by the Moon from a position.

    Same ray-sphere test as :func:`body_occulted_by_moon` and
    ``LunarOrbit.above_horizon_stack``, but driven by explicit position
    vectors rather than an orbit object, and applied to an arbitrary batch
    of directions rather than a fixed HEALPix grid.

    Parameters
    ----------
    spacecraft_pos_m : ndarray, shape (3,) or (n, 3)
        Spacecraft position(s) relative to the Moon centre [m].
    direction_gal : ndarray, shape (3,) or (n, 3)
        Unit direction(s) in galactic Cartesian coordinates (e.g. from
        :func:`body_direction_from_moon_gal`).  Broadcasts against
        ``spacecraft_pos_m``.

    Returns
    -------
    bool or ndarray of bool, shape (n,)
        True where the direction is occulted (blocked by the Moon).
    """
    pos = np.asarray(spacecraft_pos_m, dtype=float)
    d = np.linalg.norm(pos, axis=-1, keepdims=True)
    r_hat = pos / d
    direction = np.asarray(direction_gal, dtype=float)
    dot = np.sum(r_hat * direction, axis=-1)
    return dot <= -moon_limb_cos_angle(pos)


# ---------------------------------------------------------------------------
# Ray/surface helpers (absorbed from the former eigsep_sim.utils)
# ---------------------------------------------------------------------------


def moon_surface_distance(angle, d, r=R_MOON):
    """Distance to the lunar surface along a ray offset by ``angle``.

    Near root of the ray-sphere intersection for a ray leaving an observer
    at distance ``d`` from the Moon centre, making angle ``angle`` with the
    direction to that centre.

    Parameters
    ----------
    angle : array_like
        Angle between the ray and the observer->Moon-centre direction [rad].
    d : float or array_like
        Observer distance from the Moon centre [m].
    r : float, optional
        Moon radius [m].

    Returns
    -------
    ndarray
        Distance along the ray to the surface [m]; ``NaN`` where the ray
        misses the Moon or the intersection lies behind the observer.
    """
    a, b, c = 1, -2 * d * np.cos(angle), d**2 - r**2
    radical = b**2 - 4 * a * c
    ans = np.where(
        radical > 0, -b - np.sqrt(radical.clip(0)) / (2 * a), np.nan
    )
    return np.where(ans > 0, ans, np.nan)


def moon_reflect_vector(vec_to_surface, moon_pos, r=R_MOON):
    """Specular reflection of a ray off the lunar sphere.

    Parameters
    ----------
    vec_to_surface : ndarray, shape (3, n)
        Vectors from the observer to the surface intersection points.
    moon_pos : ndarray, shape (3,)
        Moon centre position in the same frame.
    r : float, optional
        Moon radius [m], used to normalise the surface normal.

    Returns
    -------
    ndarray, shape (3, n)
        Outgoing (reflected) unit direction vectors.
    """
    incident = vec_to_surface / np.linalg.norm(vec_to_surface, axis=0)
    normal = (vec_to_surface - moon_pos[:, None]) / r
    return incident - 2 * np.einsum("ij,ij->j", incident, normal) * normal


def sample_disk(pos, r_ang, nsamples):
    """Uniformly sample directions within a cone of radius ``r_ang``.

    The cone is centred on ``pos``.  Sampling is uniform in solid angle, so
    this is the right primitive for Monte-Carlo integration over an extended
    disk-like source (Sun, Earth) rather than treating it as a point.

    Parameters
    ----------
    pos : ndarray, shape (3,)
        Direction of the cone axis (need not be normalised).
    r_ang : float
        Angular radius of the cone [rad].
    nsamples : int
        Number of directions to draw.

    Returns
    -------
    ndarray, shape (3, nsamples)
        Unit vectors.
    """
    cos_theta = np.random.uniform(np.cos(r_ang), 1, nsamples)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, nsamples)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    samples = np.vstack((x, y, z))
    pos = pos / np.linalg.norm(pos)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, pos)
    rot_axis /= np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(z_axis, pos))
    return rot_m(rot_angle, rot_axis) @ samples
