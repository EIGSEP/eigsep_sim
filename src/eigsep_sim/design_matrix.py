"""Linear design matrices for a fixed-site radiometer: sky, ground, offsets.

The ground-based counterpart of the lunar-orbit recovery in ``recovery.py``.
At each frequency the measured antenna temperature is modelled as

    T(t) = sum_p A_sky[t, p] T_sky[p] + sum_k A_gnd[t, k] T_gnd[k] + T_off[g(t)]

with the sky in Galactic HEALPix pixels, the ground (everything below the
terrain horizon) as one or more uniform-temperature regions, and an additive
offset per row group. The beam is an assumed input, so the problem is linear
and is solved independently per frequency.

Integration happens on a fine topocentric HEALPix grid (``nside_int``). Beam
and horizon are fixed in that frame for a given pointing, so they are
evaluated once per distinct (pointing, horizon) configuration; only the
mapping of grid directions onto Galactic sky pixels changes with time. Rows
are normalized by the full-sphere beam integral, so the sky and ground weights
of each row sum to one. That makes one degeneracy exact: adding a constant to
every sky pixel and every ground region, and subtracting it from the offset,
leaves every row unchanged. Breaking it needs prior knowledge of the offset
(an absolute calibration) or of the sky.

Column ordering: ``[sky pixels (npix) | ground regions | offset groups]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import healpy
import numpy as np
import scipy.sparse

__all__ = [
    "HealpixBeam",
    "HorizonProfile",
    "DesignMatrix",
    "build_design_matrix",
    "solve",
    "fisher_summary",
]


class HealpixBeam:
    """Body-frame beam power on HEALPix, one map per frequency.

    The body frame is that of ``eigsep_base.rotations``: boresight +z, dipole
    arm +x. Any object with ``freqs_hz`` and ``__call__(dirs_body)`` returning
    ``(nfreq, n)`` power can stand in for this class in
    :func:`build_design_matrix`. Overall normalization is irrelevant there.

    Parameters
    ----------
    maps : ndarray, shape (nfreq, npix)
        Beam power maps.
    freqs_hz : ndarray, shape (nfreq,)
    meta : dict, optional
        Free-form provenance (source file, frequency mask, ...).
    """

    def __init__(self, maps, freqs_hz, meta=None):
        self.maps = np.atleast_2d(np.asarray(maps, dtype=float))
        self.freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        if self.maps.shape[0] != self.freqs_hz.size:
            raise ValueError(
                f"maps has {self.maps.shape[0]} frequencies, "
                f"freqs_hz has {self.freqs_hz.size}"
            )
        self.nside = healpy.npix2nside(self.maps.shape[1])
        self.meta = {} if meta is None else dict(meta)

    @classmethod
    def from_npz(cls, path, drop_last=False):
        """Load the HFSS / empirical-beam npz format.

        Reads ``gain_th + gain_ph`` (total power) and ``freqs`` in MHz. This is
        the format of ``data-analysis/hfss_beam_maps/bowtie_beam.npz`` and of
        the ``empirical_beam.npz`` files under
        ``marjum-2026-07/derived/beam/empirical_raster_vNNNN/``. For the
        empirical beams, ``meta["empirical_frequency_mask"]`` marks the slices
        that were fitted (the rest are HFSS fill).
        """
        with np.load(path) as npz:
            maps = npz["gain_th"] + npz["gain_ph"]
            freqs_hz = npz["freqs"] * 1e6
            meta = {"path": str(path)}
            if "empirical_frequency_mask" in npz:
                meta["empirical_frequency_mask"] = npz[
                    "empirical_frequency_mask"
                ]
        if drop_last:
            maps, freqs_hz = maps[:-1], freqs_hz[:-1]
            if "empirical_frequency_mask" in meta:
                meta["empirical_frequency_mask"] = meta[
                    "empirical_frequency_mask"
                ][:-1]
        return cls(maps, freqs_hz, meta=meta)

    def select(self, freq_indices):
        """Return a beam restricted to ``freq_indices``."""
        idx = np.atleast_1d(freq_indices)
        meta = dict(self.meta)
        if "empirical_frequency_mask" in meta:
            meta["empirical_frequency_mask"] = meta[
                "empirical_frequency_mask"
            ][idx]
        return HealpixBeam(self.maps[idx], self.freqs_hz[idx], meta=meta)

    def __call__(self, dirs_body):
        """Bilinearly interpolated power at body-frame unit vectors ``(3, n)``.

        Returns ``(nfreq, n)``.
        """
        theta, phi = healpy.vec2ang(np.asarray(dirs_body).T)
        pix, wgt = healpy.get_interp_weights(self.nside, theta, phi)
        return np.einsum("fkn,kn->fn", self.maps[:, pix], wgt)


class HorizonProfile:
    """Terrain horizon as elevation versus bearing, topocentric ENU.

    Parameters
    ----------
    bearings_deg : ndarray
        Bearings in degrees counter-clockwise from East, ``atan2(N, E)``
        (the ``horizon_profiles`` convention).
    elev_rad : ndarray
        Horizon elevation above horizontal at each bearing, radians.
    """

    def __init__(self, bearings_deg, elev_rad):
        b = np.mod(np.asarray(bearings_deg, dtype=float), 360.0)
        order = np.argsort(b)
        self.bearings_deg = b[order]
        self.elev_rad = np.asarray(elev_rad, dtype=float)[order]

    @classmethod
    def flat(cls, elev_deg=0.0):
        """A horizon at constant elevation (0 = the geometric horizon)."""
        return cls([0.0, 180.0], np.deg2rad([elev_deg, elev_deg]))

    @classmethod
    def from_npz(cls, path, era):
        """Load one height era from a ``horizon_profiles_vNNNN.npz``.

        ``era`` is the key suffix, e.g. ``"30m"``, ``"87.5m"``, ``"91m"``.
        """
        with np.load(path) as npz:
            return cls(npz["bearings_deg"], npz[f"elev_rad_{era}"])

    def elevation(self, bearing_deg):
        """Horizon elevation (radians) at ``bearing_deg``, periodic in 360."""
        return np.interp(
            np.mod(bearing_deg, 360.0),
            self.bearings_deg,
            self.elev_rad,
            period=360.0,
        )

    def visible(self, dirs_enu):
        """True where ENU unit vectors ``(3, n)`` are above the horizon."""
        x, y, z = np.asarray(dirs_enu)
        elev = np.arcsin(np.clip(z, -1.0, 1.0))
        bearing = np.rad2deg(np.arctan2(y, x))
        return elev > self.elevation(bearing)


@dataclass
class DesignMatrix:
    """Per-frequency design matrix and its column bookkeeping.

    ``A`` has shape ``(nfreq, ntime, ncol)``; columns are
    ``[sky pixels | ground regions | offset groups]``.
    """

    A: np.ndarray
    freqs_hz: np.ndarray
    nside_sky: int
    ground_names: list = field(default_factory=lambda: ["ground"])
    offset_names: list = field(default_factory=lambda: ["offset"])
    sky_observed: np.ndarray | None = None

    @property
    def npix(self):
        return healpy.nside2npix(self.nside_sky)

    @property
    def sky(self):
        return slice(0, self.npix)

    @property
    def ground(self):
        return slice(self.npix, self.npix + len(self.ground_names))

    @property
    def offset(self):
        start = self.npix + len(self.ground_names)
        return slice(start, start + len(self.offset_names))

    @property
    def column_names(self):
        return (
            [f"sky[{p}]" for p in range(self.npix)]
            + list(self.ground_names)
            + list(self.offset_names)
        )

    def pack(self, sky, ground, offset):
        """Assemble a parameter vector; each part broadcasts over frequency.

        ``sky`` is ``(npix,)`` or ``(nfreq, npix)``; ``ground`` and ``offset``
        are scalars, ``(n,)``, or ``(nfreq, n)``. Returns ``(nfreq, ncol)``.
        """
        nfreq = self.A.shape[0]
        parts = []
        for value, n in (
            (sky, self.npix),
            (ground, len(self.ground_names)),
            (offset, len(self.offset_names)),
        ):
            value = np.asarray(value, dtype=float)
            if value.ndim < 2:
                value = np.broadcast_to(value, (n,))
            parts.append(np.broadcast_to(value, (nfreq, n)))
        return np.concatenate(parts, axis=1)

    def predict(self, x):
        """Model temperatures ``(nfreq, ntime)`` for ``x`` of shape ``(nfreq, ncol)``."""
        return np.einsum("ftc,fc->ft", self.A, x)


def _unique_rows(arrays, decimals=9):
    """Index of the unique configuration of each row, and the unique rows."""
    key = np.concatenate(
        [np.round(np.asarray(a, dtype=float).reshape(len(a), -1), decimals)
         for a in arrays],
        axis=1,
    )
    uniq, first, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True
    )
    return inverse.ravel(), first


def build_design_matrix(
    beam,
    horizons,
    rot_gal2top,
    rot_body2enu,
    nside_sky,
    *,
    horizon_index=None,
    offset_groups=None,
    offset_names=None,
    ground_labels=None,
    ground_names=None,
    nside_int=64,
):
    """Build the sky / ground / offset design matrix for every frequency.

    Parameters
    ----------
    beam : HealpixBeam or compatible
        Assumed beam, body frame. Its frequencies set the output's.
    horizons : HorizonProfile or sequence of HorizonProfile
        Terrain horizon(s). Several are selected per row by ``horizon_index``
        (e.g. one per height era).
    rot_gal2top : ndarray, shape (ntime, 3, 3)
        Galactic -> topocentric ENU rotation per row
        (``EarthSurface.rot_gal2top_stack``).
    rot_body2enu : ndarray, shape (ntime, 3, 3) or (3, 3)
        Antenna body -> ENU rotation per row
        (``eigsep_base.rotations.mount_rotation``).
    nside_sky : int
        Resolution of the fitted Galactic sky.
    horizon_index : ndarray of int, shape (ntime,), optional
        Which of ``horizons`` applies to each row. Default: all use the first.
    offset_groups : ndarray of int, shape (ntime,), optional
        Offset group of each row (e.g. receiver regime); one offset column per
        group. Default: one offset shared by all rows. Pass ``False`` for no
        offset column.
    offset_names : list of str, optional
        Names of the offset columns, in group order.
    ground_labels : callable, optional
        ``ground_labels(dirs_enu) -> int array`` assigning each below-horizon
        direction to a ground region (e.g. by bearing sector). Default: one
        region.
    ground_names : list of str, optional
        Names of the ground regions, in label order.
    nside_int : int
        Resolution of the topocentric integration grid. Should be well above
        ``nside_sky`` and resolve the beam and horizon.

    Returns
    -------
    DesignMatrix
    """
    if isinstance(horizons, HorizonProfile):
        horizons = [horizons]
    rot_gal2top = np.asarray(rot_gal2top, dtype=float)
    ntime = rot_gal2top.shape[0]
    rot_body2enu = np.asarray(rot_body2enu, dtype=float)
    if rot_body2enu.shape == (3, 3):
        rot_body2enu = np.broadcast_to(rot_body2enu, (ntime, 3, 3))
    if rot_body2enu.shape != (ntime, 3, 3):
        raise ValueError(
            f"rot_body2enu must be (3, 3) or {(ntime, 3, 3)}, "
            f"got {rot_body2enu.shape}"
        )
    if horizon_index is None:
        horizon_index = np.zeros(ntime, dtype=int)
    horizon_index = np.asarray(horizon_index, dtype=int)

    # Topocentric integration grid and the static per-horizon quantities.
    nint = healpy.nside2npix(nside_int)
    dirs_enu = np.array(healpy.pix2vec(nside_int, np.arange(nint)))
    visible = [h.visible(dirs_enu) for h in horizons]
    if ground_labels is None:
        labels = np.zeros(nint, dtype=int)
    else:
        labels = np.asarray(ground_labels(dirs_enu), dtype=int)
    n_ground = int(labels.max()) + 1
    if ground_names is None:
        ground_names = (
            ["ground"] if n_ground == 1
            else [f"ground[{k}]" for k in range(n_ground)]
        )
    label_onehot = np.eye(n_ground)[labels]  # (nint, n_ground)

    if offset_groups is False:
        n_off, offset_groups = 0, None
        offset_names = []
    else:
        if offset_groups is None:
            offset_groups = np.zeros(ntime, dtype=int)
        offset_groups = np.asarray(offset_groups, dtype=int)
        n_off = int(offset_groups.max()) + 1
        if offset_names is None:
            offset_names = (
                ["offset"] if n_off == 1
                else [f"offset[{g}]" for g in range(n_off)]
            )

    npix = healpy.nside2npix(nside_sky)
    nfreq = len(beam.freqs_hz)
    ncol = npix + n_ground + n_off
    A = np.zeros((nfreq, ntime, ncol))

    # Beam x horizon once per distinct (pointing, horizon) configuration.
    config, first = _unique_rows([rot_body2enu, horizon_index[:, None]])
    for c, t0 in enumerate(first):
        B = beam(rot_body2enu[t0].T @ dirs_enu)  # (nfreq, nint)
        B = B / B.sum(axis=1, keepdims=True)
        vis = visible[horizon_index[t0]]
        w_sky = B * vis  # (nfreq, nint)
        w_gnd = (B * ~vis) @ label_onehot  # (nfreq, n_ground)
        rows = np.flatnonzero(config == c)
        A[:, rows, npix:npix + n_ground] = w_gnd[:, None, :]
        for t in rows:
            dirs_gal = rot_gal2top[t].T @ dirs_enu
            pix = healpy.vec2pix(nside_sky, *dirs_gal)
            S = scipy.sparse.csr_matrix(
                (np.ones(nint), (pix, np.arange(nint))), shape=(npix, nint)
            )
            A[:, t, :npix] = (S @ w_sky.T).T
    if n_off:
        A[:, np.arange(ntime), npix + n_ground + offset_groups] = 1.0

    observed = np.any(A[:, :, :npix] > 0, axis=(0, 1))
    return DesignMatrix(
        A=A,
        freqs_hz=np.asarray(beam.freqs_hz, dtype=float),
        nside_sky=nside_sky,
        ground_names=list(ground_names),
        offset_names=list(offset_names),
        sky_observed=observed,
    )


def _as_column_array(value, ncol, default):
    if value is None:
        return np.full(ncol, default, dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), (ncol,))


def solve(A, y, noise_sigma, prior_sigma=None, prior_mean=None, rcond=1e-12):
    """Gaussian-prior (MAP) least-squares solve at one frequency.

    Minimizes ``|(y - A x) / noise_sigma|^2 + |(x - prior_mean) / prior_sigma|^2``.
    ``prior_sigma = inf`` (the default) leaves a column unconstrained; exactly
    unconstrained directions are dropped by the eigenvalue cutoff ``rcond``
    (relative to the largest eigenvalue) and returned as zero.

    Parameters
    ----------
    A : ndarray, shape (ntime, ncol)
        One frequency of ``DesignMatrix.A``.
    y : ndarray, shape (ntime,)
    noise_sigma : float or ndarray, shape (ntime,)
    prior_sigma, prior_mean : float or ndarray, shape (ncol,), optional

    Returns
    -------
    dict
        ``x`` (ncol,), ``cov`` (ncol, ncol), ``eigvals`` and ``eigvecs`` of
        the posterior precision, and ``rank``.
    """
    A = np.asarray(A, dtype=float)
    ncol = A.shape[1]
    w = 1.0 / np.broadcast_to(np.asarray(noise_sigma, dtype=float), y.shape)
    prior_sigma = _as_column_array(prior_sigma, ncol, np.inf)
    prior_mean = _as_column_array(prior_mean, ncol, 0.0)
    Aw = A * w[:, None]
    precision = Aw.T @ Aw + np.diag(1.0 / prior_sigma**2)
    rhs = Aw.T @ (y * w) + prior_mean / prior_sigma**2
    lam, V = np.linalg.eigh(precision)
    keep = lam > rcond * lam[-1]
    inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    cov = (V * inv) @ V.T
    return {
        "x": cov @ rhs,
        "cov": cov,
        "eigvals": lam,
        "eigvecs": V,
        "rank": int(keep.sum()),
    }


def fisher_summary(dm, freq_index, noise_sigma, prior_sigma=None, rcond=1e-12):
    """What one frequency's data constrain, with the sky marginalized.

    Parameters
    ----------
    dm : DesignMatrix
    freq_index : int
    noise_sigma : float or ndarray, shape (ntime,)
    prior_sigma : float or ndarray, shape (ncol,), optional
        Prior width per column (inf = none). Use ``dm.pack`` to build one,
        e.g. a sky prior with free ground and offset.

    Returns
    -------
    dict
        ``names`` of the non-sky columns; their marginal ``sigma`` and
        ``corr`` matrix (``sigma`` is inf for a column the data and prior
        leave unconstrained); ``sigma_sky_mean``, the marginal error of the
        mean over observed sky pixels; ``null_vectors``, the precision eigenvectors
        below ``rcond`` restricted to the non-sky columns, plus
        ``null_sky_mean``, each null vector's mean over observed sky pixels.
    """
    A = dm.A[freq_index]
    ntime, ncol = A.shape
    if prior_sigma is not None:
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        if prior_sigma.ndim == 2:
            prior_sigma = prior_sigma[freq_index]
    sol = solve(A, np.zeros(ntime), noise_sigma, prior_sigma=prior_sigma,
                rcond=rcond)
    cov = sol["cov"]
    lam, V = sol["eigvals"], sol["eigvecs"]
    null = V[:, lam <= rcond * lam[-1]]
    nonsky = np.arange(dm.npix, ncol)
    sub = cov[np.ix_(nonsky, nonsky)]
    sigma = np.sqrt(np.clip(np.diag(sub), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sub / np.outer(sigma, sigma)
    # The pseudo-inverse treats null directions as known; any column that
    # moves along one is in fact unconstrained.
    tol = 1e-6
    sigma[np.any(np.abs(null[nonsky]) > tol, axis=1)] = np.inf

    e = np.zeros(ncol)
    obs = dm.sky_observed
    e[:dm.npix][obs] = 1.0 / max(int(obs.sum()), 1)
    sigma_sky_mean = float(np.sqrt(max(e @ cov @ e, 0.0)))
    if np.any(np.abs(e @ null) > tol / max(int(obs.sum()), 1)):
        sigma_sky_mean = np.inf
    return {
        "names": [dm.column_names[i] for i in nonsky],
        "sigma": sigma,
        "corr": corr,
        "sigma_sky_mean": sigma_sky_mean,
        "null_vectors": null[nonsky].T,
        "null_sky_mean": (e @ null),
        "rank": sol["rank"],
        "ncol": ncol,
    }
