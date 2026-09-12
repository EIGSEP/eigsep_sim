"""Shared beam sampling and linear recovery matrix construction."""

from __future__ import annotations

from collections.abc import Mapping

import healjax
import numpy as np


def sample_beam_weights(fwd, geom, freq_index, beam_coeffs=None):
    """Sample body-frame beams onto Galactic sky pixels.

    Parameters
    ----------
    fwd : ForwardModel
        Forward model providing sky and beam descriptors.
    geom : dict
        Geometry returned by ``ForwardModel.precompute_geometry``.
    freq_index : int
        Frequency channel to sample.
    beam_coeffs : ndarray, optional
        Beam coefficients. Defaults to ``fwd.beam.coeffs``.

    Returns
    -------
    ndarray
        Beam weights with shape ``(ntime, ndipole, npix)``.
    """
    if beam_coeffs is None:
        beam_coeffs = fwd.beam.coeffs
    sky_dirs = np.asarray(geom["crds_gal_jax"])
    rots = np.asarray(geom["rots_jax"])
    body_rots = np.asarray(geom["body_rots_jax"])
    body_dirs = np.einsum("tij,tjk,kn->tin", body_rots, rots, sky_dirs)
    beam_maps = fwd.beam.basis.deproject(beam_coeffs)[:, :, freq_index]
    weights = np.empty(
        (len(rots), beam_maps.shape[0], sky_dirs.shape[1]), dtype=float
    )
    for time_index, directions in enumerate(body_dirs):
        theta, phi = healjax.vec2ang(
            directions[0], directions[1], directions[2]
        )
        pixels, interp_weights = healjax.get_interp_weights(
            theta, phi, fwd.beam.nside
        )
        for dipole_index, beam_map in enumerate(beam_maps):
            sampled = sum(
                beam_map[np.asarray(pixels[k])] * np.asarray(interp_weights[k])
                for k in range(4)
            )
            weights[time_index, dipole_index] = sampled
    return weights


def build_surface_design_matrix(
    weights,
    masks,
    unresolved_surface_weight=None,
    source_columns=None,
    include_receiver_offsets=False,
):
    """Build a generic sky, blocked-surface, and optional-source matrix.

    Column ordering is ``[sky pixels | surface | sources | receiver offsets]``.
    Source columns preserve the insertion order of ``source_columns``.

    Parameters
    ----------
    weights : ndarray, shape (nobs, ndipole, npix)
        Beam weights for sky pixels.
    masks : ndarray, shape (nobs, npix)
        Visibility factors, where one means visible sky.
    unresolved_surface_weight : ndarray, shape (nobs, ndipole), optional
        Additional beam weight assigned to unresolved blocked-surface emission.
    source_columns : mapping or sequence of ndarray, optional
        Extra columns with shape ``(nobs, ndipole)``.
    include_receiver_offsets : bool
        Append one offset column per dipole.
    """
    weights = np.asarray(weights, dtype=float)
    masks = np.asarray(masks, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (nobs, ndipole, npix)")
    nobs, ndipole, npix = weights.shape
    if masks.shape != (nobs, npix):
        raise ValueError(
            f"masks must have shape {(nobs, npix)}, got {masks.shape}"
        )
    if unresolved_surface_weight is None:
        unresolved_surface_weight = np.zeros((nobs, ndipole), dtype=float)
    unresolved_surface_weight = np.asarray(
        unresolved_surface_weight, dtype=float
    )
    if unresolved_surface_weight.shape != (nobs, ndipole):
        raise ValueError(
            "unresolved_surface_weight must have shape "
            f"{(nobs, ndipole)}, got {unresolved_surface_weight.shape}"
        )
    if source_columns is None:
        source_columns = []
    elif isinstance(source_columns, Mapping):
        source_columns = list(source_columns.values())
    else:
        source_columns = list(source_columns)
    source_columns = [
        np.asarray(column, dtype=float) for column in source_columns
    ]
    for column in source_columns:
        if column.shape != (nobs, ndipole):
            raise ValueError(
                f"source columns must have shape {(nobs, ndipole)}, "
                f"got {column.shape}"
            )

    ncols = npix + 1 + len(source_columns)
    if include_receiver_offsets:
        ncols += ndipole
    matrix = np.zeros((nobs, ndipole, ncols), dtype=float)
    matrix[:, :, :npix] = weights * masks[:, None, :]
    matrix[:, :, npix] = (
        np.sum(weights * (1.0 - masks[:, None, :]), axis=2)
        + unresolved_surface_weight
    )
    for source_index, column in enumerate(source_columns):
        matrix[:, :, npix + 1 + source_index] = column
    if include_receiver_offsets:
        offset_start = npix + 1 + len(source_columns)
        for dipole_index in range(ndipole):
            matrix[:, dipole_index, offset_start + dipole_index] = 1.0
    return matrix.reshape(nobs * ndipole, ncols)


def normal_solve(A, y, npix, rcond=1e-6):
    """Least-squares sky recovery via normal equations.

    ``A`` must use the current recovery column convention where the first
    ``npix`` columns are sky pixels, followed by blocked-surface/source columns
    and optional receiver offsets. Unobserved sky pixels are returned as NaN.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    AtA = A.T @ A
    Aty = A.T @ y
    lam, V = np.linalg.eigh(AtA)
    lam_thresh = rcond**2 * lam[-1]
    inv_lam = np.where(
        lam > lam_thresh,
        1.0 / np.where(lam > lam_thresh, lam, 1.0),
        0.0,
    )
    x_est = V @ (inv_lam * (V.T @ Aty))

    sky_map = x_est[:npix].copy()
    col_norms_sq = np.diag(AtA)[:npix]
    unobserved = col_norms_sq < (1e-6**2) * col_norms_sq.max()
    sky_map[unobserved] = np.nan

    result = {
        "sky_map": sky_map,
        "surface": float(x_est[npix]),
        "t_regolith": float(x_est[npix]),
        "eigenvalues": lam,
        "eigenvectors": V,
        "inv_eigenvalues": inv_lam,
        "rank": int((lam > lam_thresh).sum()),
        "unobserved": unobserved,
    }
    if A.shape[1] > npix + 1:
        result["t_sun"] = float(x_est[npix + 1])
    if A.shape[1] == npix + 4:
        result["t_rx_0"] = float(x_est[npix + 2])
        result["t_rx_1"] = float(x_est[npix + 3])
    for index in range(npix + 1, A.shape[1]):
        result[f"extra_{index - npix - 1}"] = float(x_est[index])
    return result


class ScaleDegeneracy:
    """Multiplicative gauge coupled across parameter arrays.

    ``responses`` maps parameter names to their power-law response to one
    scale gauge. For example ``{"sky": 1, "beam": -1}`` applies
    ``sky *= scale`` and ``beam /= scale``. ``group_axes`` retains independent
    gauges, such as one scale per frequency channel. A sequence of keys is
    accepted as shorthand for unit responses.

    With no reference solution, the scale sets the joint geometric-mean gauge
    to unity. With a reference, a log-amplitude least-squares fit replaces the
    degenerate subspace with the reference gauge while preserving products
    between parameters with opposite responses.
    """

    def __init__(self, responses, group_axes=()):
        if hasattr(responses, "items"):
            self.responses = dict(responses)
        else:
            self.responses = {key: 1.0 for key in responses}
        if not self.responses or any(
            response == 0 for response in self.responses.values()
        ):
            raise ValueError("scale responses must be non-zero")
        self.group_axes = tuple(group_axes)

    def project(self, params, reference=None):
        numerator = None
        denominator = None
        for key, response in self.responses.items():
            array = np.asarray(params[key])
            axes = _reduction_axes(array, self.group_axes)
            if reference is None:
                valid = np.abs(array) > 0
                log_ratio = -np.log(
                    np.abs(array),
                    where=valid,
                    out=np.zeros_like(array, dtype=float),
                )
            else:
                target = np.asarray(reference[key])
                if target.shape != array.shape:
                    raise ValueError(
                        f"parameter {key!r} has shape {array.shape}, expected {target.shape}"
                    )
                valid = (array * target) > 0
                ratio = np.divide(
                    np.abs(target),
                    np.abs(array),
                    out=np.ones_like(array, dtype=float),
                    where=valid,
                )
                log_ratio = np.log(ratio)
            term = response * np.sum(
                np.where(valid, log_ratio, 0.0), axis=axes
            )
            weight = response**2 * np.sum(valid, axis=axes)
            numerator = term if numerator is None else numerator + term
            denominator = (
                weight if denominator is None else denominator + weight
            )
        if np.any(denominator <= 0):
            raise ValueError(
                "cannot project scale gauge without non-zero matching values"
            )
        scale = np.exp(numerator / denominator)
        for key, response in self.responses.items():
            array = np.asarray(params[key])
            params[key] = array * _expand_group_value(
                scale**response, array.ndim, self.group_axes
            )


class AdditiveDegeneracy:
    """Additive gauge coupled across parameter arrays.

    ``coefficients`` maps parameter names to their response to one additive
    gauge value. For example ``{"sky": 1, "ground": 1, "receiver": -1}``
    preserves a radiometer model when a common temperature offset moves from
    receiver temperature into sky and ground emission. ``group_axes`` retains
    independent gauges, such as one offset per frequency channel.
    """

    def __init__(self, coefficients, group_axes=()):
        self.coefficients = dict(coefficients)
        self.group_axes = tuple(group_axes)

    def project(self, params, reference=None):
        numerator = None
        denominator = None
        for key, coefficient in self.coefficients.items():
            array = np.asarray(params[key])
            axes = _reduction_axes(array, self.group_axes)
            target = 0.0 if reference is None else np.asarray(reference[key])
            term = coefficient * np.sum(target - array, axis=axes)
            weight = coefficient**2 * np.prod(
                [array.shape[axis] for axis in axes]
            )
            numerator = term if numerator is None else numerator + term
            denominator = (
                weight if denominator is None else denominator + weight
            )
        offset = numerator / denominator
        for key, coefficient in self.coefficients.items():
            array = np.asarray(params[key])
            params[key] = array + coefficient * _expand_group_value(
                offset, array.ndim, self.group_axes
            )


class RecoverySolution:
    """Parameter dictionary with redcal-style degeneracy projection.

    ``remove_degen()`` changes only explicitly registered degenerate subspaces.
    A supplied ``degen_sol`` replaces those gauges with the reference gauges;
    otherwise each degeneracy chooses its canonical zero or unit gauge.
    """

    def __init__(self, params, degeneracies=()):
        self.params = {
            key: np.array(value, copy=True) for key, value in params.items()
        }
        self.degeneracies = tuple(degeneracies)

    def remove_degen(self, degen_sol=None, inplace=True):
        """Remove degeneracies or replace them with gauges from ``degen_sol``."""
        if degen_sol is not None:
            reference = (
                degen_sol.params
                if isinstance(degen_sol, RecoverySolution)
                else degen_sol
            )
        else:
            reference = None
        solution = (
            self
            if inplace
            else RecoverySolution(self.params, self.degeneracies)
        )
        for degeneracy in solution.degeneracies:
            degeneracy.project(solution.params, reference=reference)
        if not inplace:
            return solution


def _normalize_group_axes(ndim, group_axes):
    axes = tuple(axis + ndim if axis < 0 else axis for axis in group_axes)
    if len(set(axes)) != len(axes) or any(
        axis < 0 or axis >= ndim for axis in axes
    ):
        raise ValueError(f"group_axes {group_axes} invalid for {ndim}-D array")
    return tuple(sorted(axes))


def _reduction_axes(array, group_axes):
    group_axes = _normalize_group_axes(array.ndim, group_axes)
    return tuple(axis for axis in range(array.ndim) if axis not in group_axes)


def _expand_group_value(value, ndim, group_axes):
    group_axes = _normalize_group_axes(ndim, group_axes)
    shape = [1] * ndim
    for size, axis in zip(np.shape(value), group_axes):
        shape[axis] = size
    return np.reshape(value, shape)


def relative_rms(estimate, reference):
    """Return RMS error relative to the RMS amplitude of a reference array."""
    estimate = np.asarray(estimate)
    reference = np.asarray(reference)
    return float(
        np.sqrt(np.mean((estimate - reference) ** 2))
        / np.sqrt(np.mean(reference**2))
    )


# ---------------------------------------------------------------------------
# Chunked normal equations (absorbed from bloom21cm/src/linear_solver.py)
#
# :func:`normal_solve` above forms ``A.T @ A`` from a design matrix that is
# already in memory.  For long campaigns that matrix is
# ``(n_total * ndipole, npix + 2)`` and does not fit; the routines below
# build it a chunk of observations at a time and accumulate the normal
# equations instead, then hand off to :func:`normal_solve_equations`.
#
# Column ordering of A: [sky pixels (npix) | regolith (1) | sun (1)]
# ---------------------------------------------------------------------------


def _build_A_chunk(masks, beams, omega_B, J_SUN, npix, k_start, k_end, n_obs):
    """Build one chunk of the design matrix (rows ``k_start..k_end-1``)."""
    chunk_size = k_end - k_start
    ndipole = beams.shape[1]
    t_idx = np.arange(chunk_size)

    weights = beams[k_start:k_end] / omega_B[k_start:k_end, :, np.newaxis]
    m = masks[k_start:k_end]

    j_sun = J_SUN[(k_start + t_idx) % n_obs]  # (chunk_size,)

    A_sky = (weights * m[:, np.newaxis, :]).reshape(
        chunk_size * ndipole, npix
    )
    A_reg = np.sum(
        weights * (1.0 - m[:, np.newaxis, :]), axis=2
    ).reshape(chunk_size * ndipole)

    sun_weights = weights[
        t_idx[:, np.newaxis],
        np.arange(ndipole)[np.newaxis, :],
        j_sun[:, np.newaxis],
    ]  # (chunk_size, ndipole)
    A_sun = (sun_weights * m[t_idx, j_sun][:, np.newaxis]).reshape(
        chunk_size * ndipole
    )

    A_chunk = np.empty((chunk_size * ndipole, npix + 2), dtype=float)
    A_chunk[:, :npix] = A_sky
    A_chunk[:, npix] = A_reg
    A_chunk[:, npix + 1] = A_sun
    return A_chunk


def build_normal_equations(
    masks, beams, omega_B, J_SUN, npix, x_true=None, chunk_obs=2000
):
    """Accumulate ``A.T @ A`` and optionally ``y = A @ x_true`` in chunks.

    Parameters
    ----------
    masks : ndarray, shape (n_total, npix)
    beams : ndarray, shape (n_total, ndipole, npix)
    omega_B : ndarray, shape (n_total, ndipole)
    J_SUN : ndarray, shape (n_obs,)
        Sun pixel indices; ``n_total = n_orbits * n_obs``.
    npix : int
    x_true : ndarray, shape (npix + 2,), optional
        If given, ``y = A @ x_true`` is also returned.
    chunk_obs : int
        Observations per chunk.

    Returns
    -------
    AtA : ndarray, shape (npix + 2, npix + 2)
    col_sums : ndarray, shape (npix + 2,)
        ``A.T @ 1``.
    y_nl : ndarray, shape (n_total * ndipole,) or None
    """
    n_total, ndipole, _ = beams.shape
    n_obs = len(J_SUN)
    ncols = npix + 2
    AtA = np.zeros((ncols, ncols))
    col_sums = np.zeros(ncols)
    y_nl = np.zeros(n_total * ndipole) if x_true is not None else None

    for k_start in range(0, n_total, chunk_obs):
        k_end = min(k_start + chunk_obs, n_total)
        A_chunk = _build_A_chunk(
            masks, beams, omega_B, J_SUN, npix, k_start, k_end, n_obs
        )
        AtA += A_chunk.T @ A_chunk
        col_sums += A_chunk.sum(axis=0)
        if x_true is not None:
            r0, r1 = k_start * ndipole, k_end * ndipole
            y_nl[r0:r1] = A_chunk @ x_true

    return AtA, col_sums, y_nl


def build_A_right_product(
    masks, beams, omega_B, J_SUN, npix, V_full, x_vec=None, chunk_obs=2000
):
    """Compute ``A @ V_full`` in float32 chunks, optionally also ``A @ x_vec``.

    Parameters
    ----------
    masks, beams, omega_B, J_SUN, npix
        As in :func:`build_normal_equations`.
    V_full : ndarray, shape (npix + 2, K)
        Matrix to multiply.
    x_vec : ndarray, shape (npix + 2,), optional
        If given, also computes ``A.T @ (A @ x_vec)`` and ``col_sums``.
    chunk_obs : int

    Returns
    -------
    AV_f32 : ndarray, shape (n_total * ndipole, K), float32
    y_out : ndarray, shape (n_total * ndipole,) or None
    _ : None
        Placeholder for API compatibility.
    At_Ax : ndarray, shape (npix + 2,) or None
    col_sums : ndarray, shape (npix + 2,) or None
    """
    n_total, ndipole, _ = beams.shape
    n_obs = len(J_SUN)
    n_rows = n_total * ndipole
    ncols, n_V = V_full.shape

    AV_f32 = np.zeros((n_rows, n_V), dtype=np.float32)
    y_out = np.zeros(n_rows) if x_vec is not None else None
    At_Ax = np.zeros(ncols) if x_vec is not None else None
    col_sums = np.zeros(ncols) if x_vec is not None else None

    for k_start in range(0, n_total, chunk_obs):
        k_end = min(k_start + chunk_obs, n_total)
        A_chunk = _build_A_chunk(
            masks, beams, omega_B, J_SUN, npix, k_start, k_end, n_obs
        )
        r0, r1 = k_start * ndipole, k_end * ndipole
        AV_f32[r0:r1] = (A_chunk @ V_full).astype(np.float32)
        if x_vec is not None:
            Ax_chunk = A_chunk @ x_vec
            y_out[r0:r1] = Ax_chunk
            At_Ax += A_chunk.T @ Ax_chunk
            col_sums += A_chunk.sum(axis=0)

    return AV_f32, y_out, None, At_Ax, col_sums


def normal_solve_equations(AtA, Aty, npix, rcond=1e-6):
    """Solve ``AtA @ x = Aty`` via eigendecomposition.

    Pre-accumulated counterpart to :func:`normal_solve`, which takes the
    design matrix itself.  Use this together with
    :func:`build_normal_equations` when ``A`` is too large to materialise.

    Parameters
    ----------
    AtA : ndarray, shape (ncols, ncols)
    Aty : ndarray, shape (ncols,)
    npix : int
        The first ``npix`` columns are sky pixels.
    rcond : float

    Returns
    -------
    dict
        Keys ``sky_map``, ``eigenvectors``, ``inv_eigenvalues``,
        ``unobserved``.
    """
    lam, V = np.linalg.eigh(AtA)
    lam_thresh = rcond**2 * lam[-1]
    safe = lam > lam_thresh
    inv_lam = np.where(safe, 1.0 / np.where(safe, lam, 1.0), 0.0)
    x_est = V @ (inv_lam * (V.T @ Aty))

    sky_map = x_est[:npix].copy()
    col_norms_sq = np.diag(AtA)[:npix]
    unobserved = col_norms_sq < (1e-6**2) * col_norms_sq.max()
    sky_map[unobserved] = np.nan

    return {
        "sky_map": sky_map,
        "eigenvectors": V,
        "inv_eigenvalues": inv_lam,
        "unobserved": unobserved,
    }
