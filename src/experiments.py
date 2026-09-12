"""A/B harness for systematic-bias studies on linear sky recovery.

Absorbed from ``bloom21cm/src/recovery_experiments.py`` and generalized.
The original imported ``build_design_matrix`` and ``simulate_observations``
directly from ``bloom_sim``, which hard-wired it to the BLOOM lunar-orbiter
column convention.  Those two functions are injected here instead, so any
experiment that can express itself as "simulate with one set of arrays,
recover with another" can use this module.

The central idea is that truth injection and the recovery design matrix are
built from *separate* inputs.  That is what makes a mismatch measurable: run
:func:`run_frequency_recovery` with ``truth_*`` arrays that contain a
systematic and ``recovery_*`` arrays that do not, and the returned
``sky_mean_K`` shift is the bias that systematic induces.  Add the
corresponding column to ``recovery_source_columns`` and re-run to check that
modeling it explicitly removes the bias.

Typical pairing: build the truth injection with the helpers in
:mod:`eigsep_sim.sources` (``earth_rfi_signal``, ``quiet_sun_signal``,
``surface_emission_signal``), and turn it into a recovery column with
:func:`source_column_from_signal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .recovery import normal_solve

__all__ = [
    "FrequencyRecoveryResult",
    "monopole_uncertainty_from_fit",
    "run_frequency_recovery",
    "source_column_from_signal",
]


@dataclass(frozen=True)
class FrequencyRecoveryResult:
    """Outputs from one per-frequency linear recovery experiment."""

    sky_mean_K: float
    sigma_mono_K: float
    fit: dict
    data: np.ndarray
    y: np.ndarray
    design_matrix: np.ndarray


def monopole_uncertainty_from_fit(fit, design_matrix, npix, sigma_noise):
    """Uncertainty of the observed-pixel sky mean from normal-solve output.

    Propagates the noise through the pseudo-inverse implied by the
    eigendecomposition in :func:`eigsep_sim.recovery.normal_solve`, for the
    specific linear functional "mean over observed sky pixels".

    Parameters
    ----------
    fit : dict
        Return value of :func:`eigsep_sim.recovery.normal_solve`.
    design_matrix : ndarray, shape (nrows, ncols)
    npix : int
    sigma_noise : float or array_like
        Per-observation noise sigma; the mean is used.

    Returns
    -------
    float
        ``NaN`` if no sky pixel was observed.
    """
    observed = ~fit["unobserved"]
    n_observed = int(observed.sum())
    if n_observed == 0:
        return np.nan
    e_sky = np.zeros(design_matrix.shape[1])
    e_sky[:npix][observed] = 1.0 / n_observed
    ve = fit["eigenvectors"].T @ e_sky
    return float(
        np.mean(sigma_noise)
        * np.sqrt(np.dot(ve**2, fit["inv_eigenvalues"]))
    )


def run_frequency_recovery(
    *,
    simulate_fn: Callable,
    design_fn: Callable,
    truth_masks,
    truth_beams,
    truth_omega_B,
    recovery_masks,
    recovery_beams,
    recovery_omega_B,
    sky_map,
    t_regolith,
    t_sun,
    sun_pixels,
    sigma_noise,
    rng=None,
    t_rx=None,
    include_t_rx=True,
    truth_additive_signal=None,
    recovery_source_columns: Optional[Sequence[np.ndarray]] = None,
    rcond=1e-6,
):
    """Simulate one frequency with truth arrays and recover with model arrays.

    This is the core A/B harness for systematic studies.  For example,
    inject an Earth RFI term through ``truth_additive_signal`` and either
    omit it from ``recovery_source_columns`` to measure the bias, or include
    its coupling column to test whether explicit modeling removes that bias.

    Parameters
    ----------
    simulate_fn : callable
        Forward model producing the observations.  Called as::

            simulate_fn(masks, beams, omega_B, sky_map, t_regolith, t_sun,
                        sun_pixels, sigma_noise, rng=..., t_rx=...,
                        additive_signal=...)

        and must return ``(data, y)``, where ``y`` is the flattened
        observation vector matching the design matrix's row ordering.
    design_fn : callable
        Design-matrix builder.  Called as::

            design_fn(masks, beams, omega_B, sun_pixels, npix,
                      include_t_rx=..., source_columns=...)

        and must return an array of shape ``(nrows, ncols)`` whose first
        ``npix`` columns are sky pixels.  See
        :func:`eigsep_sim.recovery.build_surface_design_matrix` for a
        builder with that column convention.
    truth_masks, truth_beams, truth_omega_B
        Arrays used to generate the data.
    recovery_masks, recovery_beams, recovery_omega_B
        Arrays used to build the recovery model.  Deliberately separate
        from the truth arrays.
    sky_map, t_regolith, t_sun, sun_pixels, sigma_noise
        Passed through to ``simulate_fn``.
    rng : numpy.random.Generator, optional
    t_rx : optional
        Per-dipole receiver temperature offsets injected into the truth.
    include_t_rx : bool
        Whether the recovery model solves for receiver offsets.
    truth_additive_signal : ndarray, optional
        Extra systematic added to the truth only, shape
        ``(nobs, ndipole, nfreq)``.
    recovery_source_columns : sequence of ndarray, optional
        Extra design columns, each shape ``(nobs, ndipole)``.
    rcond : float

    Returns
    -------
    FrequencyRecoveryResult
    """
    truth_masks = np.asarray(truth_masks, dtype=float)
    npix = truth_masks.shape[1]
    sigma_noise = np.asarray(sigma_noise, dtype=float)

    data, y = simulate_fn(
        truth_masks,
        truth_beams,
        truth_omega_B,
        sky_map,
        t_regolith,
        t_sun,
        sun_pixels,
        sigma_noise,
        rng=rng,
        t_rx=t_rx,
        additive_signal=truth_additive_signal,
    )
    design_matrix = design_fn(
        recovery_masks,
        recovery_beams,
        recovery_omega_B,
        sun_pixels,
        npix,
        include_t_rx=include_t_rx,
        source_columns=recovery_source_columns,
    )
    fit = normal_solve(design_matrix, y, npix, rcond=rcond)
    sigma_mono = monopole_uncertainty_from_fit(
        fit, design_matrix, npix, sigma_noise
    )
    return FrequencyRecoveryResult(
        sky_mean_K=float(np.nanmean(fit["sky_map"])),
        sigma_mono_K=sigma_mono,
        fit=fit,
        data=data,
        y=y,
        design_matrix=design_matrix,
    )


def source_column_from_signal(signal_tdf, source_temperature_K):
    """Infer a source design column from an additive signal and spectrum.

    ``signal_tdf`` is ``(nobs, ndipole, nfreq)``.  For a single frequency or
    scalar source temperature, this returns ``(nobs, ndipole)``.  Use this
    for modeled-source tests where the truth injection was generated by the
    helpers in :mod:`eigsep_sim.sources`.
    """
    signal = np.asarray(signal_tdf, dtype=float)
    temp = np.asarray(source_temperature_K, dtype=float)
    if temp.ndim == 0:
        denom = float(temp)
    elif temp.ndim == 1 and temp.size == 1:
        denom = float(temp[0])
    else:
        raise ValueError("source_temperature_K must be scalar or length 1")
    if denom == 0.0:
        raise ValueError("source_temperature_K must be non-zero")
    if signal.ndim == 3:
        if signal.shape[2] != 1:
            raise ValueError("signal_tdf must have one frequency channel")
        signal = signal[:, :, 0]
    if signal.ndim != 2:
        raise ValueError("signal_tdf must have shape (nobs, ndipole[, 1])")
    return signal / denom
