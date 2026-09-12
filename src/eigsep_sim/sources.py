"""Brightness-temperature models for Phase-1 external sources (Sun, Earth).

Companion to :mod:`ephemeris` (real Sun/Earth geometry) and the
``ext_source_dirs_gal``/``ext_source_temps`` hooks on
:class:`eigsep_sim.forward_model.ForwardModel` (beam-weighted, occultation-gated
point-source injection).  This module supplies the *brightness* half:
what temperature to inject once a source's direction and visibility are
known.

Quiet-Sun model
----------------
The quiet-Sun corona is optically thick to free-free absorption at
decametric wavelengths, so its brightness temperature is set by the
coronal electron temperature at the layer where the plasma frequency
equals the observing frequency -- it *rises* toward lower frequency
(deeper corona -> more tenuous, hotter plasma) over the ~20-150 MHz range.
``quiet_sun_temperature_K`` is a coarse log-log interpolation anchored on
two published measurements:

- ~6e5 K at 150 MHz (LOFAR quiet-Sun imaging, Zhang et al. 2022, ApJ 932, 17;
  Mercier & Chambe, "LOFAR observations of the quiet solar corona",
  A&A 2018)
- ~1.5e6 K at 50 MHz (same works report brightness temperatures rising
  toward ~1-2e6 K over 20-80 MHz; 1.5e6 K is a midpoint placeholder, not a
  literature value at exactly 50 MHz)

This is a **placeholder fit**, not a validated spectrum -- good enough to
gate "does an unmodelled quiet Sun bias T21 recovery," not to claim
photometric accuracy.  Replace with a digitised published spectrum (or a
proper free-free coronal model) before using this for anything more than
that gating test.

Variability
-----------
Two separate, explicitly-labelled-synthetic pieces, per the plan's
priority ordering (deterministic quiet-Sun spectrum first, variability
second, bursts treated as a flagging problem rather than fit through):

- ``solar_activity_envelope`` / ``sun_temperature_K``: a smooth
  multiplicative modulation combining an ~11 yr solar-cycle envelope and
  an ~27 day active-region rotational modulation.  This is **not** fit to
  real space-weather data (real F10.7 solar flux varies by a factor of
  ~4-5 between minimum and maximum with substantial day-to-day scatter
  this sinusoidal model does not capture) -- it exists to wire the
  *shape* of two-timescale variability through the pipeline correctly;
  swap in a real F10.7 time series when photometric accuracy matters.
- ``inject_solar_bursts``: a stochastic Poisson-process realization of
  Type-III-like bursts, amplitude power-law scaled with frequency
  (Saint-Hilaire et al. 2013 report Type III source-flux normalisation
  scaling as ``f**-2.9`` over 150-450 MHz; extrapolated here to 50-150 MHz,
  not re-derived for this band).  Individual Type III bursts last
  ~0.2-0.6 s, grouped bursts 1-5 min, storms minutes-hours (Reid & Ratcliffe,
  "A review of solar type III radio bursts"); ~90% occur without an
  associated flare/CME, i.e. they're common even in nominally quiet
  periods.  There is no validated occurrence-rate calibration here --
  ``rate_per_hour`` is a free knob for stress-testing a flagger, not a
  measured duty cycle.  Per the plan, bursts are meant to be **flagged and
  excised**, not fit through -- see ``flag_bursts``.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
import astropy.units as u

from eigsep_base.const import c, k_B, eta_0, pi

_ANCHOR_FREQS_HZ = np.array([50e6, 150e6])
_ANCHOR_TEMPS_K = np.array([1.5e6, 6e5])


def quiet_sun_temperature_K(freqs_hz):
    """Coarse quiet-Sun brightness temperature spectrum [K] over ~20-150 MHz.

    Log-log power-law through the two anchor points documented in the
    module docstring.  See the caveats there before trusting the absolute
    scale for anything beyond a rough sensitivity/flagging test.

    Parameters
    ----------
    freqs_hz : array_like

    Returns
    -------
    T_K : ndarray, same shape as ``freqs_hz``
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    log_f = np.log(_ANCHOR_FREQS_HZ)
    log_T = np.log(_ANCHOR_TEMPS_K)
    slope = (log_T[1] - log_T[0]) / (log_f[1] - log_f[0])
    log_T_f = log_T[0] + slope * (np.log(f) - log_f[0])
    return np.exp(log_T_f)


# ── slow solar variability (rotation + cycle envelope) ──────────────────────

def solar_activity_envelope(
    times,
    cycle_period_days=11 * 365.25,
    cycle_phase0=0.0,
    cycle_min=0.5,
    cycle_max=1.5,
    rotation_period_days=27.0,
    rotation_amplitude=0.15,
    rotation_phase0=0.0,
):
    """Dimensionless multiplicative solar-activity envelope (baseline ~1).

    SYNTHETIC -- see the module docstring's "Variability" section for what
    this does and does not capture.  Combines a sinusoidal ~11 yr
    solar-cycle envelope (oscillating between ``cycle_min`` and
    ``cycle_max``) with a sinusoidal ~27 day rotational modulation on top.
    Phased off days since J2000 TDB (matches the convention used in
    ``observer._moon_icrs2mcmf``) rather than the campaign start, so the
    envelope is reproducible independent of where a given campaign's time
    array begins.

    Parameters
    ----------
    times : Time-like, shape (ntimes,)

    Returns
    -------
    envelope : ndarray, shape (ntimes,)
    """
    d = Time(times).tdb.jd - 2451545.0
    cycle = 0.5 * (cycle_max + cycle_min) + 0.5 * (cycle_max - cycle_min) * np.sin(
        2 * np.pi * d / cycle_period_days + cycle_phase0
    )
    rotation = 1.0 + rotation_amplitude * np.sin(
        2 * np.pi * d / rotation_period_days + rotation_phase0
    )
    return cycle * rotation


def sun_temperature_K(freqs_hz, times, activity_kwargs=None):
    """Quiet-Sun spectrum modulated by the synthetic activity envelope.

    Assumes the spectral *shape* is fixed and only the overall amplitude
    varies with activity level (a simplification -- real solar-cycle
    variation also changes spectral shape, e.g. active-region emission is
    more structured than the quiet corona; not modelled here).

    Parameters
    ----------
    freqs_hz : array_like, shape (nfreq,)
    times : Time-like, shape (ntimes,)
    activity_kwargs : dict, optional
        Passed through to ``solar_activity_envelope``.

    Returns
    -------
    T_K : ndarray, shape (ntimes, nfreq)
    """
    T_ref = quiet_sun_temperature_K(freqs_hz)  # (nfreq,)
    envelope = solar_activity_envelope(times, **(activity_kwargs or {}))  # (ntimes,)
    return envelope[:, None] * T_ref[None, :]


# ── solar bursts: stochastic injection + flag/excision ──────────────────────

def inject_solar_bursts(
    times,
    freqs_hz,
    rng,
    rate_per_hour=2.0,
    ref_freq_hz=100e6,
    ref_amplitude_K=(1e6, 5e7),
    spectral_index=-2.9,
    duration_s=(0.5, 120.0),
):
    """Stochastic Type-III-like burst realization over ``times``.

    SYNTHETIC -- see the module docstring.  Burst start times are a Poisson
    process at ``rate_per_hour``; each burst's peak amplitude at
    ``ref_freq_hz`` is drawn log-uniformly from ``ref_amplitude_K``, scaled
    to other frequencies by ``(f / ref_freq_hz) ** spectral_index``,  with a
    fast linear rise (10% of duration) and exponential decay, duration
    drawn log-uniformly from ``duration_s``.

    Parameters
    ----------
    times : Time-like, shape (ntimes,)
    freqs_hz : array_like, shape (nfreq,)
    rng : numpy.random.Generator
    rate_per_hour : float
        Poisson rate of burst onsets.
    ref_freq_hz, ref_amplitude_K, spectral_index, duration_s : see above.

    Returns
    -------
    burst_temp_K : ndarray, shape (ntimes, nfreq)
    is_contaminated : ndarray of bool, shape (ntimes,)
        Ground-truth contamination flag (burst power above 1% of the
        smallest ``ref_amplitude_K`` at any frequency), for validating a
        flagger against.
    """
    times = Time(times)
    t_s = (times - times[0]).to(u.s).value
    ntimes = len(t_s)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    duration_campaign_hr = (t_s[-1] - t_s[0]) / 3600.0 if ntimes > 1 else 0.0

    n_bursts = rng.poisson(rate_per_hour * duration_campaign_hr)
    burst_temp = np.zeros((ntimes, len(freqs_hz)))
    freq_shape = (freqs_hz / ref_freq_hz) ** spectral_index

    for _ in range(n_bursts):
        t0 = rng.uniform(t_s[0], t_s[-1])
        dur = np.exp(rng.uniform(np.log(duration_s[0]), np.log(duration_s[1])))
        amp = np.exp(
            rng.uniform(np.log(ref_amplitude_K[0]), np.log(ref_amplitude_K[1]))
        )
        dt = t_s - t0
        rise = 0.1 * dur
        # np.where evaluates both branches everywhere before selecting, so for
        # dt << 0 (elements far before this burst) the discarded decay branch's
        # exponent can be huge and overflow exp() even though its value is
        # never used -- clip it to avoid the spurious RuntimeWarning.
        decay_exponent = np.clip(-(dt - rise) / dur, None, 700.0)
        profile = np.where(
            dt < 0, 0.0, np.where(dt < rise, dt / rise, np.exp(decay_exponent))
        )
        burst_temp += amp * profile[:, None] * freq_shape[None, :]

    threshold_K = 0.01 * ref_amplitude_K[0]
    is_contaminated = np.any(burst_temp > threshold_K, axis=1)
    return burst_temp, is_contaminated


def flag_bursts(data, threshold_sigma=6.0, axis=0):
    """Robust median/MAD outlier flag along ``axis`` (default: time).

    Computes a per-channel (all axes other than ``axis``) robust z-score
    ``|x - median| / (1.4826 * MAD)`` and flags a sample along ``axis`` if
    ANY channel exceeds ``threshold_sigma`` there.  A simple stand-in for a
    proper RFI flagger -- sufficient to test whether excising flagged
    samples protects a matched-filter/fit from burst contamination; not
    tuned for a real instrument's noise statistics.

    Parameters
    ----------
    data : ndarray, shape (n_along_axis, ...)
    threshold_sigma : float
    axis : int

    Returns
    -------
    flagged : ndarray of bool, shape (n_along_axis,)
    """
    data = np.moveaxis(np.asarray(data), axis, 0)
    med = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - med), axis=0, keepdims=True)
    mad = np.where(mad == 0, 1e-12, mad)
    z = np.abs(data - med) / (1.4826 * mad)
    other_axes = tuple(range(1, data.ndim))
    return np.any(z > threshold_sigma, axis=other_axes)


# ── Earth RFI (coarse, FM-band-dominated placeholder) ────────────────────────

FM_BAND_HZ = (88e6, 108e6)


def earth_rfi_temperature_K(freqs_hz, in_band_temp_K=5e5, out_of_band_temp_K=0.0):
    """Coarse Earth RFI brightness spectrum [K]: elevated across the FM
    broadcast band (88-108 MHz), ~zero (in this simplified model) outside it.

    SYNTHETIC amplitude, real frequency structure.  A farside radio-quiet-
    zone characterization (Bassett et al. 2020, arXiv:2003.03468) reports
    an equivalent RFI brightness temperature at the Moon's distance of
    ~7.5e5 K around 17 MHz, with the lunar bulk providing up to ~90 dB of
    shielding on the true farside once ~6 deg past the limb -- i.e. real
    Earth RFI is either negligible (deep farside) or overwhelming (any
    direct or grazing view of Earth), not a smooth in-between.  This
    function models only the FREQUENCY structure (FM band bright, rest of
    the science band dark); ``in_band_temp_K`` is a free amplitude
    parameter for exactly that reason.  Pair with
    :func:`eigsep_sim.ephemeris.body_occulted_by_moon` for on/off gating --
    but note that function is a purely geometric (line-of-sight) occultation
    test and does NOT include the ~6 deg extra limb-grazing shielding
    margin reported above; treat orbits that skim the limb as an
    unmodelled risk, not as verified-safe.

    Parameters
    ----------
    freqs_hz : array_like
    in_band_temp_K : float
    out_of_band_temp_K : float

    Returns
    -------
    T_K : ndarray, same shape as ``freqs_hz``
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    in_band = (f >= FM_BAND_HZ[0]) & (f <= FM_BAND_HZ[1])
    return np.where(in_band, in_band_temp_K, out_of_band_temp_K)


def earth_rfi_tone_temperature_K(
    freqs_hz, tone_freqs_hz, tone_temp_K=5e5, tone_width_hz=None
):
    """Earth RFI as a comb of discrete broadcast-carrier tones, rather than
    the flat in-band continuum of :func:`earth_rfi_temperature_K`.

    Real FM/broadcast RFI is many narrow carriers (each order ~100-200 kHz
    wide) separated by mostly-quiet channels, not a smooth top-hat spanning
    the whole band -- a flat top-hat is a much smoother function of
    frequency than the real environment, which makes it more degenerate
    with smooth sky/Sun continua in a joint spectral fit than the real
    signal would be. This models only the frequency structure (nonzero at
    the tone centers, zero elsewhere); ``tone_temp_K`` is a free amplitude
    parameter, same convention as ``in_band_temp_K`` above.

    Parameters
    ----------
    freqs_hz : array_like, shape (nfreq,)
        Frequency grid to evaluate on.
    tone_freqs_hz : array_like, shape (ntones,)
        Center frequencies of the discrete tones [Hz].
    tone_temp_K : float
        Brightness temperature contributed by each tone.
    tone_width_hz : float, optional
        Half-width used to associate a grid frequency with a tone. Default
        is half the median grid spacing, i.e. each tone lights up only its
        single nearest frequency channel -- appropriate when ``freqs_hz``
        is a coarse science-channel grid rather than a fine RFI-survey grid.

    Returns
    -------
    T_K : ndarray, shape (nfreq,)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    tones = np.atleast_1d(np.asarray(tone_freqs_hz, dtype=np.float64))
    if tone_width_hz is None:
        tone_width_hz = 0.5 * np.median(np.diff(np.sort(f))) if len(f) > 1 else 1.0
    T = np.zeros_like(f)
    for tf in tones:
        T[np.abs(f - tf) <= tone_width_hz] = tone_temp_K
    return T


def k_to_v_per_m_root_hz(temp_k, freq_hz):
    """Brightness temperature [K] -> spectral field strength [V/m/sqrt(Hz)].

    Rayleigh-Jeans: radiance ``2 k_B T / lambda**2``, integrated over a full
    sphere (``4 pi``) and converted to field strength through the impedance
    of free space.

    Parameters
    ----------
    temp_k : float or array_like
        Brightness temperature [K].
    freq_hz : float or array_like
        Frequency [Hz].

    Returns
    -------
    float or ndarray
        Field strength [V / m / sqrt(Hz)].
    """
    lam = c / freq_hz
    radiance = 2 * k_B * temp_k / lam**2
    flux_density = 4 * pi * radiance
    return np.sqrt(flux_density * eta_0)


# ---------------------------------------------------------------------------
# Systematic injection (absorbed from bloom21cm/src/systematics.py)
#
# These operate on the array layout used by the linear simulators:
#   masks   (nobs, npix)          -- 1 where a pixel is visible
#   beams   (nobs, ndipole, npix)
#   omega_B (nobs, ndipole)       -- full-sphere beam solid angle
#
# Truth injection is deliberately kept separate from the recovery design
# columns, so a study can stress-test a mismatched model instead of fitting
# back exactly what was injected.
# ---------------------------------------------------------------------------


def normalized_beam_weights(beams, omega_B):
    """Return beam weights normalised by full-sphere beam solid angle."""
    beams = np.asarray(beams, dtype=float)
    omega_B = np.asarray(omega_B, dtype=float)
    if beams.ndim != 3:
        raise ValueError("beams must have shape (nobs, ndipole, npix)")
    if omega_B.shape != beams.shape[:2]:
        raise ValueError(
            f"omega_B must have shape {beams.shape[:2]}, got {omega_B.shape}"
        )
    return beams / omega_B[:, :, None]


def _expand_time_pixels(pixel_index, nobs, n_source_times=None):
    pix = np.asarray(pixel_index, dtype=int)
    if pix.ndim == 0:
        return np.full(nobs, int(pix), dtype=int)
    if pix.shape == (nobs,):
        return pix
    if n_source_times is not None and pix.shape == (n_source_times,):
        return pix[np.arange(nobs) % n_source_times]
    raise ValueError(
        "pixel_index must be scalar, length nobs, or length n_source_times"
    )


def point_source_coupling(
    masks, beams, omega_B, pixel_index, n_source_times=None
):
    """Beam/visibility coupling for a point source at one pixel per time.

    Returns an array with shape ``(nobs, ndipole)``.  This is the
    design-column coefficient for a source brightness at a single frequency.
    """
    masks = np.asarray(masks, dtype=float)
    weights = normalized_beam_weights(beams, omega_B)
    nobs, ndipole, npix = weights.shape
    if masks.shape != (nobs, npix):
        raise ValueError(
            f"masks must have shape {(nobs, npix)}, got {masks.shape}"
        )
    pix = _expand_time_pixels(pixel_index, nobs, n_source_times=n_source_times)
    if np.any((pix < 0) | (pix >= npix)):
        raise ValueError("pixel_index contains pixels outside the map")
    t_idx = np.arange(nobs)
    d_idx = np.arange(ndipole)
    return (
        weights[t_idx[:, None], d_idx[None, :], pix[:, None]]
        * masks[t_idx, pix][:, None]
    )


def point_source_signal(
    masks, beams, omega_B, pixel_index, temperature_K, n_source_times=None
):
    """Antenna-temperature contribution of a point source.

    ``temperature_K`` may be scalar, ``(nfreq,)``,
    ``(n_source_times, nfreq)``, or ``(nobs, nfreq)``.  The result has shape
    ``(nobs, ndipole, nfreq)``.
    """
    coupling = point_source_coupling(
        masks, beams, omega_B, pixel_index, n_source_times=n_source_times
    )
    nobs = coupling.shape[0]
    temp = np.asarray(temperature_K, dtype=float)
    if temp.ndim == 0:
        temp = temp[None, None]
    elif temp.ndim == 1:
        temp = temp[None, :]
    elif temp.ndim != 2:
        raise ValueError("temperature_K must be scalar, 1-D, or 2-D")
    if temp.shape[0] == 1:
        temp_obs = np.broadcast_to(temp, (nobs, temp.shape[1]))
    elif temp.shape[0] == nobs:
        temp_obs = temp
    elif n_source_times is not None and temp.shape[0] == n_source_times:
        temp_obs = temp[np.arange(nobs) % n_source_times]
    else:
        raise ValueError("temperature_K time axis is incompatible with masks")
    return coupling[:, :, None] * temp_obs[:, None, :]


def extended_source_coupling(masks, beams, omega_B, source_weights):
    """Coupling for an extended source given fractional pixel weights.

    ``source_weights`` has shape ``(nobs, npix)`` and is usually zero except
    over pixels covered by the source disc.  Visibility masking is applied
    here.
    """
    masks = np.asarray(masks, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    weights = normalized_beam_weights(beams, omega_B)
    if source_weights.shape != masks.shape:
        raise ValueError(
            f"source_weights must have shape {masks.shape}, "
            f"got {source_weights.shape}"
        )
    return np.sum(
        weights * masks[:, None, :] * source_weights[:, None, :], axis=2
    )


def extended_source_signal(
    masks, beams, omega_B, source_weights, temperature_K
):
    """Antenna-temperature contribution of an extended source."""
    coupling = extended_source_coupling(masks, beams, omega_B, source_weights)
    nobs = coupling.shape[0]
    temp = np.asarray(temperature_K, dtype=float)
    if temp.ndim == 0:
        temp = temp[None, None]
    elif temp.ndim == 1:
        temp = temp[None, :]
    elif temp.ndim != 2:
        raise ValueError("temperature_K must be scalar, 1-D, or 2-D")
    if temp.shape[0] == 1:
        temp = np.broadcast_to(temp, (nobs, temp.shape[1]))
    elif temp.shape[0] != nobs:
        raise ValueError("temperature_K time axis must be length 1 or nobs")
    return coupling[:, :, None] * temp[:, None, :]


def surface_emission_signal(masks, beams, omega_B, surface_temperature_K):
    """Blocked-surface antenna contribution from a nonuniform temp map.

    ``surface_temperature_K`` may be ``(npix,)`` or ``(npix, nfreq)``.  This
    is the full blocked-surface term, not just a residual against a scalar
    model.
    """
    masks = np.asarray(masks, dtype=float)
    weights = normalized_beam_weights(beams, omega_B)
    surface = np.asarray(surface_temperature_K, dtype=float)
    if surface.ndim == 1:
        if surface.shape[0] != masks.shape[1]:
            raise ValueError(
                "surface_temperature_K pixel axis does not match masks"
            )
        surface = surface[:, None]
    elif surface.ndim != 2 or surface.shape[0] != masks.shape[1]:
        raise ValueError(
            "surface_temperature_K must have shape (npix,) or (npix, nfreq)"
        )
    return np.einsum("tdp,tp,pf->tdf", weights, 1.0 - masks, surface)


def surface_residual_signal(
    masks, beams, omega_B, surface_temperature_K, scalar_model_K
):
    """Residual blocked-surface term after subtracting a scalar model."""
    surface = np.asarray(surface_temperature_K, dtype=float)
    return surface_emission_signal(
        masks, beams, omega_B, surface - scalar_model_K
    )


def synthetic_lunar_temperature_map(
    nside, base_K=300.0, dipole_K=20.0, quadrupole_K=5.0
):
    """Small deterministic Moon-fixed-like HEALPix temperature stress map."""
    import healpy

    npix = healpy.nside2npix(nside)
    xyz = np.asarray(healpy.pix2vec(nside, np.arange(npix)))
    z = xyz[2]
    return base_K + dipole_K * z + quadrupole_K * (1.5 * z**2 - 0.5)


def earth_rfi_signal(
    masks,
    beams,
    omega_B,
    earth_pixels,
    freqs_hz,
    *,
    tone_freqs_hz=None,
    **kwargs,
):
    """Occultation-gated Earth RFI contribution.

    Uses :func:`earth_rfi_temperature_K` for the flat in-band model, or
    :func:`earth_rfi_tone_temperature_K` when ``tone_freqs_hz`` is given.
    """
    if tone_freqs_hz is None:
        temp = earth_rfi_temperature_K(freqs_hz, **kwargs)
    else:
        temp = earth_rfi_tone_temperature_K(freqs_hz, tone_freqs_hz, **kwargs)
    n_source_times = len(np.atleast_1d(earth_pixels))
    return point_source_signal(
        masks,
        beams,
        omega_B,
        earth_pixels,
        temp,
        n_source_times=n_source_times,
    )


def quiet_sun_signal(
    masks, beams, omega_B, sun_pixels, freqs_hz, times, activity_kwargs=None
):
    """Occultation-gated quiet/slowly-variable Sun contribution."""
    temp = sun_temperature_K(
        freqs_hz, times, activity_kwargs=activity_kwargs
    )
    return point_source_signal(
        masks, beams, omega_B, sun_pixels, temp, n_source_times=len(temp)
    )
