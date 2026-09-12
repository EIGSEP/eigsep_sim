"""RF calibration chain modeling for radiometric systematics studies.

Absorbed from ``bloom21cm/src/rf_calibration.py`` and generalized: the
original was hard-wired to the BLOOM (lunar orbiter) switched-load chain and
imported its dipole model from ``bloom_config``.  Nothing here is
mission-specific any more -- the antenna model is inlined and the two
calibration strategies EIGSEP actually uses are each described by their own
config dataclass:

- :class:`SwitchedCalConfig` -- space/orbiter style.  An internal ambient
  load and a noise diode are switched in on a fixed cadence; gain and
  receiver temperature come from a Y-factor solve and are interpolated
  between calibration samples.  Drive it with
  :func:`simulate_switched_calibration`.
- :class:`VNACalConfig` -- ground style.  A VNA behind an
  open/short/load (OSL) standard set measures antenna S11 on a (typically
  much slower) cadence; the mismatch efficiency ``1 - |S11|^2`` is
  interpolated between VNA sweeps.  Drive it with
  :func:`simulate_vna_mismatch_correction`.

The two are independent and compose: a ground deployment generally runs
both (switched load for gain/Trx, VNA+OSL for mismatch), while an orbiter
typically carries only the switched load and relies on a pre-flight or
modeled S11.

These functions keep the RF model explicit and compact.  They are not a
replacement for a microwave network simulator; they provide the
parameterized forward model needed to flow error-budget allocations into
calibration cadence, thermal-drift, and S11 measurement requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eigsep_base.const import c as C_LIGHT

__all__ = [
    "RFPath",
    "SwitchedCalConfig",
    "VNACalConfig",
    "thin_dipole_radiation_resistance_ohm",
    "resonant_thin_dipole_impedance_ohm",
    "dipole_impedance_ohm",
    "reflection_coefficient",
    "cable_electrical_length_m",
    "cable_gamma_per_m",
    "antenna_balun_impedance_ohm",
    "antenna_s11_at_vna",
    "mismatch_efficiency_from_s11",
    "osl_s11_uncertainty",
    "first_order_thermal_response",
    "shadow_transition_target",
    "gain_temperature_model",
    "trx_temperature_model",
    "calibration_indices",
    "estimate_gain_trx_from_load_noise",
    "interpolate_calibration",
    "simulate_switched_calibration",
    "simulate_vna_mismatch_correction",
    "project_spectral_bias_mK",
]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RFPath:
    """Parameterized antenna/balun/cable path seen by the receiver or VNA."""

    dipole_length_m: float
    r_loss_ohm: float = 5.0
    x_scale: float = 120.0
    effective_radius_m: float = 0.0
    z0_ohm: float = 50.0
    balun_impedance_ratio: float = 1.0
    balun_series_loss_ohm: float = 0.5
    cable_length_m: float = 0.10
    cable_velocity_factor: float = 0.70
    cable_loss_db_per_m_100mhz: float = 0.1
    cable_alpha_tempco_per_K: float = 3.9e-3
    cable_length_tempco_per_K: float = 17e-6
    reference_temp_K: float = 300.0


@dataclass(frozen=True)
class SwitchedCalConfig:
    """Timing and source parameters for switched-load radiometry.

    This is the space/orbiter configuration: an internal ambient load and a
    noise diode, switched in on a fixed ``cycle_s`` cadence.
    """

    cycle_s: float = 120.0
    load_integration_s: float = 1.0
    noise_integration_s: float = 1.0
    settle_s: float = 0.25
    load_temp_K: float = 300.0
    noise_temp_K: float = 1000.0
    thermistor_error_K: float = 0.1

    @property
    def cal_time_s(self) -> float:
        """Total non-science time per load/noise calibration cycle."""
        return (
            self.load_integration_s
            + self.noise_integration_s
            + 2.0 * self.settle_s
        )

    @property
    def overhead_fraction(self) -> float:
        """Fractional observing overhead from the switched calibration."""
        return self.cal_time_s / self.cycle_s


@dataclass(frozen=True)
class VNACalConfig:
    """Cadence and standard quality for a ground VNA + OSL S11 measurement.

    The ground configuration measures antenna S11 directly with a VNA
    referenced to an open/short/load standard set.  OSL standards are not
    ideal, and that imperfection sets a floor on the achievable S11
    accuracy: see :func:`osl_s11_uncertainty`.

    Parameters
    ----------
    cycle_s : float
        Interval between VNA sweeps.  Typically far longer than a switched
        load cycle because a sweep takes the antenna out of the science
        path for much longer.
    sweep_duration_s : float
        Wall-clock time the antenna is unavailable per sweep.
    open_residual, short_residual, load_residual : float
        Residual reflection magnitude of each standard after calibration
        (0 is ideal).  Directivity/match errors of the standard set.
    directivity_db : float
        Effective residual directivity of the calibrated VNA port [dB].
        More negative is better.
    """

    cycle_s: float = 3600.0
    sweep_duration_s: float = 10.0
    open_residual: float = 0.005
    short_residual: float = 0.005
    load_residual: float = 0.01
    directivity_db: float = -40.0

    @property
    def overhead_fraction(self) -> float:
        """Fractional observing overhead from the VNA sweep cadence."""
        return self.sweep_duration_s / self.cycle_s


# ---------------------------------------------------------------------------
# Antenna model
#
# Inlined from bloom_config so this module stands alone.  Both functions are
# analytic proxies for requirements trades, not substitutes for NEC/MoM or
# measured S11 tables.
# ---------------------------------------------------------------------------


def thin_dipole_radiation_resistance_ohm(length_m, freq_mhz):
    """Approximate center-fed thin-dipole radiation resistance.

    ``length_m`` is the total deployed dipole length.  The model preserves
    the short-dipole limit, ``Rrad ~ 80 pi^2 (L/lambda)^2``, then blends
    into a simple sinusoidal-current resonant approximation with maxima
    near odd half-wave lengths and reduced feed resistance near
    integer-wave current-node cases.
    """
    f = np.asarray(freq_mhz, dtype=float)
    # total length in wavelengths
    x = np.maximum(float(length_m) * f * 1e6 / C_LIGHT, 1e-9)
    r_short = 80.0 * np.pi**2 * x**2
    blend = 1.0 - np.exp(-((x / 0.30) ** 4))
    r_res = 20.0 + 55.0 * np.sin(np.pi * x) ** 2
    return np.maximum((1.0 - blend) * r_short + blend * r_res, 1e-9)


def resonant_thin_dipole_impedance_ohm(
    length_m,
    freq_mhz,
    *,
    r_loss_ohm=5.0,
    x_scale=120.0,
    effective_radius_m=0.0,
):
    """Approximate feed impedance for a fixed-terminated resonant dipole.

    The reactance is a transmission-line-like
    ``-x_scale * pi * cot(pi * L / lambda)`` term.  It crosses through zero
    near odd half-wave resonances and becomes large near integer-wave
    lengths, exposing the effect of a fixed receiver impedance in the
    realised efficiency calculation.  ``effective_radius_m`` applies a
    finite-thickness regularization to the cotangent singularity.
    """
    f = np.asarray(freq_mhz, dtype=float)
    x = np.maximum(float(length_m) * f * 1e6 / C_LIGHT, 1e-9)
    phase = np.pi * x
    sinp = np.sin(phase)
    radius_ratio = max(float(effective_radius_m), 0.0) / max(
        float(length_m), 1e-12
    )
    thickness_floor = max(1e-4, 2.0 * radius_ratio)
    sinp = np.where(
        np.abs(sinp) < thickness_floor,
        np.sign(sinp + 1e-30) * thickness_floor,
        sinp,
    )
    cot = np.cos(phase) / sinp
    reactance = np.clip(-float(x_scale) * np.pi * cot, -1e6, 1e6)
    rrad = thin_dipole_radiation_resistance_ohm(length_m, f)
    return (rrad + float(r_loss_ohm)) + 1j * reactance


def dipole_impedance_ohm(
    length_m,
    freq_mhz,
    *,
    r_loss_ohm=5.0,
    x_scale=120.0,
    effective_radius_m=0.0,
):
    """Resonant thin-dipole feed impedance.

    Thin wrapper over :func:`resonant_thin_dipole_impedance_ohm`, kept as
    the name the rest of this module and its callers use.
    """
    return resonant_thin_dipole_impedance_ohm(
        length_m,
        freq_mhz,
        r_loss_ohm=r_loss_ohm,
        x_scale=x_scale,
        effective_radius_m=effective_radius_m,
    )


# ---------------------------------------------------------------------------
# Network / mismatch
# ---------------------------------------------------------------------------


def reflection_coefficient(z_load, z0_ohm=50.0):
    """Return complex reflection coefficient referenced to ``z0_ohm``.

    Note this is the *circuit* reflection coefficient of a load impedance.
    For the material/terrain reflection coefficient at a dielectric
    boundary, see :func:`eigsep_sim.reflectivity.reflection_coefficient`.
    """
    z_load = np.asarray(z_load, dtype=complex)
    return (z_load - z0_ohm) / (z_load + z0_ohm)


def cable_electrical_length_m(path: RFPath, physical_temp_K):
    """Thermally expanded RF path length."""
    return path.cable_length_m * (
        1.0
        + path.cable_length_tempco_per_K
        * (np.asarray(physical_temp_K) - path.reference_temp_K)
    )


def cable_gamma_per_m(path: RFPath, freq_mhz, physical_temp_K=None):
    """Complex propagation constant for the 50 ohm cable/path."""
    f = np.asarray(freq_mhz, dtype=float)
    if physical_temp_K is None:
        physical_temp_K = path.reference_temp_K
    loss_db_m = path.cable_loss_db_per_m_100mhz * np.sqrt(
        np.maximum(f, 1e-12) / 100.0
    )
    loss_db_m *= 1.0 + path.cable_alpha_tempco_per_K * (
        np.asarray(physical_temp_K) - path.reference_temp_K
    )
    alpha_np_m = loss_db_m * np.log(10.0) / 20.0
    beta_rad_m = (
        2.0 * np.pi * f * 1e6 / (C_LIGHT * path.cable_velocity_factor)
    )
    return alpha_np_m + 1j * beta_rad_m


def antenna_balun_impedance_ohm(path: RFPath, freq_mhz):
    """Antenna impedance transformed to the 50 ohm side of the balun."""
    z_ant = dipole_impedance_ohm(
        path.dipole_length_m,
        freq_mhz,
        r_loss_ohm=path.r_loss_ohm,
        x_scale=path.x_scale,
        effective_radius_m=path.effective_radius_m,
    )
    return z_ant / path.balun_impedance_ratio + path.balun_series_loss_ohm


def antenna_s11_at_vna(path: RFPath, freq_mhz, physical_temp_K=None):
    """S11 seen through a nominally 50 ohm path from VNA/switch to balun."""
    if physical_temp_K is None:
        physical_temp_K = path.reference_temp_K
    z_balun = antenna_balun_impedance_ohm(path, freq_mhz)
    gamma_load = reflection_coefficient(z_balun, path.z0_ohm)
    gamma_cable = cable_gamma_per_m(path, freq_mhz, physical_temp_K)
    length = cable_electrical_length_m(path, physical_temp_K)
    return gamma_load * np.exp(-2.0 * gamma_cable * length)


def mismatch_efficiency_from_s11(s11):
    """Available-power mismatch factor ``1 - |S11|^2`` clipped to [0, 1]."""
    s11 = np.asarray(s11, dtype=complex)
    return np.clip(1.0 - np.abs(s11) ** 2, 0.0, 1.0)


def osl_s11_uncertainty(cal: VNACalConfig, s11):
    """Approximate residual S11 magnitude error from imperfect OSL standards.

    Combines, in quadrature, the residual directivity floor with a
    reflection-tracking term that scales with the measured ``|S11|``.  This
    is the standard first-order VNA error budget, retaining only the terms
    an OSL (one-port) calibration leaves behind.

    Parameters
    ----------
    cal : VNACalConfig
    s11 : complex array_like
        Measured/true reflection coefficient.

    Returns
    -------
    ndarray
        1-sigma magnitude uncertainty on S11, same shape as ``s11``.
    """
    mag = np.abs(np.asarray(s11, dtype=complex))
    directivity = 10.0 ** (cal.directivity_db / 20.0)
    tracking = np.sqrt(
        cal.open_residual**2
        + cal.short_residual**2
        + cal.load_residual**2
    )
    return np.sqrt(directivity**2 + (tracking * mag) ** 2)


# ---------------------------------------------------------------------------
# Thermal and gain models
# ---------------------------------------------------------------------------


def first_order_thermal_response(
    times_s, target_temp_K, tau_s, initial_temp_K=None
):
    """First-order low-pass thermal response to a time-dependent target."""
    times = np.asarray(times_s, dtype=float)
    target = np.asarray(target_temp_K, dtype=float)
    if times.ndim != 1 or target.shape != times.shape:
        raise ValueError(
            "times_s and target_temp_K must be 1-D arrays with the same shape"
        )
    out = np.empty_like(target)
    out[0] = target[0] if initial_temp_K is None else float(initial_temp_K)
    tau_s = float(tau_s)
    if tau_s <= 0:
        return target.copy()
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        a = 1.0 - np.exp(-dt / tau_s)
        out[i] = out[i - 1] + a * (target[i] - out[i - 1])
    return out


def shadow_transition_target(
    times_s, *, period_s=7200.0, shadow_fraction=0.35, warm_K=300.0,
    cold_K=270.0,
):
    """Smooth periodic thermal target for shadow ingress/egress tests."""
    phase = (np.asarray(times_s, dtype=float) % period_s) / period_s
    in_shadow = phase < shadow_fraction
    target = np.where(in_shadow, cold_K, warm_K)
    # Smooth the square wave edges slightly so cadence requirements depend
    # on the chosen thermal time constant, not a numerical discontinuity.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    return np.convolve(target, kernel, mode="same")


def gain_temperature_model(
    component_temp_K,
    freq_mhz,
    *,
    gain_tempco_per_K=-0.005,
    ripple_amp=0.0,
    ripple_period_mhz=20.0,
):
    """Dimensionless gain drift model normalized to unity at 300 K."""
    temp = np.asarray(component_temp_K, dtype=float)
    f = np.asarray(freq_mhz, dtype=float)
    gain_t = 1.0 + gain_tempco_per_K * (temp[..., None] - 300.0)
    if ripple_amp:
        ripple = 1.0 + ripple_amp * np.sin(
            2.0 * np.pi * (f - f.min()) / ripple_period_mhz
        )
    else:
        ripple = 1.0
    return gain_t * ripple


def trx_temperature_model(
    component_temp_K, freq_mhz, *, trx_300K=100.0, trx_tempco_K_per_K=0.25
):
    """Receiver-noise temperature model."""
    temp = np.asarray(component_temp_K, dtype=float)
    f = np.asarray(freq_mhz, dtype=float)
    spectral = (f / 80.0) ** 0.05
    return (
        trx_300K + trx_tempco_K_per_K * (temp[..., None] - 300.0)
    ) * spectral


# ---------------------------------------------------------------------------
# Calibration cadence and solve
# ---------------------------------------------------------------------------


def calibration_indices(times_s, cycle_s):
    """Indices of samples at which a calibration is performed."""
    times = np.asarray(times_s, dtype=float)
    if cycle_s <= 0:
        raise ValueError("cycle_s must be positive")
    targets = np.arange(times[0], times[-1] + 0.5 * cycle_s, cycle_s)
    return np.unique(
        np.searchsorted(times, targets).clip(0, len(times) - 1)
    )


def estimate_gain_trx_from_load_noise(
    p_load, p_noise, load_temp_K, noise_temp_K
):
    """Y-factor style gain and receiver-temperature estimate."""
    p_load = np.asarray(p_load, dtype=float)
    p_noise = np.asarray(p_noise, dtype=float)
    delta_t = np.asarray(noise_temp_K, dtype=float)
    gain = (p_noise - p_load) / delta_t
    trx = p_load / gain - np.asarray(load_temp_K, dtype=float)
    return gain, trx


def interpolate_calibration(times_s, cal_idx, values):
    """Linearly interpolate calibration estimates from calibration samples."""
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(values, dtype=float)
    cal_t = times[np.asarray(cal_idx, dtype=int)]
    if values.ndim == 1:
        return np.interp(times, cal_t, values)
    out = np.empty((len(times),) + values.shape[1:], dtype=float)
    flat = values.reshape(values.shape[0], -1)
    out_flat = out.reshape(len(times), -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(times, cal_t, flat[:, j])
    return out


def simulate_switched_calibration(
    times_s,
    freqs_mhz,
    antenna_temp_K,
    *,
    cal: SwitchedCalConfig,
    component_temp_K,
    load_temp_K=None,
    noise_temp_K=None,
    rng=None,
    radiometer_sigma_K=0.0,
    gain_tempco_per_K=-0.005,
    trx_tempco_K_per_K=0.25,
    spectral_ripple_amp=0.0,
):
    """Simulate switched-load calibration; return residual antenna error.

    ``antenna_temp_K`` must have shape ``(ntime, nfreq)`` or ``(nfreq,)``.
    The output residual is ``T_ant_hat - T_ant`` after applying interpolated
    gain and receiver-temperature estimates.
    """
    times = np.asarray(times_s, dtype=float)
    freqs = np.asarray(freqs_mhz, dtype=float)
    ant = np.asarray(antenna_temp_K, dtype=float)
    if ant.ndim == 1:
        ant = np.broadcast_to(ant[None, :], (len(times), len(freqs)))
    if ant.shape != (len(times), len(freqs)):
        raise ValueError(
            "antenna_temp_K must have shape (ntime, nfreq) or (nfreq,)"
        )
    temp = np.asarray(component_temp_K, dtype=float)
    if temp.shape != times.shape:
        raise ValueError("component_temp_K must have shape (ntime,)")
    if rng is None:
        rng = np.random.default_rng(0)

    gain = gain_temperature_model(
        temp,
        freqs,
        gain_tempco_per_K=gain_tempco_per_K,
        ripple_amp=spectral_ripple_amp,
    )
    trx = trx_temperature_model(
        temp, freqs, trx_tempco_K_per_K=trx_tempco_K_per_K
    )
    if load_temp_K is None:
        load_temp_K = cal.load_temp_K
    if noise_temp_K is None:
        noise_temp_K = cal.noise_temp_K
    load = np.asarray(load_temp_K, dtype=float)
    noise = np.asarray(noise_temp_K, dtype=float)

    p_ant = gain * (ant + trx)
    cal_idx = calibration_indices(times, cal.cycle_s)
    load_meas_temp = load + rng.normal(
        0.0, cal.thermistor_error_K, size=(len(cal_idx), 1)
    )
    p_load = gain[cal_idx] * (load_meas_temp + trx[cal_idx])
    p_noise = gain[cal_idx] * (load_meas_temp + noise + trx[cal_idx])
    if radiometer_sigma_K:
        p_ant = p_ant + rng.normal(
            0.0, radiometer_sigma_K, size=p_ant.shape
        )
        p_load = p_load + rng.normal(
            0.0, radiometer_sigma_K, size=p_load.shape
        )
        p_noise = p_noise + rng.normal(
            0.0, radiometer_sigma_K, size=p_noise.shape
        )

    gain_cal, trx_cal = estimate_gain_trx_from_load_noise(
        p_load,
        p_noise,
        load_meas_temp,
        noise,
    )
    gain_hat = interpolate_calibration(times, cal_idx, gain_cal)
    trx_hat = interpolate_calibration(times, cal_idx, trx_cal)
    ant_hat = p_ant / gain_hat - trx_hat
    return {
        "antenna_temp_hat_K": ant_hat,
        "residual_K": ant_hat - ant,
        "gain_true": gain,
        "gain_hat": gain_hat,
        "trx_true_K": trx,
        "trx_hat_K": trx_hat,
        "cal_indices": cal_idx,
        "overhead_fraction": cal.overhead_fraction,
    }


def simulate_vna_mismatch_correction(
    times_s,
    freqs_mhz,
    antenna_temp_K,
    *,
    path: RFPath,
    component_temp_K,
    vna_cycle_s=None,
    cal: VNACalConfig = None,
    rng=None,
):
    """Simulate residual after interpolated VNA mismatch correction.

    The true delivered antenna temperature is scaled by the time-dependent
    mismatch factor derived from S11.  The recovery divides by an
    interpolated VNA estimate of that factor.  The returned residual is the
    remaining calibrated antenna-temperature error.

    Parameters
    ----------
    times_s, freqs_mhz, antenna_temp_K
        Time grid [s], frequency grid [MHz], and true antenna temperature
        with shape ``(ntime, nfreq)`` or ``(nfreq,)``.
    path : RFPath
    component_temp_K : ndarray, shape (ntime,)
        Physical temperature of the RF path.
    vna_cycle_s : float, optional
        Sweep cadence [s].  Ignored when ``cal`` is given.
    cal : VNACalConfig, optional
        Ground VNA + OSL configuration.  When supplied, its ``cycle_s``
        sets the cadence and its OSL standard residuals inject a realistic
        S11 measurement error via :func:`osl_s11_uncertainty`.  When
        omitted, the VNA is treated as perfect apart from interpolation
        between sweeps (the original bloom21cm behaviour).
    rng : numpy.random.Generator, optional
        Used only when ``cal`` is supplied.

    Returns
    -------
    dict
    """
    if cal is None and vna_cycle_s is None:
        raise ValueError("supply either vna_cycle_s or cal")
    cycle_s = cal.cycle_s if cal is not None else vna_cycle_s

    times = np.asarray(times_s, dtype=float)
    freqs = np.asarray(freqs_mhz, dtype=float)
    ant = np.asarray(antenna_temp_K, dtype=float)
    if ant.ndim == 1:
        ant = np.broadcast_to(ant[None, :], (len(times), len(freqs)))
    if ant.shape != (len(times), len(freqs)):
        raise ValueError(
            "antenna_temp_K must have shape (ntime, nfreq) or (nfreq,)"
        )
    temp = np.asarray(component_temp_K, dtype=float)
    if temp.shape != times.shape:
        raise ValueError("component_temp_K must have shape (ntime,)")

    s11_true = np.empty((len(times), len(freqs)), dtype=complex)
    for i, temp_i in enumerate(temp):
        s11_true[i] = antenna_s11_at_vna(path, freqs, physical_temp_K=temp_i)
    eta_true = mismatch_efficiency_from_s11(s11_true)

    vna_idx = calibration_indices(times, cycle_s)
    s11_cal = s11_true[vna_idx]
    if cal is not None:
        if rng is None:
            rng = np.random.default_rng(0)
        sigma = osl_s11_uncertainty(cal, s11_cal)
        s11_cal = s11_cal + rng.normal(0.0, sigma) + 1j * rng.normal(
            0.0, sigma
        )
    eta_cal = mismatch_efficiency_from_s11(s11_cal)
    eta_hat = interpolate_calibration(times, vna_idx, eta_cal)
    eta_hat = np.where(np.abs(eta_hat) < 1e-12, 1e-12, eta_hat)
    ant_hat = ant * eta_true / eta_hat
    return {
        "antenna_temp_hat_K": ant_hat,
        "residual_K": ant_hat - ant,
        "s11_true": s11_true,
        "eta_true": eta_true,
        "eta_hat": eta_hat,
        "vna_indices": vna_idx,
    }


def project_spectral_bias_mK(freqs_mhz, residual_K, template_K):
    """Project residual spectra onto a normalized spectral template."""
    freqs = np.asarray(freqs_mhz, dtype=float)
    resid = np.asarray(residual_K, dtype=float)
    tmpl = np.asarray(template_K, dtype=float)
    if tmpl.shape != freqs.shape:
        raise ValueError("template_K must have shape (nfreq,)")
    tmpl0 = tmpl - np.mean(tmpl)
    norm = float(np.dot(tmpl0, tmpl0))
    if norm == 0.0:
        raise ValueError("template_K must have non-zero spectral structure")
    coeff = resid @ tmpl0 / norm
    return coeff * float(np.std(tmpl0)) * 1e3
