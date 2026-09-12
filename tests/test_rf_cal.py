"""Tests for eigsep_sim.rf_cal (absorbed and generalized from bloom21cm)."""

import numpy as np
import pytest

from eigsep_sim.rf_cal import (
    RFPath,
    SwitchedCalConfig,
    VNACalConfig,
    antenna_s11_at_vna,
    calibration_indices,
    estimate_gain_trx_from_load_noise,
    first_order_thermal_response,
    gain_temperature_model,
    mismatch_efficiency_from_s11,
    osl_s11_uncertainty,
    project_spectral_bias_mK,
    reflection_coefficient,
    resonant_thin_dipole_impedance_ohm,
    simulate_switched_calibration,
    simulate_vna_mismatch_correction,
    thin_dipole_radiation_resistance_ohm,
)

FREQS_MHZ = np.linspace(50.0, 120.0, 8)
TIMES_S = np.arange(0.0, 3600.0, 60.0)


@pytest.fixture
def path():
    return RFPath(dipole_length_m=2.0)


class TestAntennaModel:
    def test_short_dipole_radiation_resistance_limit(self):
        # At L << lambda, Rrad -> 80 pi^2 (L/lambda)^2.
        length = 0.05
        f = np.array([10.0])  # MHz -> L/lambda = 0.00167
        from eigsep_base.const import c

        x = length * f * 1e6 / c
        got = thin_dipole_radiation_resistance_ohm(length, f)
        np.testing.assert_allclose(got, 80 * np.pi**2 * x**2, rtol=1e-3)

    def test_impedance_is_complex_and_finite(self):
        z = resonant_thin_dipole_impedance_ohm(2.0, FREQS_MHZ)
        assert np.iscomplexobj(z)
        assert np.all(np.isfinite(z))

    def test_resistance_positive(self):
        z = resonant_thin_dipole_impedance_ohm(2.0, FREQS_MHZ)
        assert np.all(z.real > 0)

    def test_no_bloom_config_dependency(self):
        import eigsep_sim.rf_cal as m
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(m.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    imported.add(a.name.split(".")[0])
        assert "bloom_config" not in imported
        assert "bloom_sim" not in imported


class TestMismatch:
    def test_matched_load_has_zero_reflection(self):
        assert reflection_coefficient(50.0, 50.0) == 0

    def test_open_and_short_reflect_fully(self):
        assert abs(reflection_coefficient(1e12, 50.0)) == pytest.approx(
            1.0, abs=1e-6
        )
        assert abs(reflection_coefficient(0.0, 50.0)) == pytest.approx(1.0)

    def test_efficiency_bounded(self, path):
        s11 = antenna_s11_at_vna(path, FREQS_MHZ)
        eta = mismatch_efficiency_from_s11(s11)
        assert np.all((eta >= 0) & (eta <= 1))

    def test_perfect_match_gives_unit_efficiency(self):
        assert mismatch_efficiency_from_s11(0.0) == pytest.approx(1.0)


class TestOSLUncertainty:
    def test_ideal_standards_leave_only_directivity(self):
        cal = VNACalConfig(
            open_residual=0.0,
            short_residual=0.0,
            load_residual=0.0,
            directivity_db=-40.0,
        )
        got = osl_s11_uncertainty(cal, 0.5)
        assert got == pytest.approx(10 ** (-40.0 / 20.0))

    def test_worse_standards_increase_uncertainty(self):
        good = VNACalConfig(open_residual=0.001)
        bad = VNACalConfig(open_residual=0.05)
        assert osl_s11_uncertainty(bad, 0.8) > osl_s11_uncertainty(good, 0.8)

    def test_scales_with_reflection_magnitude(self):
        cal = VNACalConfig()
        assert osl_s11_uncertainty(cal, 0.9) > osl_s11_uncertainty(cal, 0.1)


class TestConfigs:
    def test_switched_overhead_fraction(self):
        cal = SwitchedCalConfig(
            cycle_s=100.0,
            load_integration_s=1.0,
            noise_integration_s=1.0,
            settle_s=0.5,
        )
        assert cal.cal_time_s == pytest.approx(3.0)
        assert cal.overhead_fraction == pytest.approx(0.03)

    def test_vna_overhead_fraction(self):
        cal = VNACalConfig(cycle_s=3600.0, sweep_duration_s=36.0)
        assert cal.overhead_fraction == pytest.approx(0.01)


class TestCalibrationCadence:
    def test_indices_within_range(self):
        idx = calibration_indices(TIMES_S, 300.0)
        assert idx.min() >= 0
        assert idx.max() < len(TIMES_S)

    def test_indices_sorted_unique(self):
        idx = calibration_indices(TIMES_S, 300.0)
        assert np.all(np.diff(idx) > 0)

    def test_shorter_cycle_gives_more_calibrations(self):
        assert len(calibration_indices(TIMES_S, 120.0)) > len(
            calibration_indices(TIMES_S, 600.0)
        )

    def test_nonpositive_cycle_raises(self):
        with pytest.raises(ValueError, match="cycle_s must be positive"):
            calibration_indices(TIMES_S, 0.0)


class TestYFactor:
    def test_recovers_injected_gain_and_trx(self):
        gain, trx = 2.5, 80.0
        t_load, t_noise = 300.0, 1000.0
        p_load = gain * (t_load + trx)
        p_noise = gain * (t_load + t_noise + trx)
        g_hat, trx_hat = estimate_gain_trx_from_load_noise(
            p_load, p_noise, t_load, t_noise
        )
        assert g_hat == pytest.approx(gain)
        assert trx_hat == pytest.approx(trx)


class TestThermalResponse:
    def test_constant_target_is_fixed_point(self):
        target = np.full_like(TIMES_S, 290.0)
        out = first_order_thermal_response(TIMES_S, target, 100.0)
        np.testing.assert_allclose(out, 290.0)

    def test_zero_tau_follows_target_exactly(self):
        target = 290.0 + 10.0 * np.sin(TIMES_S / 500.0)
        out = first_order_thermal_response(TIMES_S, target, 0.0)
        np.testing.assert_allclose(out, target)

    def test_lags_a_step(self):
        target = np.where(TIMES_S < 1800.0, 280.0, 300.0)
        out = first_order_thermal_response(TIMES_S, target, 600.0)
        step = np.searchsorted(TIMES_S, 1800.0)
        # immediately after the step the response has not caught up
        assert out[step] < target[step]
        assert out[-1] == pytest.approx(300.0, abs=1.0)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            first_order_thermal_response(TIMES_S, np.zeros(3), 10.0)


class TestSwitchedCalibration:
    def test_isothermal_noiseless_recovery_is_near_exact(self):
        cal = SwitchedCalConfig(cycle_s=300.0, thermistor_error_K=0.0)
        temp = np.full_like(TIMES_S, 300.0)
        ant = np.full((len(TIMES_S), len(FREQS_MHZ)), 250.0)
        out = simulate_switched_calibration(
            TIMES_S,
            FREQS_MHZ,
            ant,
            cal=cal,
            component_temp_K=temp,
            radiometer_sigma_K=0.0,
        )
        np.testing.assert_allclose(out["residual_K"], 0.0, atol=1e-8)

    def test_thermal_drift_leaves_residual(self):
        cal = SwitchedCalConfig(cycle_s=1800.0, thermistor_error_K=0.0)
        temp = 300.0 + 10.0 * np.sin(2 * np.pi * TIMES_S / 3600.0)
        ant = np.full((len(TIMES_S), len(FREQS_MHZ)), 250.0)
        out = simulate_switched_calibration(
            TIMES_S,
            FREQS_MHZ,
            ant,
            cal=cal,
            component_temp_K=temp,
            radiometer_sigma_K=0.0,
        )
        assert np.max(np.abs(out["residual_K"])) > 1e-6

    def test_faster_cadence_reduces_drift_residual(self):
        temp = 300.0 + 10.0 * np.sin(2 * np.pi * TIMES_S / 3600.0)
        ant = np.full((len(TIMES_S), len(FREQS_MHZ)), 250.0)
        resid = {}
        for cycle in (300.0, 1800.0):
            out = simulate_switched_calibration(
                TIMES_S,
                FREQS_MHZ,
                ant,
                cal=SwitchedCalConfig(
                    cycle_s=cycle, thermistor_error_K=0.0
                ),
                component_temp_K=temp,
                radiometer_sigma_K=0.0,
            )
            resid[cycle] = np.max(np.abs(out["residual_K"]))
        assert resid[300.0] < resid[1800.0]

    def test_rejects_bad_component_temp_shape(self):
        with pytest.raises(ValueError, match="component_temp_K"):
            simulate_switched_calibration(
                TIMES_S,
                FREQS_MHZ,
                np.zeros((len(TIMES_S), len(FREQS_MHZ))),
                cal=SwitchedCalConfig(),
                component_temp_K=np.zeros(3),
            )


class TestVNAMismatch:
    def test_isothermal_recovery_is_exact(self, path):
        temp = np.full_like(TIMES_S, 300.0)
        ant = np.full((len(TIMES_S), len(FREQS_MHZ)), 250.0)
        out = simulate_vna_mismatch_correction(
            TIMES_S,
            FREQS_MHZ,
            ant,
            path=path,
            component_temp_K=temp,
            vna_cycle_s=600.0,
        )
        np.testing.assert_allclose(out["residual_K"], 0.0, atol=1e-9)

    def test_osl_config_injects_error(self, path):
        temp = np.full_like(TIMES_S, 300.0)
        ant = np.full((len(TIMES_S), len(FREQS_MHZ)), 250.0)
        out = simulate_vna_mismatch_correction(
            TIMES_S,
            FREQS_MHZ,
            ant,
            path=path,
            component_temp_K=temp,
            cal=VNACalConfig(cycle_s=600.0, load_residual=0.05),
            rng=np.random.default_rng(1),
        )
        # imperfect standards must leave a residual even when isothermal
        assert np.max(np.abs(out["residual_K"])) > 0

    def test_requires_a_cadence(self, path):
        with pytest.raises(ValueError, match="vna_cycle_s or cal"):
            simulate_vna_mismatch_correction(
                TIMES_S,
                FREQS_MHZ,
                np.zeros((len(TIMES_S), len(FREQS_MHZ))),
                path=path,
                component_temp_K=np.full_like(TIMES_S, 300.0),
            )


class TestSpectralProjection:
    def test_zero_residual_projects_to_zero(self):
        tmpl = np.linspace(1.0, 2.0, len(FREQS_MHZ))
        resid = np.zeros((4, len(FREQS_MHZ)))
        np.testing.assert_allclose(
            project_spectral_bias_mK(FREQS_MHZ, resid, tmpl), 0.0
        )

    def test_flat_template_raises(self):
        tmpl = np.ones(len(FREQS_MHZ))
        with pytest.raises(ValueError, match="non-zero spectral structure"):
            project_spectral_bias_mK(
                FREQS_MHZ, np.zeros((2, len(FREQS_MHZ))), tmpl
            )

    def test_scales_linearly_with_residual(self):
        tmpl = np.linspace(1.0, 2.0, len(FREQS_MHZ))
        resid = np.tile(tmpl, (3, 1))
        a = project_spectral_bias_mK(FREQS_MHZ, resid, tmpl)
        b = project_spectral_bias_mK(FREQS_MHZ, 2 * resid, tmpl)
        np.testing.assert_allclose(b, 2 * a)


class TestGainModel:
    def test_unity_at_reference_temperature(self):
        g = gain_temperature_model(np.array([300.0]), FREQS_MHZ)
        np.testing.assert_allclose(g, 1.0)
