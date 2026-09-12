"""Tests for the backward-compatibility shims left behind by the Phase-3
reorganization (const/coord/healpix/utils/simulate).

Each shim must (a) still import, (b) warn, and (c) hand back the same object
the new canonical location provides -- identity, not just equality, so a
future divergence between the shim and the real module is caught here.
"""

import importlib
import warnings

import numpy as np
import pytest


SHIMS = ["const", "coord", "healpix", "utils", "simulate"]


def _fresh_import(name):
    """Import eigsep_sim.<name>, forcing module-level code to re-run."""
    full = f"eigsep_sim.{name}"
    import sys

    sys.modules.pop(full, None)
    return importlib.import_module(full)


@pytest.mark.parametrize("name", SHIMS)
def test_shim_warns_on_import(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fresh_import(name)
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    assert messages, f"eigsep_sim.{name} did not emit a DeprecationWarning"
    assert any(
        f"eigsep_sim.{name}" in m for m in messages
    ), f"warning does not name the deprecated module: {messages}"


@pytest.mark.parametrize("name", SHIMS)
def test_shim_importable_without_error(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _fresh_import(name) is not None


class TestConstShim:
    def test_reexports_eigsep_base_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import const
        import eigsep_base.const as base

        for name in ("c", "k_B", "R_MOON", "R_SUN", "R_EARTH", "GM_MOON",
                     "pi", "eta_0", "h", "DTYPE_R_NPY"):
            assert getattr(const, name) == getattr(base, name), name

    def test_jax_dtype_still_available_and_64bit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import const
        import jax.numpy as jnp

        assert const.DTYPE_R_JAX is jnp.float64


class TestCoordShim:
    def test_functions_are_healjax_objects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import coord
        import healjax.coord as hc

        for name in coord.__all__:
            assert getattr(coord, name) is getattr(hc, name), name

    @pytest.mark.parametrize("name", ["convert", "convert_m", "sys_dict"])
    def test_ephem_only_names_raise_informative_error(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import coord

        with pytest.raises(AttributeError, match="aipy.coord"):
            getattr(coord, name)


class TestHealpixShim:
    def test_classes_are_healjax_objects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import healpix
        import healjax
        import healjax.maps as hm
        import healjax.interp as hi

        assert healpix.HPM is hm.HPM
        assert healpix.Alm is hm.Alm
        assert healpix.HealpixMap is hm.HealpixMap
        assert healpix.HealpixBase is hm.HealpixBase
        assert healpix.add2array is hm.add2array
        assert healpix.interpolate_map is hi.interpolate_map
        assert healpix.rotate_interpolate_and_sum is (
            hi.rotate_interpolate_and_sum
        )
        assert healpix.float_dtype is healjax.FLOAT_TYPE
        assert healpix.int_dtype is healjax.INT_TYPE


class TestSimulateShim:
    def test_forward_model_is_same_class(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import simulate
        from eigsep_sim import forward_model

        assert simulate.ForwardModel is forward_model.ForwardModel
        assert (
            simulate.StackedForwardModel
            is forward_model.StackedForwardModel
        )


class TestUtilsShim:
    def test_absorbed_functions_are_same_objects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import utils
        from eigsep_sim import ephemeris, sources

        assert utils.moon_surface_distance is ephemeris.moon_surface_distance
        assert utils.moon_reflect_vector is ephemeris.moon_reflect_vector
        assert utils.sample_disk is ephemeris.sample_disk
        assert utils.k_to_v_per_m_root_hz is sources.k_to_v_per_m_root_hz

    def test_reflectivity_wrapper_matches_new_module(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from eigsep_sim import utils
        from eigsep_sim import reflectivity as refl

        freqs = np.linspace(50e6, 150e6, 11)
        resistivity = 1e4

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            got = utils.reflectivity(freqs, resistivity)
            sigma = refl.conductivity_from_resistivity(resistivity)
            expected = refl.reflection_coefficient(
                refl.complex_ref_index(1, sigma, freqs)
            )
        np.testing.assert_allclose(got, expected)
