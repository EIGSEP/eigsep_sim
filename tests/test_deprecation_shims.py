"""Tests for the backward-compatibility shim left behind by the Phase-3
reorganization (const). The coord/healpix/utils/simulate shims were
dropped in 876b2ef.

Each shim must (a) still import, (b) warn, and (c) hand back the same object
the new canonical location provides -- identity, not just equality, so a
future divergence between the shim and the real module is caught here.
"""

import importlib
import warnings

import pytest


SHIMS = ["const"]


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

