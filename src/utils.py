"""Deprecated. The contents of ``eigsep_sim.utils`` were absorbed elsewhere.

New locations:

=============================  ==========================================
Old name                       New home
=============================  ==========================================
``moon_surface_distance``      :mod:`eigsep_sim.ephemeris`
``moon_reflect_vector``        :mod:`eigsep_sim.ephemeris`
``sample_disk``                :mod:`eigsep_sim.ephemeris`
``k_to_v_per_m_root_hz``       :mod:`eigsep_sim.sources`
``reflectivity``               :mod:`eigsep_sim.reflectivity`
=============================  ==========================================

``reflectivity`` was a thin wrapper over ``eigsep_terrain.reflectivity``;
that module now lives in :mod:`eigsep_sim.reflectivity`, so the wrapper is
redundant.  It is reimplemented below against the new module so existing
callers keep working, but it still routes through the deprecated
``permittivity_from_conductivity`` (which returns a refractive index, not a
permittivity, despite the name).  New code should call
:func:`eigsep_sim.reflectivity.complex_ref_index` and
:func:`eigsep_sim.reflectivity.reflection_coefficient` directly.

Importing this module still works, but emits a ``DeprecationWarning``.
"""

import warnings

from .ephemeris import (  # noqa: F401
    moon_reflect_vector,
    moon_surface_distance,
    sample_disk,
)
from .sources import k_to_v_per_m_root_hz  # noqa: F401
from . import reflectivity as _reflectivity_mod

__all__ = [
    "moon_surface_distance",
    "moon_reflect_vector",
    "reflectivity",
    "sample_disk",
    "k_to_v_per_m_root_hz",
]


def reflectivity(freqs, resistivity_ohm_m, eta0=1):
    """Deprecated. Field reflection coefficient from bulk resistivity.

    Equivalent to the original ``eigsep_sim.utils.reflectivity``: assumes a
    relative permittivity of 1 and derives the refractive index from
    conductivity alone.  Prefer
    ``reflection_coefficient(complex_ref_index(eps_r, sigma, freqs))`` with a
    real ``eps_r``.
    """
    conductivity = _reflectivity_mod.conductivity_from_resistivity(
        resistivity_ohm_m
    )
    eta = _reflectivity_mod.permittivity_from_conductivity(
        conductivity, freqs
    )
    return _reflectivity_mod.reflection_coefficient(eta, eta0=eta0)


warnings.warn(
    "eigsep_sim.utils is deprecated; its contents moved to "
    "eigsep_sim.ephemeris, eigsep_sim.sources and eigsep_sim.reflectivity.",
    DeprecationWarning,
    stacklevel=2,
)
