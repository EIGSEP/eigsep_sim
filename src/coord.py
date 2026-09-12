"""Deprecated. Coordinate transforms moved to :mod:`healjax.coord`.

``rot_m`` used to be copy-pasted into three EIGSEP packages; ``healjax.coord``
is now the single implementation.  It also dispatches on input type, so the
same functions serve NumPy callers and jitted JAX kernels.

Importing this module still works and re-exports the transforms, but emits a
``DeprecationWarning``.

Not carried over
----------------
``convert``, ``convert_m`` and ``sys_dict`` are **not** available here.  They
were thin wrappers over ``ephem``'s Equatorial/Ecliptic/Galactic classes, and
healjax deliberately does not depend on ``ephem``.  Use ``aipy.coord.convert``
/ ``aipy.coord.convert_m`` (identical implementations), or
:mod:`eigsep_base.coord` for astropy-based galactic/equatorial conversions.
"""

import warnings

from healjax.coord import (  # noqa: F401
    angles_to_coord,
    azalt2top,
    eq2radec,
    eq2top_m,
    latlong2xyz,
    radec2eq,
    rot_m,
    thphi2xyz,
    top2azalt,
    top2eq_m,
    xyz2thphi,
)

__all__ = [
    "rot_m",
    "xyz2thphi",
    "thphi2xyz",
    "eq2top_m",
    "top2eq_m",
    "eq2radec",
    "radec2eq",
    "latlong2xyz",
    "top2azalt",
    "azalt2top",
    "angles_to_coord",
]

_REMOVED = {
    "convert": "aipy.coord.convert",
    "convert_m": "aipy.coord.convert_m",
    "sys_dict": "aipy.coord.sys_dict",
}


def __getattr__(name):
    if name in _REMOVED:
        raise AttributeError(
            f"eigsep_sim.coord.{name} is gone: it depended on `ephem`, which "
            f"healjax.coord does not. Use {_REMOVED[name]} instead, or "
            f"eigsep_base.coord for astropy-based frame conversions."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


warnings.warn(
    "eigsep_sim.coord is deprecated; import from healjax.coord instead.",
    DeprecationWarning,
    stacklevel=2,
)
