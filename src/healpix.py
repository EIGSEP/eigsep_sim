"""Deprecated. HEALPix map classes moved to :mod:`healjax.maps`.

The class hierarchy (``HealpixBase`` / ``Alm`` / ``HealpixMap`` / ``HPM``)
now lives in :mod:`healjax.maps`, the interpolation kernels in
:mod:`healjax.interp`, and the dtype aliases are ``healjax.FLOAT_TYPE`` /
``healjax.INT_TYPE``.

Importing this module still works and re-exports the same names, but emits a
``DeprecationWarning``.
"""

import warnings

import healjax  # noqa: F401
from healjax import FLOAT_TYPE as float_dtype  # noqa: F401
from healjax import INT_TYPE as int_dtype  # noqa: F401
from healjax import get_interp_weights  # noqa: F401
from healjax.interp import (  # noqa: F401
    interpolate_map,
    rotate_interpolate_and_sum,
)
from healjax.maps import (  # noqa: F401
    HEALPIX_MODES,
    HPM,
    Alm,
    HealpixBase,
    HealpixMap,
    add2array,
    default_fits_format_codes,
    mk_arr,
)
from healjax.maps.hpm import ang2pix, vec2ang, vec2pix  # noqa: F401

__all__ = [
    "add2array",
    "mk_arr",
    "HEALPIX_MODES",
    "default_fits_format_codes",
    "HealpixBase",
    "Alm",
    "HealpixMap",
    "HPM",
    "vec2ang",
    "ang2pix",
    "vec2pix",
    "interpolate_map",
    "rotate_interpolate_and_sum",
    "float_dtype",
    "int_dtype",
    "get_interp_weights",
]

warnings.warn(
    "eigsep_sim.healpix is deprecated; import from healjax.maps "
    "(classes), healjax.interp (interpolation), or healjax "
    "(FLOAT_TYPE/INT_TYPE) instead.",
    DeprecationWarning,
    stacklevel=2,
)
