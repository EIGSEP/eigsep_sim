"""Deprecated. Physical constants moved to :mod:`eigsep_base.const`.

``DTYPE_R_JAX`` could not move with them -- ``eigsep_base`` is deliberately
JAX-free -- so it (and the ``jax_enable_x64`` configuration call that must
accompany it) lives in ``eigsep_sim._jax_const``.

Importing this module still works and re-exports everything it used to, but
emits a ``DeprecationWarning``.
"""

import warnings

from eigsep_base.const import *  # noqa: F401,F403
from eigsep_base.const import (  # noqa: F401
    G,
    GM_MOON,
    DTYPE_R_NPY,
    Jy,
    R_EARTH,
    R_MOON,
    R_SUN,
    arcmin,
    arcsec,
    au,
    c,
    deg,
    description,
    e,
    eta_0,
    ft,
    h,
    k_B,
    len_ns,
    m_e,
    m_p,
    m_sun,
    pc,
    pi,
    r_sun,
    s_per_day,
    s_per_yr,
    sidereal_day,
    sigma_sb,
    sq_deg,
)

from ._jax_const import DTYPE_R_JAX  # noqa: F401

warnings.warn(
    "eigsep_sim.const is deprecated. Physical constants now live in "
    "eigsep_base.const; the JAX dtype DTYPE_R_JAX lives in "
    "eigsep_sim._jax_const.",
    DeprecationWarning,
    stacklevel=2,
)
