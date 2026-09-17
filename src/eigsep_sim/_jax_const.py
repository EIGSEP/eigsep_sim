"""
JAX-specific constants and global JAX configuration.

The physical constants formerly defined in ``eigsep_sim.const`` now live in
:mod:`eigsep_base.const`, which is deliberately JAX-free.  The two things
that could *not* move there are kept here:

1. ``DTYPE_R_JAX`` -- the real-array dtype used by the jitted kernels.
2. The ``jax_enable_x64`` configuration call.

The configuration call is an import-time side effect, and it must run before
any JAX array is created -- otherwise JAX silently truncates to 32-bit and
``DTYPE_R_JAX`` below would resolve to ``float32``.  ``eigsep_sim/__init__.py``
therefore imports this module first, and every module that needs
``DTYPE_R_JAX`` imports it from here rather than defining its own alias.
"""

import jax
import jax.numpy as jnp

# Calibration is curvature-limited and benefits from consistent 64-bit
# arithmetic. Set this before other modules create JAX arrays.
jax.config.update("jax_enable_x64", True)

# Data type for real JAX arrays.
DTYPE_R_JAX = jnp.float64

__all__ = ["DTYPE_R_JAX"]
