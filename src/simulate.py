"""Deprecated. ``simulate`` was renamed to :mod:`eigsep_sim.forward_model`.

The name ``simulate`` collided with the ``ForwardModel.simulate`` method and
read as a verb for what is really the forward-model module.

Importing this module still works and re-exports the same names, but emits a
``DeprecationWarning``.
"""

import warnings

from .forward_model import *  # noqa: F401,F403
from .forward_model import ForwardModel, StackedForwardModel  # noqa: F401

__all__ = ["ForwardModel", "StackedForwardModel"]

warnings.warn(
    "eigsep_sim.simulate is deprecated; use eigsep_sim.forward_model "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)
