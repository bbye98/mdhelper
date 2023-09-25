"""
MDHelper
========
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>
"""

from typing import TypeVar
import numpy as np
ArrayLike = TypeVar("ArrayLike", list, np.ndarray, tuple)

VERSION = "0.0.2"
__all__ = ["algorithm", "analysis", "fit", "openmm", "plot"]

from . import algorithm, analysis, fit, openmm, plot # noqa: E402