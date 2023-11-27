"""
MDHelper
========
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>
"""

from typing import TypeVar
import numpy as np
ArrayLike = TypeVar("ArrayLike", list, np.ndarray, tuple)

try:
    import openmm
    FOUND_OPENMM = True
except:
    FOUND_OPENMM = False

from pint import UnitRegistry
ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity

VERSION = "1.0.0"
__all__ = ["algorithm", "analysis", "fit", "openmm", "plot"]

from . import algorithm, analysis, fit, openmm, plot # noqa: E402