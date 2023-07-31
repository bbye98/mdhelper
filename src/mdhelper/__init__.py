"""
MDHelper
========
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>
"""

from typing import TypeVar
import numpy as np
ArrayLike = TypeVar("ArrayLike", list, np.ndarray, tuple)

VERSION = "0.0.1"
__all__ = ["algorithm", "analysis", "fit", "openmm", "plot", "utility"]

from . import algorithm, analysis, fit, openmm, plot, utility