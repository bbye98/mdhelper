"""
MDHelper
========
A batteries-included toolkit of analysis modules and helper functions
for molecular dynamics (MD) simulations.
"""

from importlib.util import find_spec

from pint import UnitRegistry
ureg = UnitRegistry(auto_reduce_dimensions=True)
Q_ = ureg.Quantity

VERSION = "1.0.0"
FOUND_OPENMM = find_spec("openmm") is not None
__all__ = ["algorithm", "analysis", "fit", "plot", "FOUND_OPENMM", "VERSION"]

from . import algorithm, analysis, fit, plot # noqa: E402
if FOUND_OPENMM:
    __all__.append("openmm")