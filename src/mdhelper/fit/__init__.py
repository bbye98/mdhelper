r"""
Curve fitting
=============
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module provides a library of curve fitting models, such as 
exponential, Fourier series, polynomial, and power law models, for use
with :func:`scipy.optimize.curve_fit`.
"""

from . import distribution, exponential, fourier, gaussian, polynomial, power

__all__ = ["distribution", "exponential", "fourier", "gaussian", "polynomial",
           "power"]