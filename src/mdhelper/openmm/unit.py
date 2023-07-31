"""
OpenMM physical constants and unit conversions
==============================================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains physical constants and helper functions for unit
reduction.
"""

import numpy as np
from openmm import unit

VACUUM_PERMITTIVITY = 8.854187812813e-12 * unit.farad / unit.meter

def lj_scaling(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}, *,
        default: bool = True) -> dict[str, unit.Quantity]:

    r"""
    Computes scaling factors for Lennard-Jones reduced units.

    By default, the following scaling factors are calculated:

    * :code:`"molar_energy"`: :math:`N_\mathrm{A}\epsilon`
    * :code:`"time"`: :math:`\sqrt{m\sigma^2/\epsilon}`
    * :code:`"velocity"`: :math:`\sigma/\tau`
    * :code:`"force"`: :math:`\epsilon/\sigma`
    * :code:`"temperature"`: :math:`\epsilon/k_\mathrm{B}T`
    * :code:`"pressure"`: :math:`\epsilon/\sigma^3`
    * :code:`"dynamic_viscosity"`: :math:`\epsilon\tau/\sigma^3`
    * :code:`"charge"`: :math:`\sqrt{4\pi\varepsilon_0\sigma\epsilon}`
    * :code:`"dipole"`: :math:`\sqrt{4\pi\varepsilon_0\sigma^3\epsilon}`
    * :code:`"electric_field"`: :math:`\sqrt{\epsilon/(4\pi\varepsilon_0\sigma^3)}`
    * :code:`"mass_density"`: :math:`m/\sigma^3`

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

    Other parameters
    ----------------    
    other : `dict`, optional
        Other scaling factors to compute. The key should be the name of
        the scaling factor, and the value should contain `tuple`
        objects with the names of bases or other scaling factors and
        their powers.

    default : `bool`, default: `True`
        Determines whether the default scaling factors are calculated.

    Returns
    -------
    bases : `dict`
        Scaling factors.
    """

    if default:
        bases["molar_energy"] = bases["energy"] * unit.AVOGADRO_CONSTANT_NA
        bases["time"] = (
            bases["mass"] * bases["length"] ** 2 / bases["molar_energy"]
        ).sqrt().in_units_of(unit.picosecond)
        bases["velocity"] = bases["length"] / bases["time"]
        bases["force"] = bases["molar_energy"] / bases["length"]
        bases["temperature"] = bases["energy"] / unit.BOLTZMANN_CONSTANT_kB
        bases["pressure"] = bases["energy"] / bases["length"] ** 3
        bases["dynamic_viscosity"] = bases["pressure"] * bases["time"]
        bases["charge"] = (
            4 * np.pi * VACUUM_PERMITTIVITY * bases["length"] * bases["energy"]
        ).sqrt().in_units_of(unit.elementary_charge)
        bases["dipole"] = bases["length"] * bases["charge"]
        bases["electric_field"] = bases["force"] / bases["charge"]
        bases["mass_density"] = bases["mass"] / (bases["length"] ** 3 
                                                 * unit.AVOGADRO_CONSTANT_NA)

    for name, params in other.items():
        factor = 1
        for base, power in params:
            factor *= bases[base] ** power
        bases[name] = factor

    return bases