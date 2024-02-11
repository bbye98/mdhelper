"""
Utility algorithms
==================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains logical and mathematical utility functions used by
other MDHelper modules.
"""

from typing import Union

import numpy as np
import sympy

from .. import FOUND_OPENMM, Q_, ureg

if FOUND_OPENMM:
    from openmm import unit
    from ..openmm.unit import VACUUM_PERMITTIVITY

def closest_factors(
        value: int, n_factors: int, reverse: bool = False
    ) -> np.ndarray[int]:

    """
    Get the :math:`n` closest factors for a given number :math:`N`, 
    sorted in ascending order.

    Parameters
    ----------
    value : `int`
        Number :math:`N` to be factorized.

    n_factors : `int`
        Number of factors :math:`n` to return.

    reverse : `bool`, optional, default: :code:`False`
        Specifies whether to sort in descending order.

    Returns
    -------
    factors : `np.ndarray`
        :math:`n` closest factors for `N`.

        **Shape**: :math:`(n,)`.
    """

    # Take the n-th root of N
    rt = value ** (1 / n_factors)
    rt_int = int(np.round(rt))
    if np.isclose(rt, rt_int):
        return rt_int * np.ones(n_factors, dtype=int)
    
    # Get all factors of N
    _factors = np.fromiter(
        (factor for factor, power in sympy.ntheory.factorint(value).items() 
         for _ in range(power)),
        dtype=int
    )

    # Find n closest factors
    i = 0
    factors = np.ones(n_factors, dtype=int)
    for j, f in enumerate(_factors[::-1]):
        while True:
            if i < n_factors:
                m = factors[i] * f
                if m <= rt_int or j < n_factors and factors[i] == 1:
                    factors[i] = m
                    break
                i += 1
            else:
                factors[np.argmin(factors)] *= f
                break

    if reverse:
        # Sort factors in descending order, if desired
        return np.sort(factors)[::-1]
    return np.sort(factors)

def replicate(
        cell_dims: np.ndarray[float], cell_pos: np.ndarray[float], 
        n_cells: np.ndarray[int]) -> np.ndarray[float]:

    r"""
    Replicate point(s) in an unit cell along the :math:`x`-, :math:`y`-,
    and :math:`z`-directions.

    Parameters
    ----------
    cell_dims : `numpy.ndarray`
        Dimensions of the unit cell.

        **Shape**: :math:`(3,)`.

    cell_pos : `numpy.ndarray`
        Positions of the :math:`N` points inside the unit cell.

        **Shape**: :math:`(N,\,3)`.

    n_cells : `numpy.ndarray`
        Number of times to replicate the unit cell in each direction.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    pos : `numpy.ndarray`
        Positions of the original and replicated points.
    """

    # Add cell x-dimensions to cell x-positions and replicate them
    # n_y * n_z times
    x = np.tile(
        np.concatenate(cell_pos[:, 0] 
                       + (cell_dims[0] * np.arange(n_cells[0]))[:, None]),
        reps=n_cells[1] * n_cells[2]
    )

    # Replicate cell y-positions n_x times, add cell y-dimensions to
    # them, and then replicate them n_z times
    y = np.tile(
        np.concatenate(np.tile(cell_pos[:, 1], reps=n_cells[0])
                       + (np.arange(n_cells[1]) * cell_dims[1])[:, None]),
        reps=n_cells[2]
    )

    # Replicate cell z-positions n_x * n_y times and add cell
    # z-dimensions to them
    z = np.concatenate(np.tile(cell_pos[:, 2], reps=n_cells[0] * n_cells[1])
                       + cell_dims[2] * np.arange(n_cells[2])[:, None])
    
    return np.vstack((x, y, z)).T

def rebin(x: np.ndarray[float], factor: int = None) -> np.ndarray[float]:

    r"""
    Rebin discrete data.

    Parameters
    ----------
    x : `numpy.ndarray`
        Discrete data to be rebinned in the last dimension.

    factor : `int`, optional
        Size reduction factor. If not specified, the biggest factor
        on the order of :math:`\mathcal{O}(1)`, if available, is used.

    Returns
    -------
    xr : `numpy.ndarray`
        Rebinned discrete data.
    """

    if factor is None:
        factors = np.array(sympy.divisors(x.shape[-1]))
        factor_indices = np.where(factors < 10)[0]
        if len(factor_indices):
            factor = factors[factor_indices[-1]]
        else:
            raise ValueError("No factor provided for rebinning.")

    return x.reshape((*x.shape[:-1], -1, factor)).mean(axis=-1)

def unit_scaling(
        bases: dict[str, Union["unit.Quantity", Q_]], 
        other: dict[str, list] = {}) -> dict[str, Union["unit.Quantity", Q_]]:

    r"""
    Computes scaling factors for reduced units.

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        **Format**: :code:`{"mass": <openmm.unit.Quantity> | <pint.Quantity>, 
        "length": <openmm.unit.Quantity> | <pint.Quantity>, 
        "energy": <openmm.unit.Quantity> | <pint.Quantity>}`.

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ/mol}`.
   
    other : `dict`, optional
        Other scaling factors to compute. The key should be the name of
        the scaling factor, and the value should contain `tuple`
        objects with the names of bases or default scaling factors and
        their powers.

        **Example**: 
        :code:`{"diffusivity": (("length", 2), ("time", -1))}`.

    Returns
    -------
    scales : `dict`
        Scaling factors.
    """
    
    # Evaluate the custom scaling factors
    for name, params in other.items():
        factor = 1
        for base, power in params:
            factor *= bases[base] ** power
        bases[name] = factor

    return bases

def lj_scaling(
        bases: dict[str, Union["unit.Quantity", Q_]], 
        other: dict[str, list] = {}) -> dict[str, Union["unit.Quantity", Q_]]:

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
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        **Format**: :code:`{"mass": <openmm.unit.Quantity> | <pint.Quantity>, 
        "length": <openmm.unit.Quantity> | <pint.Quantity>, 
        "energy": <openmm.unit.Quantity> | <pint.Quantity>}`.

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ/mol}`.
   
    other : `dict`, optional
        Other scaling factors to compute. The key should be the name of
        the scaling factor, and the value should contain `tuple`
        objects with the names of bases or default scaling factors and
        their powers.

        **Example**: 
        :code:`{"diffusivity": (("length", 2), ("time", -1))}`.

    Returns
    -------
    scales : `dict`
        Scaling factors.
    """

    if bases["mass"].__module__ == "pint":
        avogadro_constant = ureg.avogadro_constant
        boltzmann_constant = ureg.boltzmann_constant 
        bases["molar_energy"] = bases["energy"] * avogadro_constant
        bases["time"] = np.sqrt(
            bases["mass"] * bases["length"] ** 2 / bases["molar_energy"]
        ).to(ureg.picosecond)
        bases["charge"] = np.sqrt(
            4 * np.pi * ureg.vacuum_permittivity * bases["length"] * bases["energy"]
        ).to(ureg.elementary_charge)
    else:
        avogadro_constant = unit.AVOGADRO_CONSTANT_NA
        boltzmann_constant = unit.BOLTZMANN_CONSTANT_kB
        bases["molar_energy"] = bases["energy"] * avogadro_constant
        bases["time"] = (
            bases["mass"] * bases["length"] ** 2 / bases["molar_energy"]
        ).sqrt().in_units_of(unit.picosecond)
        bases["charge"] = (
            4 * np.pi * VACUUM_PERMITTIVITY * bases["length"] * bases["energy"]
        ).sqrt().in_units_of(unit.elementary_charge)
        
    # Define the default scaling factors
    bases["velocity"] = bases["length"] / bases["time"]
    bases["force"] = bases["molar_energy"] / bases["length"]
    bases["temperature"] = bases["energy"] / boltzmann_constant
    bases["pressure"] = bases["energy"] / bases["length"] ** 3
    bases["dynamic_viscosity"] = bases["pressure"] * bases["time"]
    bases["dipole"] = bases["length"] * bases["charge"]
    bases["electric_field"] = bases["force"] / bases["charge"]
    bases["mass_density"] = bases["mass"] / (bases["length"] ** 3 
                                             * avogadro_constant)

    return unit_scaling(bases, other)