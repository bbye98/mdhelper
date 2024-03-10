"""
Utility algorithms
==================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains logical and mathematical utility functions used by
other MDHelper modules.
"""

from numbers import Number
from typing import Union

import numpy as np
import sympy

from .. import FOUND_OPENMM, Q_, ureg

if FOUND_OPENMM:
    from openmm import unit
    from ..openmm.unit import VACUUM_PERMITTIVITY

def get_closest_factors(
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
        factors = np.array(sympy.divisors(x.shape[-1])[1:])
        factor_indices = np.where(factors < 10)[0]
        if len(factor_indices):
            factor = factors[factor_indices[-1]]
        else:
            raise ValueError("No factor provided for rebinning.")

    return x.reshape((*x.shape[:-1], -1, factor)).mean(axis=-1)

def get_scaling_factors(
        bases: dict[str, Union["unit.Quantity", Q_]],
        other: dict[str, list] = {}) -> dict[str, Union["unit.Quantity", Q_]]:

    r"""
    Evaluates scaling factors for reduced units.

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        .. container::

           **Format**:

           .. code::

              {
                "mass": <openmm.unit.Quantity> | <pint.Quantity>,
                "length": <openmm.unit.Quantity> | <pint.Quantity>,
                "energy": <openmm.unit.Quantity> | <pint.Quantity>
              }

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ}`.

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

    for name, params in other.items():
        factor = 1
        for base, power in params:
            factor *= bases[base] ** power
        bases[name] = factor

    return bases

def get_lj_scaling_factors(
        bases: dict[str, Union["unit.Quantity", Q_]],
        other: dict[str, list] = {}) -> dict[str, Union["unit.Quantity", Q_]]:

    r"""
    Evaluates scaling factors for Lennard-Jones reduced units.

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

        .. container::

           **Format**:

           .. code::

              {
                "mass": <openmm.unit.Quantity> | <pint.Quantity>,
                "length": <openmm.unit.Quantity> | <pint.Quantity>,
                "energy": <openmm.unit.Quantity> | <pint.Quantity>
              }

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ}`.

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

    return get_scaling_factors(bases, other)

def strip_unit(
        value: Union[Number, "unit.Quantity", Q_],
        unit_: Union[str, "unit.Unit", ureg.Unit] = None
    ) -> tuple[Number, Union[None, "unit.Unit", ureg.Unit]]:

    """
    Strips the unit from an :obj:`openmm.unit.Quantity` or
    :obj:`pint.Quantity` object.

    Parameters
    ----------
    value : `numbers.Number`, `openmm.unit.Quantity`, or `pint.Quantity`
        Physical quantity for which to get the magnitude of in the
        unit specified in `unit_`.

    unit_ : `str`, `openmm.unit.Unit`, or `pint.Unit`, optional
        Unit to convert to. If not specified, the original unit is used.

    Returns
    -------
    value : `numbers.Number`
        Magnitude of the physical quantity in the specified unit.

    unit_ : `openmm.unit.Unit` or `pint.Unit`
        Unit of the physical quantity.

    Examples
    --------
    For any quantity other than a :obj:`openmm.unit.Quantity` or
    :obj:`pint.Quantity` object, the raw quantity and user-specified
    unit are returned.

    >>> strip_unit(90.0, "deg")
    (90.0, 'deg')
    >>> strip_unit(90.0, ureg.degree)
    (90.0, <Unit('degree')>)

    If no target unit is specified, the magnitude and original unit of
    the quantity are returned.

    >>> strip_unit(1.380649e-23 * ureg.joule * ureg.kelvin ** -1)
    (1.380649e-23, <Unit('joule / kelvin')>)

    If a target unit using the same module as the quantity is specified,
    the quantity is first converted to the target unit, if necessary,
    before its magnitude and unit are returned.

    >>> g = 9.80665 * ureg.meter / ureg.second ** 2
    >>> strip_unit(g, "meter/second**2")
    (9.80665, <Unit('meter / second ** 2')>)
    >>> strip_unit(g, ureg.foot / ureg.second ** 2)
    (32.17404855643044, <Unit('foot / second ** 2')>)

    If a target unit using a different module than the quantity is
    specified, the quantity is converted to the specified target unit in
    the new module, if necessary, before its magnitude and unit are
    returned.

    >>> strip_unit(8.205736608095969e-05 * unit.meter ** 3 * unit.atmosphere
    ...            / (unit.kelvin * unit.mole),
    ...            ureg.joule / (ureg.kelvin * ureg.mole))
    (8.31446261815324, <Unit('joule / kelvin / mole')>)
    >>> strip_unit(8.205736608095969e-05 * ureg.meter ** 3 * ureg.atmosphere
    ...            / (ureg.kelvin * ureg.mole),
    ...            unit.joule / (unit.kelvin * unit.mole))
    (8.31446261815324, Unit({BaseUnit(..., name="kelvin", ...): -1.0, ...}))
    """

    if isinstance(value, Q_):

        # No target unit (unit_) specified; return Pint magnitude and
        # unit (unit__)
        if unit_ is None:
            unit__ = value.units
            value = value.magnitude
        else:

            # Convert OpenMM target unit (unit__ = unit_) to Pint unit
            # (unit_) and return unit__
            if getattr(unit_, "__module__", None) == "openmm.unit.unit":
                unit_, unit__ = ureg.Unit(""), unit_
                for u, p in unit__.iter_base_or_scaled_units():
                    unit_ *= ureg.Unit(u.name.replace(" ", "_")) ** p

            # Convert str or Pint target unit (unit_) to Pint unit
            # (unit__)
            else:
                unit__ = ureg.Unit(unit_)

            # Get magnitude of Pint quantity in Pint unit (unit_)
            value = value.m_as(unit_)

    elif getattr(value, "__module__", None) == "openmm.unit.quantity":
        swap = False

        # No target unit (unit_) specified; return OpenMM magnitude and
        # unit (unit__)
        if unit_ is None:
            unit_ = unit__ = value.unit

        # Store OpenMM target unit (unit_) in return value (unit__)
        elif getattr(unit_, "__module__", None) == "openmm.unit.unit":
            unit__ = unit_
        else:

            # Determine whether unit_ and unit__ need to be swapped at
            # the end; str target unit should give OpenMM unit, while
            # Pint target unit should give Pint unit
            swap = not isinstance(unit_, str)

            # Convert str or Pint target unit (unit_) to OpenMM unit
            # (unit__)
            unit_ = ureg.Unit(unit_)
            try:
                unit__ = unit.dimensionless
                for u, p in unit_._units.items():
                    unit__ *= getattr(unit, u) ** p
            except AttributeError:
                emsg = ("strip_unit() relies on the pint module for "
                        "parsing units. At least one unit in 'unit' is "
                        "not defined the same way in openmm.unit and pint, "
                        "so the unit conversion cannot be performed. Try "
                        "specifying a openmm.unit.Quantity object for "
                        "'unit' instead.")
                raise ValueError(emsg)
        value = value.value_in_unit(unit__)
        if swap:
            unit_, unit__ = unit__, unit_
    else:
        unit__ = unit_
    return value, unit__