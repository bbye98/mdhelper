"""
OpenMM physical constants and unit conversions
==============================================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains physical constants and helper functions for unit
reduction.
"""

from openmm import unit

from ..algorithm import unit as u

VACUUM_PERMITTIVITY = 8.854187812813e-12 * unit.farad / unit.meter

def get_scaling_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

    r"""
    Computes scaling factors for reduced units.

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        **Format**: :code:`{"mass": <openmm.unit.Quantity>,
        "length": <openmm.unit.Quantity>,
        "energy": <openmm.unit.Quantity>}`.

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

    return u.get_scaling_factors(bases, other)

def get_lj_scaling_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

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

        **Format**: :code:`{"mass": <openmm.unit.Quantity>,
        "length": <openmm.unit.Quantity>,
        "energy": <openmm.unit.Quantity>}`.

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

    return u.get_lj_scaling_factors(bases, other)