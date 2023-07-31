"""
Custom OpenMM bond potentials
=============================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains implementations of commonly used bond potentials
that are not available in OpenMM, such as the finite extension nonlinear
elastic (FENE) potential. Generally, the bond potentials are named after
their LAMMPS :code:`bond_style` counterparts, if available.
"""

from typing import Any, Iterable, Union

import openmm
from openmm import unit

from .pair import ljts

def _setup_bond(
        cbforce: openmm.CustomBondForce,
        global_params: dict[str, Union[float, unit.Quantity]],
        per_params: Iterable[Any]) -> None:

    for param in global_params.items():
        cbforce.addGlobalParameter(*param)
    for param in per_params:
        cbforce.addPerBondParameter(param)

def fene(
        globals: dict[str, Union[float, unit.Quantity]] = {},
        pers: Iterable[str] = ["k", "r0"], wca: bool = True, **kwargs
    ) -> tuple[openmm.CustomBondForce, openmm.CustomNonbondedForce]:

    r"""
    Implements the finite extensible nonlinear elastic (FENE) potential
    used for bead-spring polymer models:

    .. math::

       u_\textrm{FENE}=-\frac{1}{2}k_{12}r_{0,12}^2
       \ln{\left[1-\left(\frac{r_{12}}{r_{0,12}}\right)^2\right]}
       +4\epsilon_{12}\left[\left(\frac{\sigma_{12}}{r_{12}}\right)^{12}
       -\left(\frac{\sigma_{12}}{r_{12}}\right)^6\right]+\epsilon_{12}

    where :math:`k_{12}` is the bond coefficient in
    :math:`\textrm{kJ}/(\textrm{nm}^2\cdot\textrm{mol})`,
    :math:`r_{0,12}` is the equilibrium bond length in
    :math:`\textrm{nm}`, :math:`\sigma_{12}` is the size of the particle
    in :math:`\textrm{nm}`, and :math:`\epsilon_{12}` is the dispersion
    energy in :math:`\textrm{kJ/mol}`. :math:`k_{12}`, :math:`r_{0,12}`,
    :math:`\sigma_{12}` and :math:`\epsilon_{12}` are  determined from 
    per-particle parameters `k`, `r0`, `sigma` and `epsilon`, 
    respectively, which are set in the main script using
    :meth:`openmm.openmm.CustomBondForce.addBond` and
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    Parameters
    ----------
    globals : `dict`, optional
        Additional global parameters for use in the definition of
        :math:`k_{12}` and :math:`r_{0,12}`.

    pers : `array_like`, optional
        Additional per-particle parameters for use in the definition of
        :math:`k_{12}` and :math:`r_{0,12}`.

    wca : `bool`, default: `True`
        Determines whether the Weeks–Chander–Andersen (WCA) potential is
        included.

    **kwargs
        Keyword arguments to be passed to 
        :meth:`mdhelper.openmm.pair.ljts` if :code:`wca=True`.
    
    Returns
    -------
    bond_fene : `openmm.CustomBondForce`
        FENE bond potential.
    
    pair_wca : `openmm.CustomNonbondedForce`
        WCA pair potential, if :code:`wca=True`.
    """

    bond_fene = openmm.CustomBondForce("-0.5*k*r0^2*log(1-(r/r0)^2)")
    _setup_bond(bond_fene, globals, pers)

    if wca:
        pair_wca = ljts(wca=wca, **kwargs)
        return bond_fene, pair_wca

    return bond_fene