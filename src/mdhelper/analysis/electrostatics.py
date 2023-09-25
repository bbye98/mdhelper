"""
Electrostatics
==============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to quantify electrostatic properties, such
as the instantaneous dipole moment and the relative permittivity.
"""

from typing import Union

import MDAnalysis as mda
import numpy as np
from openmm import unit

from .base import SerialAnalysisBase
from .. import ArrayLike
from ..openmm import unit as u

def relative_permittivity(
        M: ArrayLike, temp: Union[float, unit.Quantity], 
        volume: Union[float, unit.Quantity], *, reduced: bool = False
    ) -> float:

    r"""
    Computes the relative permittivity (or static dielectric constant)
    :math:`\varepsilon_\mathrm{r}` of a medium using the instantaneous
    dipole moments :math:`\mathbf{M}(t)`.

    The dipole moment fluctuation formula [1]_ relates the relative 
    permittivity to the dipole moment via

    .. math::

       \varepsilon_\mathrm{r}=1+\frac{\langle M^2\rangle
       -\langle M\rangle^2}{3\varepsilon_0 Vk_\mathrm{B}T}

    where the angular brackets :math:`\langle\,\cdot\,\rangle` denote 
    the ensemble average, :math:`\varepsilon_0` is the vacuum 
    permittivity, :math:`k_\mathrm{B}` is the Boltzmann constant, and
    :math:`T` is the system temperature.

    .. note::

       If residues (molecules) in your system have net charges, the
       dipole moments must be made position-independent by subtracting
       the product of the net charge and the center of mass or geometry.

    Parameters
    ----------
    M : array-like
        Instantaneous dipole moments over :math:`N_t` frames.

        **Shape**: :math:`(N_t, 3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    temp : `float` or `openmm.unit.Quantity`
        System temperature :math:`T`.

        .. note::

           If :code:`reduced=True`, `temp` should be equal to the energy
           scale. When the Lennard-Jones potential is used, it generally
           means that :math:`T^* = 1`, or `temp=1`.

        **Reference unit**: :math:`\mathrm{K}`.

    volume : `float` or `openmm.unit.Quantity`
        System volume :math:`V`.

        **Reference unit**: :math:`\mathrm{Å^3}`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Affects `temp` 
        and `volume`.

    Returns
    -------
    vareps_r : `float`
        Relative permittivity (or static dielectric constant).

    References
    ----------
    .. [1] Neumann, M. Dipole Moment Fluctuation Formulas in Computer
       Simulations of Polar Systems. *Molecular Physics* **1983**, 
       *50* (4), 841–858. https://doi.org/10.1080/00268978300102721.
    """
    
    if not isinstance(M, unit.Quantity):
        M *= unit.elementary_charge * unit.angstrom
    elif reduced:
        emsg = ("'M' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    if not isinstance(temp, unit.Quantity):
        temp *= unit.kelvin
    elif reduced:
        emsg = ("'temp' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    if not isinstance(volume, unit.Quantity):
        volume *= unit.angstrom ** 3
    elif reduced:
        emsg = ("'volume' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    
    if reduced:
        return 1 + 4 * np.pi * ((M ** 2).mean(axis=0) - M.mean(axis=0) ** 2) \
               / (3 * volume.mean() * temp)
    else:
        return 1 + ((M ** 2).mean(axis=0) - M.mean(axis=0) ** 2) \
           / (3 * u.VACUUM_PERMITTIVITY * volume.mean() * unit.BOLTZMANN_CONSTANT_kB 
              * temp)

class DipoleMoment(SerialAnalysisBase):

    """
    A serial implementation to calculate the instantaneous dipole moment
    vectors :math:`\mathbf{M}`.

    For a system with :math:`N` atoms or molecules, the dipole moment is
    given by

    .. math::

       \mathbf{M}=\sum_i^{N}q_i\mathbf{z}_i

    where :math:`q_i` and :math:`\mathbf{z}_i` are the charge and
    position of entity :math:`i`.

    The dipole moment can be used to estimate the relative permittivity
    (or static dielectric constant) via the dipole moment fluctuation
    formula [1]_:

    .. math::

       \varepsilon_\mathrm{r}=1+\frac{\langle M^2\rangle
       -\langle M\rangle^2}{3\varepsilon_0 Vk_\mathrm{B}T}

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms for which the dipole moments are calculated.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines if atom positions are unwrapped. Ensure that 
        :code:`unwrap=False` when the trajectory already contains
        unwrapped particle positions, as this parameter is used in
        conjunction with `com_wrap` to determine the appropriate
        system center of mass.

    charges : array-like, keyword-only, optional
        Charge information for the atoms in the :math:`N_\mathrm{g}` 
        groups in `groups`. If not provided, it should be available in 
        and will be retrieved from the main 
        :class:`MDAnalysis.core.universe.Universe` object.

        **Shape**: :math:`(N_\mathrm{g},)` array of :math:`(N_i,)` 
        arrays, where :math:`N_i` is the number of atoms in group 
        :math:`i`.

        **Reference unit**: :math:`\mathrm{e}`.

    dims : array-like, keyword-only, optional
        Raw system dimensions. If the 
        :class:`MDAnalysis.core.universe.Universe` object that the 
        groups in `groups` belong to does not contain dimensionality 
        information, provide it here.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    Attributes
    ----------


    References
    ----------
    .. [1] Neumann, M. Dipole Moment Fluctuation Formulas in Computer
       Simulations of Polar Systems. *Molecular Physics* **1983**, 
       *50* (4), 841–858. https://doi.org/10.1080/00268978300102721.
    """
    
    def __init__(
            self, groups: Union[mda.AtomGroup, ArrayLike], unwrap: bool = False,
            dims: ArrayLike = None, charges: ArrayLike = None, 
            verbose: bool = True, **kwargs):
        
        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        self._dims = self.universe.dimensions[:3].copy()
        if self._dims is None:
            if dims is None:
                emsg = "Trajectory does not contain system dimensions."
                raise ValueError(emsg)
            self._dims = dims
            if isinstance(dims, unit.Quantity):
                self._dims = self._dims.value_in_unit(unit.nanometer)
        self.results.units = {"_dims": unit.angstrom}

        if charges:
            if len(charges) != len(self._groups):
                emsg = ("The dimension of 'charges' is incompatible "
                        "with the number of groups.")
                raise ValueError(emsg)
            if not hasattr(self.universe.atoms, "charges"):
                self.universe.atoms.add_TopologyAttr("charges")
            self._charges = charges
            for g, q in zip(self._groups, self._charges):
                g.charges = q
        else:
            if not hasattr(self.universe.atoms, "charges"):
                raise ValueError("The topology has no charge information.")
            self._charges = [g.charges for g in self._groups]
        self._electroneutral = [np.isclose(g.total_charge(), 0) 
                                for g in self._groups]
        self.results.units["_charges"] = unit.elementary_charge

        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        if self._unwrap:
            self._trajectory[self.start]
            self._positions_prev = [g.positions for g in self._groups]
            self._images = [np.zeros((g.n_atoms, 3), dtype=int) for g in self._groups]
            self._threshold = self._dims / 2

        self.results.dipole = np.empty((self.n_frames, 3), dtype=float)
        self.results.units["dipole"] = unit.elementary_charge * unit.angstrom
        self.results.volume = np.empty(self.n_frames, dtype=float)
        self.results.units["volume"] = unit.angstrom ** 3

    def _single_frame(self) -> None:

        self.results.dipole[self._frame_index] = np.dot(self._charges, 
                                                        self._group.positions)
        self.results.volume[self._frame_index] = self.universe.trajectory.ts.volume
        # dipole_vector
        
    def calculate_relative_permittivity(
            self, temp: Union[float, unit.Quantity]) -> None:
        
        """
        Computes the relative permittivity (or static dielectric 
        constant) :math:`\varepsilon_\mathrm{r}` of a medium using the
        instantaneous dipole moments :math:`\mathbf{M}`.
        """
        
        self.results.dielectric = relative_permittivity(
            self.results.dipole, temp, self.results.volume.mean()
        )