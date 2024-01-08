"""
Electrostatics
==============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to quantify electrostatic properties, such
as the instantaneous dipole moment.
"""

from typing import Union

import MDAnalysis as mda
import numpy as np

from .base import SerialAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg

if FOUND_OPENMM:
    from openmm import unit

def relative_permittivity(
        M: Union[np.ndarray[float], "unit.Quantity", Q_],
        temp: Union[float, "unit.Quantity", Q_], 
        volume: Union[float, "unit.Quantity", Q_], *, reduced: bool = False
    ) -> float:

    r"""
    Computes the relative permittivity (or static dielectric constant)
    :math:`\varepsilon_\mathrm{r}` of a medium using the instantaneous
    dipole moments :math:`\mathbf{M}(t)`.

    The dipole moment fluctuation formula [1]_ relates the relative 
    permittivity to the dipole moment via

    .. math::

       \varepsilon_\mathrm{r}=1+\frac{\langle |\mathbf{M}|^2\rangle
       -|\langle\mathbf{M}\rangle |^2}{3\varepsilon_0 Vk_\mathrm{B}T}

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
    M : array-like, `openmm.unit.Quantity`, or `pint.Quantity`
        Instantaneous dipole moments over :math:`N_t` frames.

        **Shape**: :math:`(N_t, 3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    temp : `float`, `openmm.unit.Quantity`, or `pint.Quantity`
        System temperature :math:`T`.

        .. note::

           If :code:`reduced=True`, `temp` should be equal to the energy
           scale. When the Lennard-Jones potential is used, it generally
           means that :math:`T^* = 1`, or `temp=1`.

        **Reference unit**: :math:`\mathrm{K}`.

    volume : `float`, `openmm.unit.Quantity`, or `pint.Quantity`
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
    
    if isinstance(M, (int, float, np.ndarray)):
        M *= ureg.elementary_charge * ureg.angstrom
    elif reduced:
        emsg = ("'M' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    elif M.__module__ == "openmm.unit.quantity":
        M = (M.value_in_unit(unit.elementary_charge * unit.angstrom) 
             * ureg.elementary_charge * ureg.angstrom)

    if isinstance(temp, (int, float)):
        temp *= ureg.kelvin
    elif reduced:
        emsg = ("'temp' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    elif temp.__module__ == "openmm.unit.quantity":
        temp = temp.value_in_unit(unit.kelvin) * ureg.kelvin
    
    if isinstance(volume, (int, float, np.ndarray)):
        volume *= ureg.angstrom ** 3
    elif reduced:
        emsg = ("'volume' has units, but the rest of the data is "
                "or should be reduced.")
        raise ValueError(emsg)
    elif volume.__module__ == "openmm.unit.quantity":
        volume = (volume.value_in_unit(unit.angstrom ** 3) 
                  * ureg.angstrom ** 3)
    
    if reduced:
        return 1 + 4 * np.pi * ((M ** 2).mean(axis=0) - M.mean(axis=0) ** 2).mean() \
               / (volume.mean() * temp)
    else:
        return (1 + ((M ** 2).mean(axis=0) - M.mean(axis=0) ** 2).mean()
                / (ureg.vacuum_permittivity * volume.mean() * ureg.boltzmann_constant 
                   * temp)).magnitude

class DipoleMoment(SerialAnalysisBase):

    r"""
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

       \varepsilon_\mathrm{r}=1+\frac{\langle |\mathbf{M}|^2\rangle
       -|\langle\mathbf{M}\rangle |^2}{3\varepsilon_0 Vk_\mathrm{B}T}

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms for which the dipole moments are calculated.

    charges : array-like, keyword-only, optional
        Charge information for the atoms in the :math:`N_\mathrm{g}` 
        groups in `groups`. If not provided, it should be available in 
        and will be retrieved from the main 
        :class:`MDAnalysis.core.universe.Universe` object.

        **Shape**: :math:`(N_\mathrm{g},)` array of :math:`(N_i,)` 
        arrays, where :math:`N_i` is the number of atoms in group 
        :math:`i`.

        **Reference unit**: :math:`\mathrm{e}`.

    dimensions : array-like, keyword-only, optional
        System dimensions. If the 
        :class:`MDAnalysis.core.universe.Universe` object that the 
        groups in `groups` belong to does not contain dimensionality 
        information, provide it here.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    average : `bool`, keyword-only, default: :code:`False`
        Determines if the dipole moment vectors and volumes are 
        time-averaged.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines if atom positions are unwrapped.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines if progress is printed to the console.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.

    Attributes
    ----------
    universe : `MDAnalysis.Universe`
        :class:`MDAnalysis.core.universe.Universe` object containing all
        information describing the system.

    results.units : `dict`
        Reference units for the results. For example, to get the 
        reference units for :code:`results.time`, call 
        :code:`results.units["results.time"]`.

    results.time : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\mathrm{ps}`.

    results.dipole : `numpy.ndarray`
        Instantaneous dipole moment vectors :math:`\mathbf{M}`.

        **Shape**: :math:`(N_t, 3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    results.volume : `numpy.ndarray`
        System volumes :math:`V`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\mathrm{Å^3}`.

    results.dielectric : `numpy.ndarray`
        Relative permittivity (or static dielectric constant)
        :math:`\varepsilon_\mathrm{r}` in each dimension.

        **Shape**: :math:`(3,)`.
    
    References
    ----------
    .. [1] Neumann, M. Dipole Moment Fluctuation Formulas in Computer
       Simulations of Polar Systems. *Molecular Physics* **1983**, 
       *50* (4), 841–858. https://doi.org/10.1080/00268978300102721.
    """
    
    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]], 
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            average: bool = False, unwrap: bool = False, verbose: bool = True,
            **kwargs) -> None:
        
        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self._n_groups = len(self._groups)
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        self.results.units = {"_charges": ureg.elementary_charge,
                              "_dimensions": ureg.angstrom}

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            if not isinstance(dimensions, (list, tuple, np.ndarray)):
                if dimensions.__module__ == "openmm.unit.quantity":
                    dimensions = dimensions.value_in_unit(unit.angstrom)
                else:
                    dimensions = dimensions.m_as(
                        self.results.units["_dimensions"]
                    )
            self._dimensions = np.asarray(dimensions)
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        if charges is not None:
            charges = list(charges)
            if len(charges) == self._n_groups:
                for i, (g, q) in enumerate(zip(self._groups, charges)):
                    if not isinstance(q, (list, tuple, np.ndarray)):
                        if q.__module__ == "openmm.unit.quantity":
                            q = q.value_in_unit(unit.elementary_charge)
                        else:
                            q = q.m_as(self.results.units["_charges"])
                    if isinstance(q, (int, float)):
                        q *= np.ones(g.n_atoms, dtype=float)
                    elif g.n_atoms != len(q):
                        emsg = (f"The number of charges in charges[{i}] "
                                "is not equal to the number of atoms in "
                                "the corresponding group.")
                        raise ValueError(emsg)
                    charges[i] = q
                self._charges = charges
            else:
                emsg = ("The number of group charge arrays is not "
                        "equal to the number of groups.")
                raise ValueError(emsg)
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = [g.charges for g in self._groups]
        else:
            raise ValueError("The topology has no charge information.")
        self._electroneutral = sum(np.sum(q) for q in self._charges)
        self._electroneutral_groups = [np.isclose(np.sum(q), 0) 
                                       for q in self._charges]

        self._average = average
        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to number of boundary crossings for each 
        # atom
        if self._unwrap:
            self._trajectory[self.start]
            self._positions_old = self.universe.atoms.positions
            self._images = np.zeros((self.universe.atoms.n_atoms, 3), dtype=int)
            self._threshold = self._dimensions / 2

        # Preallocate arrays to store results and store reference units
        if not self._average:
            self.results.time = (self.step * self._trajectory.dt 
                                 * np.arange(self.n_frames))
            self.results.units["time"] = ureg.picosecond
        self.results.dipole = np.zeros((self.n_frames, 3), dtype=float)
        self.results.units["dipole"] = ureg.elementary_charge * ureg.angstrom
        self.results.volume = np.empty(self.n_frames, dtype=float)
        self.results.units["volume"] = ureg.angstrom ** 3

    def _single_frame(self) -> None:

        # Unwrap all particle positions, if necessary
        positions = self.universe.atoms.positions.copy()
        if self._unwrap:
            dpos = positions - self._positions_old
            mask = np.abs(dpos) >= self._threshold
            self._images[mask] -= np.sign(dpos[mask]).astype(int)
            self._positions_old = self.universe.atoms.positions.copy()
            positions += self._images * self._dimensions

        # Compute dipole moment vectors and store per-frame volume
        for i, (g, c) in enumerate(zip(self._groups, self._charges)):
            self.results.dipole[self._frame_index] += np.dot(c, positions[g.indices])
        self.results.volume[self._frame_index] = self.universe.trajectory.ts.volume

    def _conclude(self) -> None:
        
        # Average results, if requested
        if self._average:
            self.results.dipole = self.results.dipole.mean(axis=0)
            self.results.volume = self.results.volume.mean()
        
    def calculate_relative_permittivity(
            self, temp: Union[float, unit.Quantity]) -> None:
        
        r"""
        Computes the relative permittivity (or static dielectric 
        constant) :math:`\varepsilon_\mathrm{r}` of a medium using the
        instantaneous dipole moments :math:`\mathbf{M}(t)`.

        Parameters
        ----------
        temp : `float`, `openmm.unit.Quantity`, or `pint.Quantity`
            System temperature :math:`T`.

            .. note::

               If :code:`reduced=True`, `temp` should be equal to the 
               energy scale. When the Lennard-Jones potential is used, 
               it generally means that :math:`T^* = 1`, or `temp=1`.

            **Reference unit**: :math:`\mathrm{K}`.
        """

        if self._average:
            emsg = ("Cannot compute relative permittivity using the"
                    "averaged dipole moment.")
            raise RuntimeError(emsg)
        
        self.results.dielectric = relative_permittivity(
            self.results.dipole, temp, self.results.volume.mean()
        )