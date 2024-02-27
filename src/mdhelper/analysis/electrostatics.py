"""
Electrostatics
==============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to quantify electrostatic properties, such
as the instantaneous dipole moment.
"""

from numbers import Real
from typing import Union

import MDAnalysis as mda
import numpy as np

from .base import SerialAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm.topology import unwrap
from ..algorithm.utility import strip_unit

if FOUND_OPENMM:
    from openmm import unit

def calculate_relative_permittivity(
        M: np.ndarray[float], temperature: float, volume: float, *,
        reduced: bool = False) -> float:

    r"""
    Calculates the relative permittivity (or static dielectric constant)
    :math:`\varepsilon_\mathrm{r}` of a medium using the instantaneous
    dipole moments :math:`\mathbf{M}(t)`.

    The dipole moment fluctuation formula [1]_ relates the relative
    permittivity to the dipole moment via

    .. math::

       \varepsilon_\mathrm{r}=1+\frac{\overline{\langle\mathbf{M}^2\rangle
       -\langle\mathbf{M}\rangle^2}}{3\varepsilon_0 Vk_\mathrm{B}T}

    where the angular brackets :math:`\langle\,\cdot\,\rangle` denote
    the ensemble average, the overline signifies the spatial average,
    :math:`\varepsilon_0` is the vacuum permittivity,
    :math:`k_\mathrm{B}` is the Boltzmann constant, and :math:`T` is
    the system temperature.

    .. note::

       If residues (molecules) in your system have net charges, the
       dipole moments must be made position-independent by subtracting
       the product of the net charge and the center of mass or geometry.

    Parameters
    ----------
    M : array-like
        Instantaneous dipole moments over :math:`N_t` frames.

        **Shape**: :math:`(N_t,\,3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    temperature : `float`
        System temperature :math:`T`.

        .. note::

           If :code:`reduced=True`, `temperature` should be equal to the
           energy scale. When the Lennard-Jones potential is used, it
           generally means that :math:`T^* = 1`, or `temperature=1`.

        **Reference unit**: :math:`\mathrm{K}`.

    volume : `float`
        System volume :math:`V`.

        **Reference unit**: :math:`\mathrm{Å^3}`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

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

    if reduced:
        return (1 + 4 * np.pi * (M ** 2 - M.mean(axis=0) ** 2).mean()
                    / (volume.mean() * temperature))
    else:
        M *= ureg.elementary_charge * ureg.angstrom
        temperature *= ureg.kelvin
        volume *= ureg.angstrom ** 3
        return (1 + (M ** 2 - M.mean(axis=0) ** 2).mean()
                / (ureg.vacuum_permittivity * volume.mean()
                   * ureg.boltzmann_constant * temperature)).magnitude

class DipoleMoment(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the instantaneous dipole moment
    vectors :math:`\mathbf{M}(t)`.

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

       \varepsilon_\mathrm{r}=1+\frac{\overline{\langle\mathbf{M}^2\rangle
       -\langle\mathbf{M}\rangle^2}}{3\varepsilon_0 Vk_\mathrm{B}T}

    where the angular brackets :math:`\langle\,\cdot\,\rangle` denote
    the ensemble average, the overline signifies the spatial average,
    :math:`\varepsilon_0` is the vacuum permittivity,
    :math:`k_\mathrm{B}` is the Boltzmann constant, and :math:`T` is
    the system temperature.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms for which the dipole moments are calculated.

    charges : array-like, keyword-only, optional
        Charge information for the atoms in the :math:`N_\mathrm{g}`
        groups in `groups`. If not provided, it should be available in
        and will be retrieved from the main
        :class:`MDAnalysis.core.universe.Universe` object.

        **Shape**: :math:`(N_\mathrm{g},)` array of `int`s, `float`s, or
        :math:`(N_i,)` arrays, where :math:`N_i` is the number of atoms
        in group :math:`i`.

        **Reference unit**: :math:`\mathrm{e}`.

    dimensions : array-like, keyword-only, optional
        System dimensions. If the
        :class:`MDAnalysis.core.universe.Universe` object that the
        groups in `groups` belong to does not contain dimensionality
        information, provide it here. Affected by `scales`.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    scales : array-like, keyword-only, optional
        Scaling factors for each system dimension. If an `int` is
        provided, the same value is used for all axes.

        **Shape**: :math:`(3,)`.

    average : `bool`, keyword-only, default: :code:`False`
        Determines whether the dipole moment vectors and volumes are
        time-averaged.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Only affects
        :meth:`calculate_relative_permittivity` calls.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines whether atom positions are unwrapped.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether progress is printed to the console.

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
        reference units for :code:`results.times`, call
        :code:`results.units["results.times"]`.

    results.times : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\mathrm{ps}`.

    results.dipoles : `numpy.ndarray`
        Instantaneous dipole moment vectors :math:`\mathbf{M}`.

        **Shape**: :math:`(N_t, 3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    results.volumes : `numpy.ndarray`
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
            scales: Union[float, tuple[float]] = 1, average: bool = False,
            reduced: bool = False, unwrap: bool = False, verbose: bool = True,
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
            self._dimensions = np.asarray(strip_unit(dimensions, "angstrom")[0])
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        if isinstance(scales, Real) or (len(scales) == 3
                                        and isinstance(scales[0], Real)):
            self._dimensions *= scales
        else:
            emsg = ("The scaling factor(s) must be provided as a "
                    "floating-point number or in an array with shape (3,). ")
            raise ValueError(emsg)

        if charges is not None:
            charges = list(charges)
            if len(charges) == self._n_groups:
                for i, (g, q) in enumerate(zip(self._groups, charges)):
                    charges[i] = strip_unit(q, "elementary_charge")[0]
                    if isinstance(charges[i], Real):
                        charges[i] *= np.ones(g.n_atoms)
                    elif g.n_atoms != len(charges[i]):
                        emsg = ("The number of charges in "
                                f"'charges[{i}]' is not equal to the "
                                "number of atoms in the corresponding "
                                "group.")
                        raise ValueError(emsg)
                self._charges = charges
            else:
                emsg = ("The number of group charge arrays is not "
                        "equal to the number of groups.")
                raise ValueError(emsg)
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = [g.charges for g in self._groups]
        else:
            raise ValueError("The topology has no charge information.")

        # TODO: Add support for charged molecules.
        self._all_neutral = np.allclose(
            self.universe.atoms.total_charge(compound="fragments"), 0,
            atol=1e-6
        )
        self._all_included = sum(g.n_atoms for g in self._groups) \
                             == self.universe.atoms.n_atoms

        self._average = average
        self._reduced = reduced
        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to number of boundary crossings for each
        # atom
        if self._unwrap:
            self._trajectory[self.start]
            self._positions_old = self.universe.atoms.positions
            self._images = np.zeros((self.universe.atoms.n_atoms, 3), dtype=int)
            self._thresholds = self._dimensions / 2

        # Preallocate arrays to store results and store reference units
        if not self._average:
            self.results.times = (self.step * self._trajectory.dt
                                  * np.arange(self.n_frames))
            self.results.units["times"] = ureg.picosecond
        self.results.dipoles = np.zeros((self.n_frames, 3))
        self.results.volumes = np.empty(self.n_frames)
        self.results.units["dipoles"] = ureg.elementary_charge * ureg.angstrom
        self.results.units["volumes"] = ureg.angstrom ** 3

    def _single_frame(self) -> None:

        # Store atom positions in the current frame
        positions = self.universe.atoms.positions.copy()

        # Unwrap all particle positions, if necessary
        if self._unwrap:
            unwrap(
                positions,
                self._positions_old,
                self._dimensions,
                thresholds=self._thresholds,
                images=self._images
            )

        # Compute dipole moment vectors and store per-frame volume
        for g, c in zip(self._groups, self._charges):
            self.results.dipoles[self._frame_index] += c @ positions[g.indices]
        self.results.volumes[self._frame_index] = self.universe.trajectory.ts.volume

    def _conclude(self) -> None:

        # Average results, if requested
        if self._average:
            self.results.dipoles = self.results.dipoles.mean(axis=0)
            self.results.volumes = self.results.volumes.mean()

    def calculate_relative_permittivity(
            self, temperature: Union[float, "unit.Quantity", Q_]) -> None:

        r"""
        Calculates the relative permittivity (or static dielectric
        constant) :math:`\varepsilon_\mathrm{r}` of a medium using the
        instantaneous dipole moments :math:`\mathbf{M}(t)`.

        Parameters
        ----------
        temperature : `float`, `openmm.unit.Quantity`, or `pint.Quantity`
            System temperature :math:`T`.

            .. note::

               If :code:`reduced=True` was set in the
               :class:`DipoleMoment` constructor, `temperature` should 
               be equal to the energy scale. When the Lennard-Jones 
               potential is used, it generally means that 
               :math:`T^* = 1`, or `temperature=1`.

            **Reference unit**: :math:`\mathrm{K}`.
        """

        if self._average:
            emsg = ("Cannot compute relative permittivity using the"
                    "averaged dipole moment.")
            raise RuntimeError(emsg)
        elif not self._all_neutral:
            emsg = ("Cannot compute relative permittivity for a "
                    "non-neutral system or a system with ions.")
            raise RuntimeError(emsg)
        elif not self._all_included:
            emsg = ("Cannot compute relative permittivity when not all"
                    "atoms in the system are accounted for in the "
                    "groups.")
            raise RuntimeError(emsg)
        else:
            temperature, unit_ = strip_unit(temperature, "kelvin")
            if self._reduced and not isinstance(unit_, str):
                emsg = ("'temperature' cannot have units when reduced=True.")
                raise ValueError(emsg)

            self.results.dielectric = calculate_relative_permittivity(
                self.results.dipoles, temperature, self.results.volumes.mean(),
                reduced=self._reduced
            )