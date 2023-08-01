"""
Linear profiles
===============
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains classes to quantify properties along axes.
"""

from typing import Any, Union
import warnings

import MDAnalysis as mda
import numpy as np
from openmm import unit
from scipy import integrate

from .base import SerialAnalysisBase
from .. import ArrayLike
from ..algorithm import molecule
from ..openmm import unit as u

def potential_profile(
        bins: Union[np.ndarray, unit.Quantity], 
        charge_density: Union[np.ndarray, unit.Quantity], 
        L: Union[float, unit.Quantity], dielectric: float = 1, *,
        sigma_e: Union[float, unit.Quantity] = None,
        dV: Union[float, unit.Quantity] = None,
        threshold: float = 1e-5, V0: Union[float, unit.Quantity] = 0,
        reduced: bool = False) -> None:
    
    r"""
    Calculates the potential profile :math:`\varphi(z)` using the charge
    density profile by numerically solving Poisson's equation for 
    electrostatics.

    The Poisson's equation is

    .. math::

       \varepsilon_0\varepsilon_\mathrm{r}\nabla^2\varphi(z)=-\rho_e(z)

    where :math:`\varepsilon_0` is the vacuum permittivity, 
    :math:`\varepsilon_\mathrm{r}` is the relative permittivity, 
    :math:`\rho_e` is the charge density, and :math:`\varphi` is the
    potential.

    Parameters
    ----------
    bins : `numpy.ndarray` or `openmm.unit.Quantity`
        Histogram bin centers corresponding to the charge density 
        profile in `charge_density`.
        
        **Shape**: :math:`(N_\mathrm{bins},)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    charge_density : `numpy.ndarray` or `openmm.unit.Quantity`
        Array containing the charge density profile.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{e/Å}^{-3}`.

    L : `float` or `openmm.unit.Quantity`
        System size in the dimension that `bins` and `charge_density`
        were calculated in.

        **Reference unit**: :math:`\mathrm{Å}`.

    dielectric : `float`, default: :code:`1`
        Relative permittivity or dielectric constant 
        :math:`\varepsilon_\mathrm{r}`.

    sigma_e : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Total surface charge density :math:`\sigma_e`. Used to 
        ensure that the electric field in the bulk of the solution
        is zero. If not provided, it is determined using `dV` and 
        the charge density profile, or the average value in the 
        center of the integrated charge density profile. 
        
        **Reference unit**: :math:`\mathrm{e/Å^2}`.

    dV : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Potential difference :math:`\Delta\varphi` across the system 
        dimension specified in `axis`. Has no effect if `sigma_e` is
        provided since this value is used solely to calculate 
        `sigma_e`.
            
        .. note::

           By specifying `dV` to calculate `sigma_e` using Gauss's law,
           it is assumed that the boundaries are perfectly conducting.

        **Reference unit**: :math:`\mathrm{V}`.
    
    threshold : `float`, keyword-only, default: :code:`1e-5`
        Threshold for determining the plateau region of the first
        integral of the charge density profile to calculate
        `sigma_e`. Has no effect if `sigma_e` is provided or if
        `sigma_e` can be calculated using `dV` and `dielectric`.

    V0 : `float` or `openmm.unit.Quantity`, keyword-only, default: :code:`0`
        Potential :math:`\varphi_0` at the left boundary. 
        
        **Reference unit**: :math:`\mathrm{V}`.
    """
        
    # Check V0 for unit consistency
    if isinstance(V0, unit.Quantity):
        if reduced:
            emsg = ("'V0' has units, while the rest of the data is "
                    "reduced.")
            raise ValueError(emsg)
        V0 /= unit.volt
        if isinstance(V0, unit.Quantity):
            raise ValueError("'V0' has invalid units.")
    
    # Calculate the first integral of the charge density profile
    potential = integrate.cumulative_trapezoid(charge_density, bins, initial=0)

    if sigma_e is None:
        
        # Calculate surface charge density for system with perfectly
        # conducting boundaries
        if dV is not None:

            # Check dV for unit consistency
            if isinstance(dV, unit.Quantity):
                if reduced:
                    emsg = ("'dV' has units, while the rest of the data is "
                            "reduced.")
                    raise ValueError(emsg)
                dV /= unit.volt
                if isinstance(V0, unit.Quantity):
                    raise ValueError("'dV' has invalid units.")
        
            sigma_e = dielectric * dV / L
            if reduced:
                sigma_e /= 4 * np.pi
            else:
                sigma_e *= (u.VACUUM_PERMITTIVITY * unit.volt * unit.angstrom 
                            / unit.elementary_charge)
            sigma_e -= integrate.trapezoid(bins * charge_density, bins) / L

        else:
            wmsg = ("No surface charge density information. The value will "
                    "be extracted from the integrated charge density "
                    "profile, which may be inaccurate due to numerical "
                    "errors.")
            warnings.warn(wmsg)

            # Get surface charge density from the integrated charge
            # density profile
            cut_indices = np.where(
                np.diff(np.abs(np.gradient(potential)) < threshold)
            )[0] + 1
            target_index = len(potential) // 2
            sigma_e = potential[
                cut_indices[cut_indices <= target_index][-1]:
                cut_indices[cut_indices >= target_index][0]
            ].mean()
    
    # Check sigma_e for unit consistency
    elif isinstance(sigma_e, unit.Quantity):
        if reduced:
            emsg = ("'sigma_e' has units, while the rest of the data "
                    "is reduced.")
            raise ValueError(emsg)
        sigma_e /= unit.elementary_charge / unit.angstrom ** 2
        if isinstance(sigma_e, unit.Quantity):
            raise ValueError("'sigma_e' has invalid units.")

    # Calculate the second integral of the charge density profile
    potential = -integrate.cumulative_trapezoid(potential - sigma_e, bins, 
                                                initial=V0) / dielectric
    if reduced:
        potential *= 4 * np.pi
    else:
        potential *= unit.elementary_charge \
                     / (u.VACUUM_PERMITTIVITY * unit.angstrom * unit.volt)
        
    return potential

class DensityProfile(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the number and charge density
    profiles :math:`\phi_i(z)` and :math:`\phi_e(z)` of a system
    along the specified axes.

    The microscopic number density profile of species :math:`i` in a
    canonical :math:`NVT` ensemble is calculated by binning particle
    positions along an axis :math:`z` using

    .. math::

       \rho_i(z)=\frac{V}{N_\mathrm{bin}}\left\langle
       \sum_\alpha\delta(z-z_\alpha)\right\rangle

    where :math:`V` is the system volume and :math:`N_\mathrm{bins}` is
    the number of bins. The angular brackets denote an ensemble average.

    If the species carry charges, the charge density profile can be
    obtained using

    .. math::

       \rho_e(z)=\sum_i z_ie\rho_i(z)

    where :math:`z_i` is the charge number of species :math:`i` and 
    :math:`e` is the elementary charge.

    With the charge density profile, the potential profile can be
    computed by numerically solving Poisson's equation for 
    electrostatics:

    .. math::
       
       \varepsilon_0\varepsilon_\mathrm{r}\nabla^2\varphi(z)=-\rho_e(z)

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of atoms for which density profiles are calculated.
    
    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally for 
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for 
             atomistic simulations).
           * :code:`"segments"`: Segments' centers of mass (for 
             atomistic polymer simulations).
    
    axes : `int`, `str`, or array-like, default: :code:`"xyz"`
        Axes along which to compute the density profiles.

        .. container::

           **Examples**:

           * :code:`2` for the :math:`z`-direction.
           * :code:`"xy"` for the :math:`x`- and :math:`y`-directions.
           * :code:`(0, 1)` for the :math:`x`- and :math:`y`-directions.
    
    n_bins : `int` or array-like
        Number of bins for each axis. If an `int` is provided, the same
        value is used for all axes.

    charges : array-like, keyword-only, optional
        Charge information for the specified `groupings` in the 
        :math:`N_\mathrm{g}` `groups`. If not provided, it will be 
        retrieved from the main 
        :class:`MDAnalysis.core.universe.Universe` object if available.

        .. note::
        
           Depending on the grouping for a specific group, all atoms,
           residues, or segments should have the same charge since the
           charge density profile for the group would not make sense 
           otherwise. If this condition does not hold, change how the
           particles are grouped in `grouping` such that all entities
           share the same charge.

        **Shape**: :math:`(N_\mathrm{g})`.

    dims : array-like, keyword-only, optional
        Raw system dimensions. Affected by `scales`. If the 
        :class:`MDAnalysis.core.universe.Universe` object that the 
        groups in `groups` belong to does not contain dimensionality 
        information, provide it here.

        **Shape**: :math:`(3,)`.

    scales : array-like, keyword-only, optional
        Scaling factors for each system dimension. If an `int` is 
        provided, the same value is used for all axes.

        **Shape**: :math:`(3,)`.

    recenter : `MDAnalysis.AtomGroup` or `tuple`, keyword-only, optional
        Constrains the center of mass of an atom group by adjusting the
        particle coordinates every analysis frame. Either specify an
        :class:`MDAnalysis.core.groups.AtomGroup` or a tuple containing
        an :class:`MDAnalysis.core.groups.AtomGroup` and the fixed 
        center of mass coordinates, in that order. If the center of mass
        is not specified, the center of the simulation box is used.

        **Shape**: :math:`(3,)` for the fixed center of mass.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Affects 
        `results.number_density`, `results.charge_density`, etc.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.
        
    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.

    Attributes
    ----------
    universe : `MDAnalysis.Universe`
        :class:`MDAnalysis.core.universe.Universe` object containing all
        information describing the simulation system.

    results.bins : `list`
        Bin centers corresponding to the density profiles in each 
        dimension.
        
        **Shape**: :math:`(N_\mathrm{axes},)` list of 
        :math:`(N_\mathrm{bins},)` arrays.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    results.number_density : `list`
        Number density profiles. 

        **Shape**: :math:`(N_\mathrm{axes},)` list of 
        :math:`(N_\mathrm{bins},)` arrays.

        **Reference unit**: :math:`\mathrm{Å}^{-3}`.

    results.charge_density : `list`
        Charge density profiles, if charge information is available. 

        **Shape**: :math:`(N_\mathrm{axes},)` list of 
        :math:`(N_\mathrm{bins},)` arrays.

        **Reference unit**: :math:`\mathrm{e/Å}^{-3}`.

    results.potential : `dict`
        Potential profiles, if charge information is available, with
        the key being the axis index. Only available after running 
        :meth:`calculate_potential_profile`.

        **Shape**: :math:`(N_\mathrm{bins},)` for the potential profiles.
            
        **Reference unit**: :math:`\mathrm{V}`.
    """

    _GROUPINGS = {"atoms", "residues", "segments"}

    def __init__(
            self, groups: Union[mda.AtomGroup, ArrayLike],
            groupings: Union[str, ArrayLike] = "atoms",
            axes: Union[int, str, ArrayLike] = "xyz",
            n_bins: Union[int, ArrayLike] = 201, *, 
            dims: np.ndarray = None, charges: ArrayLike = None,
            recenter: dict[str, Any] = None, reduced: bool = False,
            scales: Union[float, ArrayLike] = 1, verbose: bool = True,
            **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        self._dims = self.universe.dimensions[:3].copy()
        if self._dims is None:
            if dims is None:
                raise ValueError("Trajectory does not contain system dimensions.")
            self._dims = dims

        if isinstance(scales, (int, float)) or \
                len(scales) == 3 and isinstance(scales[0], (int, float)):
            self._dims *= scales
        else:
            emsg = ("The scaling factor(s) must be provided as a "
                    "floating-point number or in an array with shape (3,). ")
            raise ValueError(emsg)

        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in self._GROUPINGS:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms', 'residues', and 'segments'.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to the "
                        "number of groups.")
                raise ValueError(emsg)
            for g in groupings:
                if g not in self._GROUPINGS:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms', 'residues', and 'segments'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        if isinstance(axes, int):
            self._axes = np.array((axes,), dtype=int)
        elif isinstance(axes[0], int):
            self._axes = np.asarray(axes, dtype=int)
        elif isinstance(axes[0], str):
            self._axes = np.fromiter((ord(a.lower()) - 120 for a in axes),
                                     dtype=int)

        if isinstance(n_bins, int):
            self._n_bins = n_bins * np.ones(self._axes.shape, dtype=int)
        elif not isinstance(n_bins, str):
            if len(n_bins) == len(self._axes):
                self._n_bins = n_bins
            else:
                emsg = ("The dimension of the array of bin counts is "
                        "incompatible with the number of axes to calculate "
                        "density profiles along.")
                raise ValueError(emsg)
        else:
            emsg = ("The specified bin counts must be an integer or an "
                    "iterable object.")
            raise ValueError(emsg)

        self._reduced = reduced
        if not self._reduced:
            self.results.units = {"_dims": unit.angstrom}

        if charges is None:
            self._charges = np.fromiter(
                (getattr(g, gr).charges[0] 
                 for g, gr in zip(self._groups, self._groupings)),
                dtype=float
            ) if hasattr(self.universe.atoms, "charges") else None
        elif len(charges) == self._n_groups:
            self._charges = np.asarray(charges, dtype=float)
        else:
            emsg = ("The dimension of the array of group charges is "
                    f"incompatible with the number of {self._groupings}.")
            raise ValueError(emsg)
        if self._charges is not None and not self._reduced:
            self.results.units["_charges"] = unit.elementary_charge

        if recenter is None:
            self._recenter = recenter
        elif isinstance(recenter, mda.AtomGroup):
            self._recenter = (recenter, self._dims / 2)
        elif isinstance(recenter, tuple) \
                and isinstance(recenter[0], mda.AtomGroup) \
                and len(recenter) == 2:
            self._recenter = recenter
        else:
            emsg = ("Invalid value passed to 'recenter'. The argument "
                    "must either be a MDAnalysis.AtomGroup or a tuple "
                    "containing a MDAnalysis.AtomGroup and a specified "
                    "center of mass, in that order.")
            raise ValueError(emsg)

        self._verbose = verbose
    
    def _prepare(self) -> None:

        # Define the bin centers for all axes
        self.results.bins = [
            np.linspace(self._dims[a] / (2 * self._n_bins[i]),
                        self._dims[a] - self._dims[a] / (2 * self._n_bins[i]),
                        self._n_bins[i]) for i, a in enumerate(self._axes)
        ]

        # Preallocate arrays to store number of boundary crossings for
        # each particle
        if self._recenter is not None:
            self._trajectory[self.start]
            self._positions_prev = self.universe.atoms.positions
            self._images = np.zeros((self.universe.atoms.n_atoms, 3), dtype=int)
            self._threshold = self._dims / 2

        # Preallocate arrays to hold number density data
        self.results.number_density = [
            np.zeros((self._n_groups, n), dtype=float) for n in self._n_bins
        ]

        # Store reference units
        if not self._reduced:
            self.results.units["results.bins"] = unit.angstrom
            self.results.units["results.number_density"] = unit.angstrom ** -3

        # Preallocate arrays to hold charge density data, if charge
        # information is available
        if self._charges is not None:
            self.results.charge_density = [
                np.zeros((self._n_groups, n), dtype=float) for n in self._n_bins
            ]
            if not self._reduced:
                self.results.units["_charge"] = unit.elementary_charge
                self.results.units["results.charge_density"] = \
                    self.results.units["_charge"] / unit.angstrom ** 3
    
    def _single_frame(self):

        positions = self.universe.atoms.positions.copy()

        if self._recenter is not None:

            # Unwrap all particle positions
            dpos = positions - self._positions_prev
            mask = np.abs(dpos) >= self._threshold
            self._images[mask] -= np.sign(dpos[mask]).astype(int)
            self._positions_prev = self.universe.atoms.positions.copy()
            positions += self._images * self._dims

            # Calculate difference in center of mass
            scom = molecule.center_of_mass(
                positions=positions[self._recenter[0].indices],
                masses=self._recenter[0].masses
            )
            dcom = np.fromiter((0 if np.isnan(cx) else sx - cx
                                for sx, cx in zip(scom, self._recenter[1])),
                               dtype=float, count=3)

            # Shift all particle positions
            positions -= dcom
            indices = (positions < 0) | (positions > self._dims)
            positions[indices] -= (np.floor(positions / self._dims)
                                   * self._dims)[indices]
            
        for i, (ag, g) in enumerate(zip(self._groups, self._groupings)):

            # Get particle positions
            if g == "atoms":
                pos_group = positions[ag.indices]
            else:
                pos_group = molecule.center_of_mass(ag, g)
                if self._recenter is not None:
                    pos_group -= dcom
                    indices = (pos_group < 0) | (pos_group > self._dims)
                    pos_group[indices] -= (np.floor(pos_group / self._dims)
                                           * self._dims)[indices]
            
            # Wrap particles outside of the unit cell
            pos_group += ((pos_group < 0).astype(int) 
                          - (pos_group >= self._dims).astype(int)) * self._dims

            for a, (axis, n_bins) in enumerate(zip(self._axes, self._n_bins)):

                # Compute and tally the bin counts for the current positions
                self.results.number_density[a][i] += np.histogram(
                    pos_group[:, axis], n_bins, (0, self._dims[axis])
                )[0]
        
    def _conclude(self):

        # Compute the volume of the real system
        V = np.prod(self._dims)

        for a in range(len(self._axes)):

            # Divide the bin counts by the bin volumes and number of
            # timesteps to obtain the averaged number density profiles
            self.results.number_density[a] *= self._n_bins[a] / (V * self.n_frames)

            # Compute the charge density profiles
            if self._charges is not None:
                self.results.charge_density[a] = np.dot(
                    self._charges, self.results.number_density[a]
                )

    def calculate_potential_profile(
            self, dielectric: float, axis: Union[int, str], *,
            sigma_e: Union[float, unit.Quantity] = None,
            dV: Union[float, unit.Quantity] = None,
            threshold: float = 1e-5, V0: Union[float, unit.Quantity] = 0
        ) -> None:
        
        r"""
        Calculates the potential profile in the given dimension using
        the charge density profile by numerically solving Poisson's 
        equation for electrostatics.

        Parameters
        ----------
        dielectric : `float`
            Relative permittivity or dielectric constant 
            :math:`\varepsilon_\mathrm{r}`.

        axis : `int` or `str`
            Axis along which to compute the potential profiles.

            .. container::

               **Examples**:

               * :code:`2` for the :math:`z`-direction.
               * :code:`"x"` for the :math:`x`-direction.

        sigma_e : `float` or `openmm.unit.Quantity`, keyword-only, optional
            Total surface charge density :math:`\sigma_e`. Used to 
            ensure that the electric field in the bulk of the solution
            is zero. If not provided, it is determined using `dV` and 
            the charge density profile, or the average value in the 
            center of the integrated charge density profile. 
            
            **Reference unit**: :math:`\mathrm{e/Å^2}`.

        dV : `float` or `openmm.unit.Quantity`, keyword-only, optional
            Potential difference :math:`\Delta \Psi` across the system 
            dimension specified in `axis`. Has no effect if `sigma_e` is
            provided since this value is used solely to calculate 
            `sigma_e`.
             
            .. note::

               By specifying `dV` to calculate `sigma_e` using Gauss's 
               law, it is assumed that the boundaries are perfectly 
               conducting.

            **Reference unit**: :math:`\mathrm{V}`.
        
        threshold : `float`, keyword-only, default: :code:`1e-5`
            Threshold for determining the plateau region of the first
            integral of the charge density profile to calculate
            `sigma_e`. Has no effect if `sigma_e` is provided, or if
            `sigma_e` can be calculated using `dV` and `dielectric`.

        V0 : `float` or `openmm.unit.Quantity`, keyword-only, default: :code:`0`
            Potential :math:`\varphi_0` at the left boundary. 
            
            **Reference unit**: :math:`\mathrm{V}`.
        """

        if not hasattr(self.results, "charge_density"):
            emsg = ("Either call run() before calculate_potential_profile() "
                    "or provide charge information when initializing the "
                    "DensityProfile object.")
            raise RuntimeError(emsg)
        
        if not hasattr(self.results, "potential"):
            self.results.potential = {}

            # Store reference units
            if not self._reduced:
                self.results.units["results.potential"] = unit.volt

        if isinstance(axis, str):
            axis = ord(axis.lower()) - 120
        index = np.where(self._axes == axis)[0][0]

        self.results.potential[axis] = potential_profile(
            self.results.bins[index],
            self.results.charge_density[index],
            self._dims[axis], dielectric, sigma_e=sigma_e, dV=dV, 
            threshold=threshold, V0=V0, reduced=self._reduced
        )