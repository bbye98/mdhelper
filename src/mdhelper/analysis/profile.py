"""
Linear profiles
===============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to quantify properties along axes, such as
density profiles.
"""

import logging
from numbers import Real
from typing import Any, Union
import warnings

import MDAnalysis as mda
import numpy as np
from scipy import integrate, sparse

from .base import DynamicAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm.molecule import center_of_mass
from ..algorithm.topology import unwrap
from ..algorithm.utility import strip_unit

if FOUND_OPENMM:
    from openmm import unit

def calculate_potential_profile(
        bins: np.ndarray[float], charge_density: np.ndarray[float],
        L: float, dielectric: float = 1, *, sigma_q: float = None,
        dV: float = None, threshold: float = 1e-5, V0: float = 0,
        method: str = "integral", pbc: bool = False, reduced: bool = False
    ) -> np.ndarray[float]:

    r"""
    Calculates the potential profile :math:`\Psi(z)` using the charge
    density profile by numerically solving Poisson's equation for
    electrostatics.

    Poisson's equation is given by

    .. math::

       \varepsilon_0\varepsilon_\mathrm{r}\nabla^2\Psi(z)=\rho_q(z)

    where :math:`\varepsilon_0` is the vacuum permittivity,
    :math:`\varepsilon_\mathrm{r}` is the relative permittivity,
    :math:`\rho_q` is the charge density, and :math:`\Psi` is the
    potential.

    The boundary conditions (BCs) are

    .. math::

       \left.\frac{\partial\Psi}{\partial z}\right\rvert_{z=0}&
       =-\frac{\sigma_q}{\varepsilon_0\varepsilon_\mathrm{r}}\\
       \left.\Psi\right\rvert_{z=0}&=\Psi_0

    The first BC is used to ensure that the electric field in the bulk
    of the solution is zero, while the second BC is used to set the 
    potential on the left electrode.

    Poisson's equation can be evaluated by using the trapezoidal rule
    to numerically integrate the charge density profile twice:

    1. Integrate the charge density profile.
    2. Apply the first BC by adding :math:`\sigma_q` to all points.
    3. Integrate the profile from Step 2.
    4. Divide by :math:`-\varepsilon_0\varepsilon_\mathrm{r}`.
    5. Apply the second BC by adding :math:`\Psi_0` to all points.

    This method is fast but requires many histogram bins to accurately
    resolve the potential profile.

    Alternatively, Poisson's equation can be written as a system of
    linear equations 

    .. math::

       A\mathbf{\Psi}=\mathbf{b}
    
    using second-order finite differences. :math:`A` is a tridiagonal
    matrix and :math:`b` is a vector containing the charge density
    profile, with boundary conditions applied in the first and last 
    rows.
    
    The inner equations are given by

    .. math::

       \Psi_i^{''}=\frac{\Psi_{i-1}-2\Psi_i+\Psi_{i+1}}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,i}

    where :math:`i` is the bin index and :math:`h` is the bin width.

    In the case of periodic boundary conditions, the first and last
    equations are given by

    .. math::
    
       \Psi_0^{''}&=\frac{\Psi_{N-1}-2\Psi_0+\Psi_1}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,0}\\
       \Psi_{N-1}^{''}&=\frac{\Psi_{N-2}-2\Psi_{N-1}+\Psi_0}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,N-1}

    When the system has slab geometry, the boundary conditions are
    implemented via

    .. math::

       \Psi_0&=\Psi_0\\
       \Psi_0^\prime&=\frac{-3\Psi_0+4\Psi_1-\Psi_2}{2h}
       =-\frac{\sigma_q}{\varepsilon_0\varepsilon_\mathrm{r}}

    This method is slower but can be more accurate even with fewer
    histogram bins for bulk systems with periodic boundary conditions.

    Parameters
    ----------
    bins : array-like
        Histogram bin centers corresponding to the charge density
        profile in `charge_density`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    charge_density : array-like
        Array containing the charge density profile.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{e/Å}^{-3}`.

    L : `float`
        System size in the dimension that `bins` and `charge_density`
        were calculated in.

        **Reference unit**: :math:`\mathrm{Å}`.

    dielectric : `float`, default: :code:`1`
        Relative permittivity or static dielectric constant
        :math:`\varepsilon_\mathrm{r}`.

    sigma_q : `float`, keyword-only, optional
        Total surface charge density :math:`\sigma_q`. Used to
        ensure that the electric field in the bulk of the solution
        is zero. If not provided, it is determined using `dV` and
        the charge density profile, or the average value in the
        center of the integrated charge density profile if
        :code:`method="integral"`.

        .. note::

           This value should be negative if the potential difference
           is positive and vice versa.

        **Reference unit**: :math:`\mathrm{e/Å^2}`.

    dV : `float`, keyword-only, optional
        Potential difference :math:`\Delta\Psi` across the system
        dimension specified in `axis`. Has no effect if `sigma_q` is
        provided since this value is used solely to calculate
        `sigma_q`.

        **Reference unit**: :math:`\mathrm{V}`.

    threshold : `float`, keyword-only, default: :code:`1e-5`
        Threshold for determining the plateau region of the first
        integral of the charge density profile to calculate
        `sigma_q`. Has no effect if `sigma_q` is provided or if
        `sigma_q` can be calculated using `dV` and `dielectric`.

    V0 : `float`, keyword-only, default: :code:`0`
        Potential :math:`\Psi_0` at the left boundary.

        **Reference unit**: :math:`\mathrm{V}`.

    method : `str`, keyword-only, default: :code:`"integral"`
        Method to use to calculate the potential profile.

        **Valid values**: :code:`"integral"`, :code:`"matrix"`.

    pbc : `bool`, keyword-only, default: :code:`False`
        Specifies whether the axis has periodic boundary conditions.
        Only used when :code:`method="matrix"`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    potential : `numpy.ndarray`
        Potential profile :math:`\Psi(z)`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{V}`.
    """

    if len(bins) != len(charge_density):
        emsg = ("'bins' and 'charge_density' arrays must have the same "
                "length.")
        raise ValueError(emsg)

    CONVERSION_FACTOR = 4 * np.pi if reduced else (
        1 * ureg.elementary_charge / (ureg.vacuum_permittivity * ureg.angstrom)
    ).m_as(ureg.volt)

    # Calculate surface charge density for system with perfectly
    # conducting boundaries
    if sigma_q is None and dV is not None:
        sigma_q = (integrate.trapezoid(bins * charge_density, bins)
                   - dielectric * dV / CONVERSION_FACTOR) / L

    if method == "integral":

        # Calculate the first integral of the charge density profile
        potential = integrate.cumulative_trapezoid(charge_density, bins, 
                                                   initial=0)

        if sigma_q is None:
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
            if len(cut_indices) == 0:
                logging.warning(
                    "No bulk plateau region found in the charge "
                    "density profile. The average value over the "
                    "entire profile will be used."
                )
                sigma_q = potential.mean()
            else:
                target_index = len(potential) // 2
                sigma_q = potential[
                    cut_indices[cut_indices <= target_index][-1]:
                    cut_indices[cut_indices >= target_index][0]
                ].mean()

        # Calculate the second integral of the charge density profile
        return (
            -CONVERSION_FACTOR
            * integrate.cumulative_trapezoid(potential + sigma_q, bins,
                                             initial=V0) / dielectric
        )

    elif method == "matrix":
        if sigma_q is None:
            emsg = ("No surface charge density information. Either "
                    "'sigma_q' or 'dV' must be provided when "
                    "method='matrix'.")
            raise ValueError(emsg)

        h = bins[1] - bins[0]
        if not np.allclose(np.diff(bins), h):
            raise ValueError("'bins' must be uniformly spaced.")
        
        # Set up matrix and load vector for second-order finite
        # difference method
        N = len(bins)
        A = sparse.diags((1, -2, 1), (-1, 0, 1), shape=(N, N), format="csc")
        b = charge_density.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", 
                                  category=sparse.SparseEfficiencyWarning)
            if pbc:
                A[0, -1] = A[-1, 0] = 1
                b *= -CONVERSION_FACTOR * h ** 2 / dielectric
                psi = np.empty_like(b)
                psi[1:] = sparse.linalg.spsolve(A[1:, 1:], b[1:])
                psi[0] = psi[-1]
                return psi
            else:
                A[0, :3] = -1.5, 2, -0.5
                A[-1, 0] = 1
                A[-1, -2:] = 0
                b[0] = -CONVERSION_FACTOR * h * sigma_q / dielectric
                b[1:-1] *= -CONVERSION_FACTOR * h ** 2 / dielectric
                b[-1] = 0
                return sparse.linalg.spsolve(A, b)

class DensityProfile(DynamicAnalysisBase):

    """
    A serial implementation to calculate the number and charge density
    profiles :math:`\\rho_i(z)` and :math:`\\rho_q(z)` of a system
    along the specified axes.

    The microscopic number density profile of species :math:`i` in a
    constant-volume system is calculated by binning particle positions
    along an axis :math:`z` using

    .. math::

       \\rho_i(z)=\\frac{V}{N_\mathrm{bin}}\left\langle
       \sum_\\alpha\delta(z-z_\\alpha)\\right\\rangle

    where :math:`V` is the system volume and :math:`N_\mathrm{bins}` is
    the number of bins. The angular brackets denote an ensemble average.

    If the species carry charges, the charge density profile can be
    obtained using

    .. math::

       \\rho_q(z)=\sum_i z_ie\\rho_i(z)

    where :math:`z_i` is the charge number of species :math:`i` and
    :math:`e` is the elementary charge.

    With the charge density profile, the potential profile can be
    computed by numerically solving Poisson's equation for
    electrostatics:

    .. math::

       \\varepsilon_0\\varepsilon_\\mathrm{r}\\nabla^2\Psi(z)=-\\rho_q(z)

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
        value is used for all axes. If :code:`parallel=True`, the number
        of bins in all axes must be the same.

    charges : array-like, keyword-only, optional
        Charge numbers :math:`z_i` for the specified `groupings` in the
        :math:`N_\\mathrm{g}` `groups`. If not provided, it will be
        retrieved from the main
        :class:`MDAnalysis.core.universe.Universe` object if available.

        .. note::

           Depending on the grouping for a specific group, all atoms,
           residues, or segments should have the same charge since the
           charge density profile for the group would not make sense
           otherwise. If this condition does not hold, change how the
           particles are grouped in `grouping` such that all entities
           share the same charge.

        **Shape**: :math:`(N_\\mathrm{g})`.

        **Reference unit**: :math:`\\mathrm{e}`.

    dimensions : array-like, keyword-only, optional
        System dimensions. If the
        :class:`MDAnalysis.core.universe.Universe` object that the
        groups in `groups` belong to does not contain dimensionality
        information, provide it here. Affected by `scales`.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    dt : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Time between frames :math:`\Delta t`. While this is normally
        determined from the trajectory, the trajectory may not have the
        correct information if the data is in reduced units. For
        example, if your reduced timestep is :math:`0.01` and you output
        trajectory data every :math:`10,000` timesteps, then
        :math:`\Delta t = 100`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    scales : array-like, keyword-only, optional
        Scaling factors for each system dimension. If an `int` is
        provided, the same value is used for all axes.

        **Shape**: :math:`(3,)`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the density profiles are averaged over the
        specified frames.

    recenter : `int`, `MDAnalysis.AtomGroup` or `tuple`, keyword-only, optional
        Constrains the center of mass of an atom group by adjusting the
        particle coordinates every analysis frame. Either specify an
        :class:`MDAnalysis.core.groups.AtomGroup`, its index within
        `groups`, or a tuple containing an 
        :class:`MDAnalysis.core.groups.AtomGroup` or `int` and the fixed
        center of mass coordinates, in that order. If the center of mass
        is not specified, the center of the simulation box is used.

        **Shape**: :math:`(3,)` for the fixed center of mass.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Affects
        `results.number_densities`, `results.charge_densities`, etc.

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

    results.units : `dict`
        Reference units for the results. For example, to get the
        reference units for :code:`results.bins`, call
        :code:`results.units["results.bins"]`.

    results.times : `numpy.ndarray`
        Times at which the density profiles are calculated.

        **Shape**: :math:`(N_\\mathrm{frames},)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.bins : `list`
        Bin centers corresponding to the density profiles in each
        dimension.

        **Shape**: :math:`(N_\\mathrm{axes},)` list of
        :math:`(N_\\mathrm{bins},)` arrays.

        **Reference unit**: :math:`\\mathrm{Å}`.

    results.number_densities : `list`
        Number density profiles.

        **Shape**: :math:`(N_\\mathrm{axes},)` list of
        :math:`(N_\\mathrm{bins},)` arrays.

        **Reference unit**: :math:`\\mathrm{Å}^{-3}`.

    results.charge_densities : `list`
        Charge density profiles, if charge information is available.

        **Shape**: :math:`(N_\\mathrm{axes},)` list of
        :math:`(N_\\mathrm{bins},)` arrays.

        **Reference unit**: :math:`\\mathrm{e/Å}^{-3}`.

    results.potentials : `dict`
        Potential profiles, if charge information is available, with
        the key being the axis index (e.g., :code:`1` for the :math:`z`-
        direction if :code:`axes="yz"`). Only available after running
        :meth:`calculate_potential_profile`.

        **Shape**: :math:`(N_\\mathrm{bins},)` for the potential 
        profiles.

        **Reference unit**: :math:`\\mathrm{V}`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            axes: Union[int, str, tuple[Union[int, str]]] = "xyz",
            n_bins: Union[int, tuple[int]] = 201, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dt: Union[float, "unit.Quantity", Q_] = None,
            scales: Union[float, tuple[float]] = 1, average: bool = True,
            recenter: dict[str, Any] = None, reduced: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe

        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in (GROUPINGS := {"atoms", "residues",
                                               "segments"}):
                emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                        f"{', '.join(GROUPINGS)}.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            for g in groupings:
                if g not in (GROUPINGS := {"atoms", "residues", "segments"}):
                    emsg = (f"Invalid grouping '{g}'. Valid values: "
                            f"{', '.join(GROUPINGS)}.")
                    raise ValueError(emsg)
            self._groupings = groupings

        # Determine the number of particles in each group and their
        # corresponding indices
        self._Ns = tuple(getattr(a, f"n_{g}")
                         for (a, g) in zip(self._groups, self._groupings))
        self._N = sum(self._Ns)
        self._slices = []
        index = 0
        for N in self._Ns:
            self._slices.append(slice(index, index + N))
            index += N

        if isinstance(axes, int):
            self._axes = np.array((axes,), dtype=int)
        else:
            self._axes = np.fromiter(
                (ord(a.lower()) - 120 if isinstance(a, str) else a
                 for a in axes),
                count=len(axes),
                dtype=int
            )

        if isinstance(n_bins, int):
            self._n_bins = n_bins * np.ones(self._axes.shape, dtype=int)
        elif not isinstance(n_bins, str):
            if len(n_bins) == len(self._axes):
                if parallel and np.any(n_bins != n_bins[0]):
                    emsg = ("All axes must use the same number of bins "
                            "when parallel=True.")
                    raise ValueError(emsg)
                self._n_bins = n_bins
            else:
                emsg = ("The dimension of the array of bin counts is "
                        "incompatible with the number of axes to "
                        "calculate density profiles along.")
                raise ValueError(emsg)
        else:
            emsg = ("The specified bin counts must be an integer or an "
                    "iterable object.")
            raise ValueError(emsg)

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            self._dimensions = np.asarray(
                strip_unit(dimensions, "angstrom")[0]
            )
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        if isinstance(scales, Real) or (len(scales) == 3
                                        and isinstance(scales[0], Real)):
            self._dimensions *= scales
        else:
            emsg = ("The scaling factor(s) must be provided as a "
                    "floating-point number or in an array with shape "
                    "(3,).")
            raise ValueError(emsg)

        self._dt, unit_ = strip_unit(dt or self._trajectory.dt, "picosecond")
        if reduced and not isinstance(unit_,  str):
            raise TypeError("'dt' cannot have units when reduced=True.")

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            charges, unit_ = strip_unit(charges, "elementary_charge")
            if reduced and not isinstance(unit_,  str):
                emsg = "'charges' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._charges = np.asarray(charges)
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = np.fromiter(
                (getattr(g, gr).charges[0]
                 for g, gr in zip(self._groups, self._groupings)),
                count=self._n_groups,
                dtype=float
            )
        else:
            self._charges = None

        if recenter is None:
            self._recenter = recenter
        elif isinstance(recenter, int):
            if not 0 <= recenter < self._n_groups:
                raise ValueError("Invalid group index passed to 'recenter'.")
            self._recenter = (recenter, self._dimensions / 2)
        elif isinstance(recenter, mda.AtomGroup):
            try:
                self._recenter = (self._groups.index(recenter), 
                                  self._dimensions / 2)
            except ValueError:
                emsg = ("The specified AtomGroup in 'recenter' is not "
                        "in 'groups'.")
                raise ValueError(emsg)
        elif isinstance(recenter, tuple) \
                and isinstance(recenter[0], (int, mda.AtomGroup)) \
                and len(recenter) == 2:
            if isinstance(recenter[0], mda.AtomGroup):
                try:
                    self._recenter = (self._groups.index(recenter[0]), 
                                      np.asarray(recenter[1]))
                except ValueError:
                    emsg = ("The specified AtomGroup in 'recenter' is not "
                            "in 'groups'.")
                    raise ValueError(emsg)
            else:
                if not 0 <= recenter[0] < self._n_groups:
                    raise ValueError("Invalid group index passed to 'recenter'.")
                self._recenter = (recenter[0], np.asarray(recenter[1]))
        else:
            emsg = ("Invalid value passed to 'recenter'. The argument "
                    "must either be a MDAnalysis.AtomGroup, its index "
                    "in 'groups', or a tuple containing a "
                    "MDAnalysis.AtomGroup or an integer and a "
                    "specified center of mass, in that order.")
            raise ValueError(emsg)

        self._average = average
        self._reduced = reduced
        self._verbose = verbose

    def _prepare(self) -> None:

        # Define the bin centers for all axes
        self.results.bins = [
            np.linspace(
                self._dimensions[a] / (2 * self._n_bins[i]),
                self._dimensions[a]
                    - self._dimensions[a] / (2 * self._n_bins[i]),
                self._n_bins[i]
            ) for i, a in enumerate(self._axes)
        ]

        # Preallocate arrays to store number of boundary crossings for
        # each particle
        if self._recenter is not None:
            self._trajectory[self.start]
            self._positions_old = np.empty((self._N, 3))
            for g, gr, s in zip(self._groups, self._groupings, self._slices):
                self._positions_old[s] = (g.positions if gr == "atoms"
                                          else center_of_mass(g, gr))
            self._images = np.zeros((self._N, 3), dtype=int)
            self._thresholds = self._dimensions / 2

            # Store unwrapped atom positions in a shared memory array
            # for parallel analysis
            if self._parallel:
                self._positions = np.empty((self.n_frames, self._N, 3))
                for i, _ in enumerate(self.universe.trajectory[
                    list(self._sliced_trajectory.frames)
                    if hasattr(self._sliced_trajectory, "frames")
                    else slice(self.start, self.stop, self.step)
                ]):
                    for g, gr, s in zip(self._groups, self._groupings, self._slices):
                        self._positions[i, s] = (g.positions if gr == "atoms"
                                                 else center_of_mass(g, gr))
                        unwrap(self._positions[i, s], 
                               self._positions_old, 
                               self._dimensions,
                               thresholds=self._thresholds,
                               images=self._images)
                        
                        # Calculate difference in center of mass
                        scom = center_of_mass(
                            positions=self._positions
                                      [i, self._slices[self._recenter[0]]],
                            masses=getattr(
                                self._groups[self._recenter[0]], 
                                self._groupings[self._recenter[0]]
                            ).masses
                        )
                        dcom = np.fromiter((0 if np.isnan(cx) else sx - cx
                                            for sx, cx 
                                            in zip(scom, self._recenter[1])),
                                           dtype=float, count=3)

                        # Shift all particle positions
                        self._positions[i, s] -= dcom

                    # Wrap particles outside of the unit cell into the unit cell
                    self._positions[i, s] += (
                        (self._positions[i, s] < 0).astype(int)
                        - (self._positions[i, s] >= self._dimensions).astype(int)
                    ) * self._dimensions

                # Clean up memory
                del self._positions_old
                del self._images
                del self._thresholds

        # Store reference units
        self.results.units = {"results.bins": ureg.angstrom}
        self.results.units["results.number_densities"] = \
            self.results.units["results.bins"] ** -3

        # Preallocate arrays to hold number density data
        if not self._average:
            self.results.times = self.step * self._dt * np.arange(self.n_frames)
        if not self._parallel:
            shape = [self._n_groups]
            if not self._average:
                shape.append(self.n_frames)
            self.results.number_densities = [np.zeros((*shape, n)) 
                                             for n in self._n_bins]

        # Preallocate a list to hold charge density data, if charge
        # information is available
        if self._charges is not None:
            self.results.charge_densities = [None for _ in self._axes]
            self.results.units["results.charge_densities"] = (
                ureg.elementary_charge
                * self.results.units["results.number_densities"]
            )

    def _single_frame(self):

        # Store atom positions in the current frame
        positions = np.empty((self._N, 3))
        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            positions[s] = (g.positions if gr == "atoms" 
                            else center_of_mass(g, gr))

        if self._recenter is not None:

            # Unwrap all particle positions
            unwrap(positions, self._positions_old, self._dimensions,
                   thresholds=self._thresholds, images=self._images)

            # Calculate difference in center of mass
            scom = center_of_mass(
                positions=positions[self._slices[self._recenter[0]]],
                masses=getattr(self._groups[self._recenter[0]], 
                               self._groupings[self._recenter[0]]).masses
            )
            dcom = np.fromiter((0 if np.isnan(cx) else sx - cx
                                for sx, cx in zip(scom, self._recenter[1])),
                               dtype=float, count=3)

            # Shift all particle positions
            positions -= dcom
        
        # Wrap particles outside of the unit cell into the unit cell
        positions += (
            (positions < 0).astype(int)
            - (positions >= self._dimensions).astype(int)
        ) * self._dimensions

        for i, (gr, s) in enumerate(zip(self._groupings, self._slices)):
            for a, (axis, n_bins) in enumerate(zip(self._axes, self._n_bins)):

                # Compute and tally the bin counts for the current positions
                if self._average:
                    self.results.number_densities[a][i] += np.histogram(
                        positions[s, axis], n_bins, (0, self._dimensions[axis])
                    )[0]
                else:
                    self.results.number_densities[a][i, self._frame_index] \
                        = np.histogram(positions[s, axis], n_bins,
                                       (0, self._dimensions[axis]))[0]
                    
    def _single_frame_parallel(
            self, frame: int, index: int) -> tuple[int, np.ndarray[float]]:
        
        self._trajectory[frame]
        results = np.empty((len(self._axes), self._n_groups, self._n_bins[0]))

        if self._recenter is None:

            # Store atom positions in the current frame
            positions = np.empty((self._N, 3))
            for g, gr, s in zip(self._groups, self._groupings, self._slices):
                positions[s] = (g.positions if gr == "atoms" 
                                else center_of_mass(g, gr))
                
            # Wrap particles outside of the unit cell into the unit cell
            positions += (
                (positions < 0).astype(int)
                - (positions >= self._dimensions).astype(int)
            ) * self._dimensions

        else:

            # Get unwrapped positions from shared array
            positions = self._positions[index]

        for i, (gr, s) in enumerate(zip(self._groupings, self._slices)):
            for a, axis in enumerate(self._axes):

                # Compute and tally the bin counts for the current positions 
                results[a, i] = np.histogram(
                    positions[s, axis], self._n_bins[0], (0, self._dimensions[axis])
                )[0]

        return (index, results)

    def _conclude(self):

        # Consolidate parallel results
        if self._parallel:
            self.results.number_densities = np.stack(
                [r[1] for r in sorted(self._results)], axis=1
            )
            if self._average:
                self.results.number_densities \
                    = self.results.number_densities.sum(axis=1)

        # Compute the volume of the real system
        V = np.prod(self._dimensions)

        for a in range(len(self._axes)):

            # Divide the bin counts by the bin volumes and number of
            # timesteps to obtain the averaged number density profiles
            denom = self._n_bins[a] / V
            if self._average:
                denom /= self.n_frames
            self.results.number_densities[a] *= denom

            # Compute the charge density profiles
            if self._charges is not None:
                self.results.charge_densities[a] = np.einsum(
                    "g,g...b->...b",
                    self._charges,
                    self.results.number_densities[a]
                )

    def calculate_potential_profile(
            self, dielectric: float, axis: Union[int, str], *,
            sigma_q: Union[float, "unit.Quantity", Q_] = None,
            dV: Union[float, "unit.Quantity", Q_] = None,
            threshold: float = 1e-5, V0: Union[float, "unit.Quantity", Q_] = 0,
            method: str = "integral", pbc: bool = False) -> None:

        """
        Calculates the average potential profile in the given dimension
        using the charge density profile by numerically solving Poisson's
        equation for electrostatics.

        Parameters
        ----------
        dielectric : `float`
            Relative permittivity or dielectric constant
            :math:`\\varepsilon_\\mathrm{r}`.

        axis : `int` or `str`
            Axis along which to compute the potential profiles.

            .. container::

               **Examples**:

               * :code:`2` for the :math:`z`-direction.
               * :code:`"x"` for the :math:`x`-direction.

        sigma_q : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
        keyword-only, optional
            Total surface charge density :math:`\\sigma_q`. Used to
            ensure that the electric field in the bulk of the solution
            is zero. If not provided, it is determined using `dV` and
            the charge density profile, or the average value in the
            center of the integrated charge density profile.

            **Reference unit**: :math:`\\mathrm{e/Å^2}`.

        dV : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
        keyword-only, optional
            Potential difference :math:`\Delta\Psi` across the system
            dimension specified in `axis`. Has no effect if `sigma_q` is
            provided since this value is used solely to calculate
            `sigma_q`.

            **Reference unit**: :math:`\\mathrm{V}`.

        threshold : `float`, keyword-only, default: :code:`1e-5`
            Threshold for determining the plateau region of the first
            integral of the charge density profile to calculate
            `sigma_q`. Has no effect if `sigma_q` is provided, or if
            `sigma_q` can be calculated using `dV` and `dielectric`.

        V0 : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
        keyword-only, default: :code:`0`
            Potential :math:`\Psi_0` at the left boundary.

            **Reference unit**: :math:`\\mathrm{V}`.

        method : `str`, keyword-only, default: :code:`"integral"`
            Method to use to calculate the potential profile.

            **Valid values**: :code:`"integral"`, :code:`"matrix"`.

        pbc : `bool`, keyword-only, default: :code:`False`
            Specifies whether the system has periodic boundary conditions.
            Only used when :code:`method="matrix"`.
        """

        if not hasattr(self.results, "charge_densities"):
            emsg = ("Either call run() before calculate_potential_profile() "
                    "or provide charge information when initializing the "
                    "DensityProfile object.")
            raise RuntimeError(emsg)

        if not hasattr(self.results, "potentials"):
            self.results.potentials = {}

            # Store reference units
            self.results.units["results.potentials"] = ureg.volt

        if isinstance(axis, str):
            axis = ord(axis.lower()) - 120
        index = np.where(self._axes == axis)[0][0]

        if sigma_q is not None:
            sigma_q, unit_ = strip_unit(sigma_q, "elementary_charge/angstrom**2")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'sigma_q' cannot have units when reduced=True."
                raise ValueError(emsg)

        if dV is not None:
            dV, unit_ = strip_unit(dV, "volt")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'dV' cannot have units when reduced=True."
                raise ValueError(emsg)

        if V0 is not None:
            V0, unit_ = strip_unit(V0, "volt")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'V0' cannot have units when reduced=True."
                raise ValueError(emsg)

        charge_density = self.results.charge_densities[index]
        if charge_density.ndim == 3:
            charge_density = charge_density.mean(axis=1)
        self.results.potentials[index] = calculate_potential_profile(
            self.results.bins[index],
            charge_density,
            self._dimensions[axis],
            dielectric,
            sigma_q=sigma_q,
            dV=dV,
            threshold=threshold,
            V0=V0,
            method=method,
            pbc=pbc,
            reduced=self._reduced
        )