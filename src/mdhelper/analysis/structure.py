"""
Bulk structural analysis
========================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains classes to analyze the structure of bulk fluid and
electrolyte systems.
"""

from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
from scipy.integrate import simpson
from scipy.signal import argrelextrema

try:
    from openmm import unit
    FOUND_OPENMM = True
except:
    FOUND_OPENMM = False

from .base import SerialAnalysisBase, ParallelAnalysisBase
from .. import ArrayLike
from ..algorithm import molecule

def radial_histogram(
        pos1: np.ndarray, pos2: np.ndarray, n_bins: int, range: ArrayLike,
        dims: ArrayLike, *, exclusion: ArrayLike = None) -> np.ndarray:
    
    r"""
    Calculates the radial histogram of distances between particles of
    the same type or two different types.

    Parameters
    ----------
    pos1 : `numpy.ndarray`
        :math:`N_1` positions or center of masses of particles in the
        first group.
        
        **Shape**: :math:`(N_1,\,3)`.

    pos2 : `numpy.ndarray`
        :math:`N_2` positions or center of masses of particles in the
        second group.

        **Shape**: :math:`(N_2,\,3)`.

    n_bins : `int`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range : array-like
        Range of radii values.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    dims : array-like
        System dimensions and orthogonality.

        **Shape**: :math:`(6,)`.
        
        **Reference unit**: :math:`\mathrm{Å}` (dimensions), 
        :math:`^\circ` (orthogonality).

    exclusion : array-like, keyword-only, optional
        Tiles to exclude from the interparticle distances.

        **Shape**: :math:`(2,)`.

        **Example**: :code:`(1, 1)` to exclude self-interactions.

    Returns
    -------
    histogram : `numpy.ndarray`
        Radial histogram.

        **Shape**: :math:`(N_\mathrm{bins},)`.
    """

    # Get pair separation distances of atom pairs within range
    pairs, dist = distances.capped_distance(pos1, pos2, range[1], 
                                            box=dims)
    
    # TODO: MAKE MEMORY SAFE
    
    # Exclude atom pairs with the same atoms or atoms from the
    # same residue
    if exclusion is not None:
        dist = dist[np.where(pairs[:, 0] // exclusion[0] 
                             != pairs[:, 1] // exclusion[1])[0]]
    
    return np.histogram(dist, bins=n_bins, range=range)[0]

def coordination_number(
        bins: np.ndarray, rdf: np.ndarray, rho: float, *, 
        n: int = 2, threshold: float = 0.1) -> np.ndarray:

    r"""
    Calculates coordination numbers :math:`n_k` from a radial 
    distribution function :math:`g_{ij}(r)`.

    The definition is

    .. math::
       
       n_k=4\pi\rho_j\int_{r_{k-1}}^{r_k}r^2g_{ij}(r)\,dr

    where :math:`k` is the index, :math:`\rho_j` is the bulk number 
    density of species :math:`j` and :math:`r_k` is the 
    :math:`(k + 1)`-th local minimum of :math:`g_{ij}(r)`.

    If the radial distribution function :math:`g_{ij}(r)` does not 
    contain as many local minima as `n`, this method will return 
    `numpy.nan` for the coordination numbers that could not be 
    calculated.

    Parameters
    ----------
    bins : `numpy.ndarray`
        Centers of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    rdf : `numpy.ndarray`
        Radial distribution function :math:`g_{ij}(r)`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    rho : `float`
        Number density :math:`\rho_j` of species :math:`j`.
        
        **Reference unit**: :math:`\mathrm{Å}^{-3}`.

    n : `int`, keyword-only, default: :code:`2`
        Number of coordination numbers to calculate.

    threshold : `float`, keyword-only, default: :code:`0.1`
        Minimum :math:`g_{ij}(r)` value that must be reached before 
        local minima are found.

    Returns
    -------
    coord_nums : `numpy.ndarray`
        Coordination numbers :math:`n_k`.
    """

    coord_nums = np.empty(n, dtype=float)
    coord_nums[:] = np.nan

    # Find indices of minima in the radial distribution function
    i_min, = argrelextrema(rdf, np.less)
    i_min = i_min[rdf[i_min] >= threshold]
    n_min = len(i_min)

    # Integrate the radial distribution function to get the coordination
    # number(s)
    if n_min:
        r = bins[:i_min[0] + 1]
        coord_nums[0] = 4 * np.pi * rho \
                        * simpson(r ** 2 * rdf[:i_min[0] + 1], r)
        for i in range(min(n, n_min) - 1):
            r = bins[i_min[i]:i_min[i + 1] + 1]
            coord_nums[i + 1] = \
                4 * np.pi * rho \
                * simpson(r ** 2 * rdf[i_min[i]:i_min[i + 1] + 1], r)
    else:
        warnings.warn("No local minima found.")

    return coord_nums

def radial_fourier_transform(
        r: np.ndarray, f: np.ndarray, q: np.ndarray) -> np.ndarray:

    r"""
    Computes the radial Fourier transform :math:`\hat{f}(q)` of
    discrete data :math:`f(r)`.

    .. math::

       \hat{f}(q)=\frac{4\pi}{q}\int_0^\infty f(r)r\sin(qr)\,dr

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.

    f : `numpy.ndarray`
        Discrete data :math:`f(r)` to Fourier transform. 
        
        **Shape**: Same as `r`.
        
    q : `numpy.ndarray`
        Wavenumbers :math:`q` to evaluate the Fourier transforms at.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    Returns
    -------
    rft : `numpy.ndarray`
        Radial Fourier transform of the discrete data.

        **Shape**: Same as `q`.
    """

    rft = 4 * np.pi * np.divide(simpson(f * r * np.sin(np.outer(q, r)), r), q)
    if 0 in q:
        rft[q == 0] = 4 * np.pi * simpson(f * r ** 2, r)
    return rft

def structure_factor(
        r: np.ndarray, g: np.ndarray, equal: bool, rho: float,
        x_i: float = 1, x_j: float = None, q: np.ndarray = None, *, 
        q_lower: float = None, q_upper: float = None, n_q: int = 1000,
        formalism: str = "FZ") -> tuple[np.ndarray, np.ndarray]:
    
    r"""
    Computes the (partial) static structure factor :math:`S_{ij}(q)` 
    using the radial histogram bins :math:`r` and the radial 
    distribution function :math:`g_{ij}(r)` for an isotropic fluid.

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.
    
    g : `numpy.ndarray`
        Radial distribution function :math:`g_{ij}(r)`.

        **Shape**: Same as `r`.

    equal : `bool`
        Specifies whether `g` is between the same species, or 
        :math:`i = j`. If :code:`False`, the number concentrations of
        species :math:`i` and :math:`j` must be specified in `x_i` and
        `x_j`.
    
    rho : `float`
        Bulk number density :math:`\rho`.

        **Reference unit**: :math:`\mathrm{Å}^{-3}`.

    x_i : `float`, default: :code:`1`
        Number concentration of species :math:`i`. Required if
        :code:`equal=False`.

    x_j : `float`, optional
        Number concentration of species :math:`j`. Required if
        :code:`equal=False`.

    q : `numpy.ndarray`, optional
        Wavenumbers :math:`q`.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    q_lower : `float`, keyword-only, optional
        Lower bound for the wavenumbers :math:`q`. Has no effect if `q`
        is specified.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.
    
    q_upper : `float`, keyword-only, optional
        Upper bound for the wavenumbers :math:`q`. Has no effect if `q`
        is specified.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.
    
    n_q : `int`, keyword-only, default: :code:`1000`
        Number of wavenumbers :math:`q` to generate. Has no effect if `q`
        is specified.

    formalism : `str`, keyword-only, default: :code:`"FZ"`
        Formalism to use for the partial structure factor. Has no effect
        if :code:`equal=True`.

        .. container::
        
           **Valid values**: 
           
           * :code:`"general"`: A general formalism given by

             .. math::

                S_{ij}(q)=1+x_ix_j\frac{4\pi\rho}{q}\int_0^\infty 
                (g_{ij}(r)-1)r\sin{(qr)}\,dr
           
           * :code:`"FZ"`: Faber–Ziman formalism [1]_

             .. math::

                S_{ij}(q)=1+\frac{4\pi\rho}{q}\int_0^\infty
                (g_{ij}(r)-1)r\sin{(qr)}\,dr
             
           * :code:`"AL"`: Ashcroft–Langreth formalism [2]_

             .. math::

                S_{ij}(q)=\delta_{ij}+(x_ix_j)^{1/2}\frac{4\pi\rho}{q}
                \int_0^\infty (g_{ij}(r)-1)r\sin{(qr)}\,dr

    Returns
    -------
    q : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.
    
    S : `numpy.ndarray`
        (Partial) static structure factor :math:`S(q)`. 

        **Shape**: :math:`(N_q,)`.

    References
    ----------
    .. [1] T. E. Faber and J. M. Ziman, A Theory of the Electrical 
       Properties of Liquid Metals: III. the Resistivity of Binary 
       Alloys, *Philosophical Magazine* **11**, **153** (1965).
       https://doi.org/10.1080/14786436508211931

    .. [2] N. W. Ashcroft and D. C. Langreth, Structure of Binary Liquid
       Mixtures. I, *Phys. Rev.* **156**, **685** (1967).
       https://doi.org/10.1103/PhysRev.156.685
    """

    if q is None:
        if q_lower is None:
            q_lower = 2 * np.pi / r[-1]
        if q_upper is None:
            q_upper = 2 * np.pi / r[0]
        q = np.linspace(q_lower, q_upper, 
                        int((q_upper - q_lower) / q_lower) 
                        if n_q is None else n_q)
    
    rho_sft = rho * radial_fourier_transform(r, g - 1, q)
    if equal or formalism == "FZ":
        return q, 1 + rho_sft
    elif not equal:
        if formalism == "AL":
            return q, (x_i == x_j) + np.sqrt(x_i * x_j) * rho_sft
        elif formalism == "general":
            return q, 1 + x_i * x_j * rho_sft
    raise ValueError("Invalid formalism.")

class RDF(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the radial distribution 
    function (RDF) :math:`g_{ij}(r)` between types :math:`i` and 
    :math:`j`.

    It is given by

    .. math::

       g_{ij}(r)=\frac{1}{N_iN_j}\sum_\alpha\sum_\beta \left\langle
       \delta\left(|\pmb{r}_\alpha-\pmb{r}_\beta|-r\right)\right\rangle

    where :math:`N_i` and :math:`N_j` are the number of particles, and
    :math:`\pmb{r}_\alpha` and :math:`\pmb{r}_\beta` are the positions 
    of particles :math:`\alpha` and :math:`\beta` belonging to 
    species :math:`i` and :math:`j`, respectively. The RDF is normalized
    such that :math:`\lim_{r\rightarrow\infty}g_{ij}(r)=1` in a 
    homogeneous system.

    (A closely related quantity is the single particle density 
    :math:`n_{ij}(r)=\rho_jg_{ij}(r)`, where :math:`\rho_j` is the
    number density of species :math:`j`.)

    The cumulative RDF is

    .. math::

       G_{ij}(r)=\int_0^r4\pi Rg_{ij}(R)\,dR

    and the average number of :math:`j` particles found within radius
    :math:`r` is

    .. math::

       N_{ij}(r)=\rho_jG_{ij}(r)

    The expression above can be used to compute the coordination numbers
    (number of neighbors in each solvation shell) by setting :math:`r`
    to the :math:`r`-values where :math:`g_{ij}(r)` is locally 
    maximized, which signify the solvation shell boundaries.

    .. container::

       The RDF can also be used to obtain other relevant structural 
       properties, such as

       * the potential of mean force

         .. math::

            w_{ij}(r)=-k_\mathrm{B}T\ln{g_{ij}(r)}

         where :math:`k_\mathrm{B}` is the Boltzmann constant and 
         :math:`T` is the system temperature, and

       * the (partial) static structure factor (see 
         :func:`structure_factor` for the possible definitions).

    Parameters
    ----------
    ag1 : `MDAnalysis.AtomGroup`
        First atom group :math:`i`.

    ag2 : `MDAnalysis.AtomGroup`
        Second atom group :math:`j`.
    
    n_bins : `int`, default: :code:`201`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range : array-like, default: :code:`(0.0, 15.0)`
        Range of radii values.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    norm : `str`, keyword-only, default: :code:`"rdf"`
        Determines how the radial histograms are normalized.
        
        .. container::

           **Valid values**: 
        
           * :code:`norm="rdf"`: The radial distribution function
             :math:`g_{ij}(r)` is computed.
           * :code:`norm="density"`: The single particle density 
             :math:`n_{ij}(r)` is computed.
           * :code:`norm=None`: The raw particle pair count in the 
             radial histogram bins is returned.

    exclusion : array-like, keyword-only, optional
        Tiles to exclude from the interparticle distances. The 
        `groupings` parameter dictates what a tile represents.

        **Shape**: :math:`(2,)`.

        **Example**: :code:`(1, 1)` to exclude self-interactions.

    groupings : `str` or array-like, keyword-only, default: :code:`"atoms"`
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

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

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
        reference units for :code:`results.bins`, call 
        :code:`results.units["results.bins"]`. Only available if OpenMM
        is installed.

    results.edges : `numpy.ndarray`
        Edges of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins}+1,)`.

        **Reference unit**: :math:`\textrm{Å}`.

    results.bins : `numpy.ndarray`
        Centers of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\textrm{Å}`.

    results.counts : `numpy.ndarray`
        Raw particle pair counts in the radial histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    results.rdf : `numpy.ndarray`
        .. container::

           One of
        
           * :code:`norm="rdf"`: the radial distribution function
             :math:`g_{ij}(r)`,
           * :code:`norm="density"`: the single particle density
             :math:`n_{ij}(r)`, or
           * :code:`norm=None`: the raw particle pair count in the
             radial histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    results.coordination_numbers : `numpy.ndarray`
        Coordination numbers :math:`n_k`. Only available after running 
        :meth:`calculate_coordination_number`.

    results.pmf : `numpy.ndarray`
        Potential of mean force :math:`w(r)`. Only available after
        running :meth:`calculate_pmf`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{kJ/mol}`.

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`. Only available after running
        :meth:`calculate_structure_factor`.

        **Reference unit**: :math:`\textrm{Å}^{-1}`.

    results.ssf : `numpy.ndarray`
        (Partial) static structure factor. Only available after running
        :meth:`calculate_structure_factor`.

        **Shape**: Same as `results.wavenumbers`.
    """

    _GROUPINGS = {"atoms", "residues", "segments"}

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: ArrayLike = (0.0, 15.0), *,
            norm: str = "rdf", exclusion: ArrayLike = None,
            groupings: Union[str, ArrayLike] = "atoms",
            reduced: bool = False, verbose: bool = False, **kwargs) -> None:

        self.ag1 = ag1
        self.ag2 = ag1 if ag2 is None else ag2
        self.universe = self.ag1.universe
        if self.universe.dimensions is None and self._ts.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        if isinstance(groupings, str):
            if groupings not in self._GROUPINGS:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms', 'residues', and 'segments'.")
                raise ValueError(emsg)
            self._groupings = 2 * [groupings]
        else:
            for g in groupings:
                if g not in self._GROUPINGS:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms', 'residues', and 'segments'.")
                    raise ValueError(emsg)
            self._groupings = 2 * groupings if len(groupings) == 1 else groupings

        self._n_bins = n_bins
        self._range = range
        self._norm = norm
        self._exclusion = exclusion
        self._reduced = reduced
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to store neighbor counts
        self.results.edges = np.linspace(*self._range, self._n_bins + 1)
        self.results.bins = (self.results.edges[:-1] 
                             + self.results.edges[1:]) / 2
        self.results.counts = np.zeros(self._n_bins, dtype=int)
        if not self._reduced and FOUND_OPENMM:
            self.results.units = {"results.bins": unit.angstrom,
                                  "results.edges": unit.angstrom}

        # Preallocate floating-point number for total volume analyzed
        # (for when system dimensions can change, such as during NpT
        # equilibration)
        if self._norm == "rdf":
            self._volume = 0.0

    def _single_frame(self) -> None:

        # Tally counts in each pair separation distance bin
        self.results.counts += radial_histogram(
            pos1=self.ag1.positions 
                 if self._groupings[0] == "atoms"
                 else molecule.center_of_mass(self.ag1, self._groupings[0]),
            pos2=self.ag2.positions 
                 if self._groupings[1] == "atoms"
                 else molecule.center_of_mass(self.ag2, self._groupings[1]),
            n_bins=self._n_bins,
            range=self._range,
            dims=self._ts.dimensions,
            exclusion=self._exclusion
        )
        
        # Add volume analyzed
        if self._norm == "rdf":
            self._volume += self._ts.volume
            
    def _conclude(self):

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            norm *= 4 * np.pi * np.diff(self.results.edges ** 3) / 3
            if self._norm == "rdf":
                _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self.ag1, f"n_{self._groupings[0]}") * _N2
                         * self.n_frames / self._volume)
        
        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm

    def _get_rdf(self) -> np.ndarray:

        """
        Returns the existing radial distribution function (RDF) if 
        :code:`norm="rdf"` was passed to the :class:`RDF` constructor. 
        Otherwise, the RDF is calculated and returned.

        Returns
        -------
        rdf : `numpy.ndarray`
            Radial distribution function :math:`g_{ij}(r)`.
        """

        if self._norm == "rdf":
            return self.results.rdf
        else:
            _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
            if self._exclusion:
                _N2 -= self._exclusion[1]
            return 3 * self._volume * self.results.counts / (
                4 * np.pi * self.n_frames ** 2 * _N2 
                * getattr(self.ag1, f"n_{self._groupings[0]}") 
                * np.diff(self.results.edges ** 3)
            )

    def calculate_coordination_number(
            self, rho: float, *, n: int = 2, threshold: float = 0.1) -> None:

        r"""
        Calculates the coordination numbers :math:`n_k`.

        If the radial distribution function :math:`g_{ij}(r)` does not 
        contain :math:`k` local minima, this method will return 
        `numpy.nan` for the coordination numbers that could not be 
        calculated.

        Parameters
        ----------
        rho : `float`
            Number density :math:`\rho_j` of species :math:`j`.
            
            **Reference unit**: :math:`\mathrm{nm}^{-3}`.

        n : `int`, keyword-only, default: :code:`2`
            Number of coordination numbers to calculate.

        threshold : `float`, keyword-only, default: :code:`0.1`
            Minimum :math:`g_{ij}(r)` value for a local minimum to be
            considered the boundary of a radial shell.
        """

        self.results.coordination_numbers = coordination_number(
            self.results.bins, self._get_rdf(), rho, n=n, threshold=threshold
        )

    def calculate_pmf(self, temp: Union[float, "unit.Quantity"]) -> None:

        r"""
        Calculates the potential of mean force :math:`w_{ij}(r)`.

        Parameters
        ----------
        temp : `float` or `openmm.unit.Quantity`
            System temperature :math:`T`.

            .. note::

               If :code:`reduced=True` was set in the :class:`RDF` 
               constructor, `temp` should be equal to the energy scale.
               When the Lennard-Jones potential is used, it generally
               means that :math:`T^*=1`, or `temp=1`.

            **Reference unit**: :math:`\mathrm{K}`.
        """

        if self._reduced:
            kBT = temp
        else:
            if isinstance(temp, (int, float)):
                if FOUND_OPENMM:
                    kBT = (unit.AVOGADRO_CONSTANT_NA 
                           * unit.BOLTZMANN_CONSTANT_kB * temp * unit.kelvin
                           / unit.kilojoule_per_mole)
                else:
                    kBT = 0.00831446261815324 * temp
            else:
                kBT = (unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB
                       * temp / unit.kilojoule_per_mole)
            if FOUND_OPENMM:
                self.results.units = {"results.pmf": unit.kilojoule_per_mole}

        self.results.pmf = -kBT * np.log(self._get_rdf())

    def calculate_structure_factor(
            self, rho: float, x_i: float = None, x_j: float = None, 
            q: np.ndarray = None, *, q_lower: float = None, 
            q_upper: float = None, n_q: int = 1000, formalism: str = "FZ"
        ) -> None:
    
        r"""
        Computes the (partial) static structure factor :math:`S_{ij}(q)`
        using the radial histogram bins :math:`r` and the radial 
        distribution function :math:`g_{ij}(r)` for an isotropic fluid.

        Parameters
        ----------
        rho : `float`
            Bulk number density :math:`rho`.

            **Reference unit**: :math:`\mathrm{Å}^{-3}`.

        x_i : `float`, default: :code:`1`
            Number concentration of species :math:`i`. Required if
            the two atom groups are not identical.

        x_j : `float`, optional
            Number concentration of species :math:`j`. Required if
            the two atom groups are not identical.
            
        q : `numpy.ndarray`, optional
            Wavenumbers :math:`q`.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.

        q_lower : `float`, keyword-only, optional
            Lower bound for the wavenumbers :math:`q`. Has no effect if `q`
            is specified.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.
        
        q_upper : `float`, keyword-only, optional
            Upper bound for the wavenumbers :math:`q`. Has no effect if `q`
            is specified.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.
        
        n_q : `int`, keyword-only, default: :code:`1000`
            Number of wavenumbers :math:`q` to generate. Has no effect if `q`
            is specified.

        formalism :`str`, keyword-only, default: :code:`"FZ"`
            Formalism to use for the partial structure factor. Has no effect
            if the two atom groups are the same.

            .. container::
            
               **Valid values**: 
            
               * :code:`"general"`: A general formalism similar to that
                 of the static structure factor, except the second term
                 is multiplied by :math:`x_ix_j`.
               * :code:`"FZ"`: Faber–Ziman formalism.
               * :code:`"AL"`: Ashcroft–Langreth formalism.
        """

        self.results.wavenumbers, self.results.ssf = structure_factor(
            self.results.bins, self._get_rdf(), self.ag1 == self.ag2, 
            rho, x_i, x_j, q=q, q_lower=q_lower, q_upper=q_upper, 
            n_q=n_q, formalism=formalism
        )

class ParallelRDF(RDF, ParallelAnalysisBase):

    """
    A multithreaded implementation to calculate the radial distribution
    function :math:`g_{ij}(r)` and its related properties.
    
    .. note::
       For a theoretical background and a complete list of
       parameters, attributes, and available methods, see :class:`RDF`.
    """

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: ArrayLike = (0.0, 15.0), *,
            norm: str = "rdf", exclusion: ArrayLike = None, 
            groupings: Union[str, ArrayLike] = "atoms",
            reduced: bool = False, verbose: bool = True, **kwargs) -> None:

        RDF.__init__(self, ag1, ag2, n_bins, range, norm=norm,
                     exclusion=exclusion, groupings=groupings, 
                     reduced=reduced, verbose=verbose, **kwargs)
        ParallelAnalysisBase.__init__(self, ag1.universe.trajectory,
                                      verbose=verbose, **kwargs)

    def _single_frame(self, frame: int, timestep: int) -> np.ndarray:

        _ts = self._trajectory[frame]
        result = np.empty(1 + self._n_bins, dtype=float)

        # Compute radial histogram for a single frame
        result[:self._n_bins] = radial_histogram(
            pos1=self.ag1.positions 
                 if self._groupings[0] == "atoms"
                 else molecule.center_of_mass(self.ag1, self._groupings[0]),
            pos2=self.ag2.positions 
                 if self._groupings[1] == "atoms"
                 else molecule.center_of_mass(self.ag2, self._groupings[1]),
            n_bins=self._n_bins,
            range=self._range,
            dims=_ts.dimensions,
            exclusion=self._exclusion
        )

        # Store system volume of current frame in the last slot of the
        # results array
        result[self._n_bins] = _ts.volume

        return result

    def _conclude(self):

        # Tally counts in each pair separation distance bin over all
        # frames
        self._results = np.vstack(self._results).sum(axis=0)
        self.results.counts[:] = self._results[:self._n_bins]
        self._volume = self._results[self._n_bins]

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            norm *= 4 * np.pi * np.diff(self.results.edges ** 3) / 3
            if self._norm == "rdf":
                _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self.ag1, f"n_{self._groupings[0]}") * _N2
                         * self.n_frames / self._volume)
        
        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm
        
class StructureFactor(SerialAnalysisBase):

    r"""
    The static structure factor :math:`S(q)` is a measure of how a
    material scatters incident radiation. It is defined as

    .. math::

        S(\mathbf{q})&=\frac{1}{N}\left\langle\sum_{j=1}^N\sum_{k=1}^N
        \exp{[-i\mathbf{q}\cdot(\mathbf{r}_j-\mathbf{r}_k)]}\right\rangle\\
        &=\frac{1}{N}\left\langle\left[
        \sum_{j=1}^N\sin{(\mathbf{q}\cdot\mathbf{r}_j)}\right]^2+\left[
        \sum_{j=1}^N\cos{(\mathbf{q}\cdot\mathbf{r}_j)}\right]^2\right\rangle

    where :math:`N` is the number of particles, :math:`\mathbf{q}` is
    the scattering wavevector, and :math:`\mathbf{r}_i` is the position 
    of the :math:`i`-th monomer.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms that share the same grouping type. All atoms
        in the universe must be assigned to a group.

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

    n_points : `int`, default: :code:`32`
        Number of points in the scattering wavevector grid.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines whether atom positions are unwrapped.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

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
        reference units for :code:`results.wavenumbers`, call 
        :code:`results.units["results.wavenumbers"]`. Only available if
        OpenMM is installed.

    results.wavenumbers : `numpy.ndarray`
        Scattering wavenumbers :math:`q`.

    results.ssf : `numpy.ndarray`
        Static structure factor :math:`S(q)`.

        **Shape**: Same as `results.wavenumbers`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, ArrayLike],
            groupings: Union[str, ArrayLike] = "atoms", n_points: int = 32,
            *, unwrap: bool = False, verbose: bool = True, **kwargs):

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        if sum(g.n_atoms for g in self._groups) != self.universe.atoms.n_atoms:
            emsg = ("The provided atom groups do not contain all atoms "
                    "in the universe.")
            raise ValueError(emsg)
        
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)
        self._dims = self.universe.dimensions[:3].copy()

        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in {"atoms", "residues"}:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms' and 'residues'.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to the "
                        "number of groups.")
                raise ValueError(emsg)
            for g in groupings:
                if g not in {"atoms", "residues"}:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms' and 'residues'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        self._Ns = tuple(getattr(a, f"n_{g}")
                         for (a, g) in zip(self._groups, self._groupings))
        self._N = sum(self._Ns)
        self._slices = []
        index = 0
        for N in self._Ns:
            self._slices.append(slice(index, index + N))
            index += N

        self._n_points = n_points
        self._wavevectors = np.stack(
            np.meshgrid(
                *[2 * np.pi * np.arange(self._n_points) / L
                 for L in self._dims]
            ), -1
        ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        self._unwrap = unwrap
        self._verbose = verbose

        if FOUND_OPENMM:
            self.results.units = {"_dims": unit.angstrom}

    def _prepare(self) -> None:

        # Unwrap particle positions if necessary
        self._positions = np.empty((self._N, 3), dtype=float)
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0] 
                if hasattr(self._sliced_trajectory, "frames") 
                else (self.start or 0)
            ]
            self._positions_old = np.empty(self._positions.shape, dtype=float)
            for g, gr, s in zip(self._groups, self._groupings, self._slices):
                self._positions_old[s] = \
                    g.positions if gr == "atoms" \
                    else molecule.center_of_mass(g, gr)
            self._images = np.zeros(self._positions_old.shape, dtype=int)
            self._threshold = self._dims / 2

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))
        if FOUND_OPENMM:
            self.results.units["results.wavenumbers"] = unit.angstrom ** -1

        # Preallocate arrays to store results
        self.results.ssf = np.zeros(len(self._wavenumbers), dtype=float)
        
    def _single_frame(self) -> None:

        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            
            # Store atom or center-of-mass positions in the current frame
            self._positions[s] = g.positions if gr == "atoms" \
                                 else molecule.center_of_mass(g, gr)

            # Unwrap particle positions if necessary
            if self._unwrap:
                dpos = self._positions[s] - self._positions_old[s]
                mask = np.abs(dpos) >= self._threshold
                self._images[s][mask] -= np.sign(dpos[mask]).astype(int)
                self._positions_old[s] = self._positions[s].copy()
                self._positions[s] += self._images[s] * self._dims

        # Compute the static structure factor by squaring the
        # cosine and sine terms and adding them together
        arg = np.einsum("ij,kj->ki", self._wavevectors, self._positions)
        self.results.ssf += np.sin(arg).sum(axis=0) ** 2 \
                            + np.cos(arg).sum(axis=0) ** 2

    def _conclude(self) -> None:

        # Normalize the static structure factor by the number of 
        # particles and timesteps
        self.results.ssf /= self._N * self.n_frames
        
        # Flatten the array by combining values sharing the same
        # wavevector magnitude
        self.results.ssf = np.fromiter(
            (self.results.ssf[np.isclose(q, self._wavenumbers)].mean() 
             for q in self.results.wavenumbers),
            dtype=float, 
            count=len(self.results.wavenumbers)
        )

class ParallelStructureFactor(StructureFactor, ParallelAnalysisBase):

    """
    A multithreaded implementation to calculate the static structure 
    factor :math:`S(q)`.
    
    .. note::
       For a theoretical background and a complete list of parameters,
       attributes, and available methods, see :class:`StructureFactor`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, ArrayLike],
            groupings: Union[str, ArrayLike] = "atoms", n_points: int = 32,
            *, unwrap: bool = False, verbose: bool = True, **kwargs):
        
        StructureFactor.__init__(self, groups, groupings, n_points, 
                                 unwrap=unwrap, verbose=verbose, **kwargs)
        ParallelAnalysisBase.__init__(self, self.universe.trajectory, 
                                      verbose=verbose, **kwargs)
        
    def _prepare(self) -> None:

        # Unwrap particle positions if necessary
        self._positions = np.empty((self.n_frames, self._N, 3), dtype=float)
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0] 
                if hasattr(self._sliced_trajectory, "frames") 
                else (self.start or 0)
            ]
            positions_old = np.empty((self._N, 3), dtype=float)
            for g, gr, s in zip(self._groups, self._groupings, self._slices):
                positions_old[s] = g.positions if gr == "atoms" \
                                   else molecule.center_of_mass(g, gr)
            images = np.zeros(positions_old.shape, dtype=int)
            threshold = self._dims / 2

        # Store particle positions in a shared memory array
        for i, _ in enumerate(self.universe.trajectory[
                list(self._sliced_trajectory.frames) 
                if hasattr(self._sliced_trajectory, "frames")
                else slice(self.start, self.stop, self.step)
            ]):

            for g, gr, s in zip(self._groups, self._groupings, self._slices):

                # Store atom or center-of-mass positions in the current frame
                self._positions[i, s] = g.positions if gr == "atoms" \
                                        else molecule.center_of_mass(g, gr)
                
                # Unwrap particle positions if necessary
                if self._unwrap:
                    dpos = self._positions[i, s] - positions_old[s]
                    mask = np.abs(dpos) >= threshold
                    images[s][mask] -= np.sign(dpos[mask]).astype(int)
                    positions_old[s] = self._positions[i, s].copy()
                    self._positions[i, s] += images[s] * self._dims

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))

    def _single_frame(self, frame: int, timestep: int) -> np.ndarray:

        # Compute the static structure factor by squaring the
        # cosine and sine terms and adding them together
        arg = np.einsum("ij,kj->ki", self._wavevectors, 
                        self._positions[timestep])
        return np.sin(arg).sum(axis=0) ** 2 + np.cos(arg).sum(axis=0) ** 2
    
    def _conclude(self) -> None:

        # Tally static structure factor for each wavevector over all
        # frames and normalize by the number of particles and timesteps
        self.results.ssf = np.vstack(self._results).sum(axis=0) \
                           / (self._N * self.n_frames)
        
        # Flatten the array by combining values sharing the same
        # wavevector magnitude
        self.results.ssf = np.fromiter(
            (self.results.ssf[np.isclose(q, self._wavenumbers)].mean() 
             for q in self.results.wavenumbers),
            dtype=float, 
            count=len(self.results.wavenumbers)
        )