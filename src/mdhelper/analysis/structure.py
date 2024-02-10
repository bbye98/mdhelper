"""
Bulk structural analysis
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to analyze the structure of bulk fluid and
electrolyte systems.
"""

from itertools import combinations_with_replacement
from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
from scipy.integrate import simpson
from scipy.signal import argrelextrema

from .base import SerialAnalysisBase, ParallelAnalysisBase
from .. import FOUND_OPENMM, ureg, Q_
from ..algorithm import correlation, molecule, utility

if FOUND_OPENMM:
    from openmm import unit

def radial_histogram(
        pos1: np.ndarray[float], pos2: np.ndarray[float], n_bins: int, 
        range: tuple[float], dims: tuple[float], *, 
        exclusion: tuple[int] = None) -> np.ndarray[float]:
    
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
    pairs, dist = distances.capped_distance(
        pos1, pos2, range[1], range[0] - np.finfo(np.float64).eps,
        box=dims
    )
    
    # Exclude atom pairs with the same atoms or atoms from the
    # same residue
    if exclusion is not None:
        dist = dist[np.where(pairs[:, 0] // exclusion[0] 
                             != pairs[:, 1] // exclusion[1])[0]]
    
    return np.histogram(dist, bins=n_bins, range=range)[0]

def coordination_numbers(
        bins: np.ndarray[float], rdf: np.ndarray[float], rho: float, *, 
        n: int = 2, threshold: float = 0.1) -> np.ndarray[float]:

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
        r: np.ndarray[float], f: np.ndarray[float], q: np.ndarray[float]
    ) -> np.ndarray[float]:

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
        r: np.ndarray[float], g: np.ndarray[float], equal: bool, rho: float,
        x_i: float = 1, x_j: float = None, q: np.ndarray[float] = None, *, 
        q_lower: float = None, q_upper: float = None, n_q: int = 1000,
        formalism: str = "FZ") -> tuple[np.ndarray[float], np.ndarray[float]]:
    
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
    :math:`j` and its related properties.

    It is given by

    .. math::

       g_{ij}(r)=\frac{1}{N_iN_j}\sum_\alpha\sum_\beta \left\langle
       \delta\left(|\mathbf{r}_\alpha-\mathbf{r}_\beta|-r\right)
       \right\rangle

    where :math:`N_i` and :math:`N_j` are the number of particles, and
    :math:`\mathbf{r}_\alpha` and :math:`\mathbf{r}_\beta` are the 
    positions of particles :math:`\alpha` and :math:`\beta` belonging 
    to species :math:`i` and :math:`j`, respectively. The RDF is 
    normalized such that :math:`\lim_{r\rightarrow\infty}g_{ij}(r)=1` in
    a homogeneous system.

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
        Range of radii values. The upper bound should be less than half
        the largest system dimension.

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

    n_batches : `int`, keyword-only, optional
        Number of batches to divide the histogram calculation into.
        This is useful for large systems that cannot be processed in a
        single pass.

        .. note::

           If you use too few bins and too many batches, the histogram
           counts may be off by a few due to the floating-point nature
           of the cutoffs. However, when the RDF is averaged over a
           long trajectory with many particles, the difference should
           be negligible.

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
        :code:`results.units["results.bins"]`.

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

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: tuple[float] = (0.0, 15.0), *,
            norm: str = "rdf", exclusion: tuple[int] = None,
            groupings: Union[str, tuple[str]] = "atoms",
            reduced: bool = False, n_batches: int = None,
            verbose: bool = True, **kwargs) -> None:

        self.ag1 = ag1
        self.ag2 = ag1 if ag2 is None else ag2
        self.universe = self.ag1.universe
        if self.universe.dimensions is None and self._ts.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        if isinstance(groupings, str):
            if groupings not in {"atoms", "residues", "segments"}:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms', 'residues', and 'segments'.")
                raise ValueError(emsg)
            self._groupings = 2 * [groupings]
        else:
            for g in groupings:
                if g not in {"atoms", "residues", "segments"}:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms', 'residues', and 'segments'.")
                    raise ValueError(emsg)
            self._groupings = 2 * groupings if len(groupings) == 1 else groupings

        self._n_bins = n_bins
        self._range = range
        self._norm = norm
        self._exclusion = exclusion
        self._reduced = reduced
        self._n_batches = n_batches
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to store neighbor counts
        self.results.edges = np.linspace(*self._range, self._n_bins + 1)
        self.results.bins = (self.results.edges[:-1] 
                             + self.results.edges[1:]) / 2
        self.results.counts = np.zeros(self._n_bins, dtype=int)
        self.results.units = {"results.bins": unit.angstrom,
                              "results.edges": unit.angstrom}

        # Preallocate floating-point number for total volume analyzed
        # (for when system dimensions can change, such as during NpT
        # equilibration)
        if self._norm == "rdf":
            self._volume = 0.0

    def _single_frame(self) -> None:

        # Tally counts in each pair separation distance bin
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0]) 
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]), 
                          *((a[-1], b[-1]) 
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            pos1 = self.ag1.positions if self._groupings[0] == "atoms" \
                   else molecule.center_of_mass(self.ag1, self._groupings[0])
            pos2 = self.ag2.positions if self._groupings[1] == "atoms" \
                   else molecule.center_of_mass(self.ag2, self._groupings[1])
            for r, i in ranges_indices.items():
                self.results.counts[i] += radial_histogram(
                    pos1=pos1, pos2=pos2, n_bins=i.shape[0], range=r, 
                    dims=self._ts.dimensions, exclusion=self._exclusion
                )
        else:
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

    def _get_rdf(self) -> np.ndarray[float]:

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

    def calculate_coordination_numbers(
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

        self.results.coordination_numbers = coordination_numbers(
            self.results.bins, self._get_rdf(), rho, n=n, threshold=threshold
        )

    def calculate_pmf(self, temp: Union[float, "unit.Quantity", Q_]) -> None:

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

        self.results.units = {"results.pmf": ureg.kilojoule / ureg.mole}

        if self._reduced:
            if isinstance(temp, (int, np.integer, float, np.floating)):
                kBT = temp
            else:
                emsg = ("'temp' has units, but the rest of the data is "
                        "or should be reduced.")
                raise ValueError(emsg)
        else:
            if isinstance(temp, (int, np.integer, float, np.floating)):
                temp *= ureg.kelvin
            elif temp.__module__ == "openmm.unit.quantity":
                temp = temp.value_in_unit(unit.kelvin) * ureg.kelvin
            kBT = (ureg.avogadro_constant * ureg.boltzmann_constant 
                   * temp).m_as(self.results.units["results.pmf"])

        self.results.pmf = -kBT * np.log(self._get_rdf())

    def calculate_structure_factor(
            self, rho: float, x_i: float = None, x_j: float = None, 
            q: np.ndarray[float] = None, *, q_lower: float = None, 
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
    function (RDF) :math:`g_{ij}(r)` between types :math:`i` and 
    :math:`j` and its related properties.
    
    .. note::
    
       For a theoretical background and a complete list of
       parameters, attributes, and available methods, see :class:`RDF`.
    """

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: tuple[float] = (0.0, 15.0), *,
            norm: str = "rdf", exclusion: tuple[int] = None, 
            groupings: Union[str, tuple[str]] = "atoms",
            reduced: bool = False, n_batches: int = None, 
            verbose: bool = True, **kwargs) -> None:

        RDF.__init__(self, ag1, ag2, n_bins, range, norm=norm, 
                     exclusion=exclusion, groupings=groupings, reduced=reduced,
                     n_batches=n_batches, verbose=verbose, **kwargs)
        ParallelAnalysisBase.__init__(self, ag1.universe.trajectory, 
                                      verbose=verbose, **kwargs)

    def _single_frame(self, frame: int, index: int) -> np.ndarray[float]:

        _ts = self._trajectory[frame]
        result = np.empty(1 + self._n_bins, dtype=float)

        # Compute radial histogram for a single frame
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0]) 
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]), 
                          *((a[-1], b[-1]) 
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            pos1 = self.ag1.positions if self._groupings[0] == "atoms" \
                   else molecule.center_of_mass(self.ag1, self._groupings[0])
            pos2 = self.ag2.positions if self._groupings[1] == "atoms" \
                   else molecule.center_of_mass(self.ag2, self._groupings[1])
            for r, i in ranges_indices.items():
                result[i] = radial_histogram(pos1=pos1, pos2=pos2, 
                                             n_bins=i.shape[0],
                                             range=r, dims=_ts.dimensions,
                                             exclusion=self._exclusion)
        else:
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
        
class RDF2D(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the two-dimensional radial
    distribution function (RDF) :math:`g_{ij}(r)` between types 
    :math:`i` and :math:`j` and its related properties.

    It is given by

    .. math::

       g_{ij}(r)=\frac{1}{N_iN_j}\sum_\alpha\sum_\beta \left\langle
       \delta\left(|\mathbf{r}_\alpha-\mathbf{r}_\beta|-r\right)
       \right\rangle

    where :math:`N_i` and :math:`N_j` are the number of particles, and
    :math:`\mathbf{r}_\alpha` and :math:`\mathbf{r}_\beta` are the positions 
    of particles :math:`\alpha` and :math:`\beta` belonging to 
    species :math:`i` and :math:`j`, respectively. The RDF is normalized
    such that :math:`\lim_{r\rightarrow\infty}g_{ij}(r)=1` in a 
    homogeneous system.

    (A closely related quantity is the single particle density 
    :math:`n_{ij}(r)=\rho_jg_{ij}(r)`, where :math:`\rho_j` is the
    number density of species :math:`j`.)

    The RDF can also be used to obtain the potential of mean force

    .. math::

       w_{ij}(r)=-k_\mathrm{B}T\ln{g_{ij}(r)}

    where :math:`k_\mathrm{B}` is the Boltzmann constant and :math:`T`
    is the system temperature.

    Parameters
    ----------
    ag1 : `MDAnalysis.AtomGroup`
        First atom group :math:`i`.

    ag2 : `MDAnalysis.AtomGroup`
        Second atom group :math:`j`.
    
    n_bins : `int`, default: :code:`201`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range : array-like, default: :code:`(0.0, 15.0)`
        Range of radii values. The upper bound should be less than half
        the largest system dimension.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    drop_axis : `int` or `str`, keyword-only, default: :code:`2`
        Axis in three-dimensional space to ignore in the two-dimensional
        analysis.

        **Valid values**: :code:`0` or :code:`x` for the :math:`x`-axis,
        :code:`1` or :code:`y` for the :math:`y`-axis, and :code:`2` or
        :code:`z` for the :math:`z`-axis.

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

    n_batches : `int`, keyword-only, optional
        Number of batches to divide the histogram calculation into.
        This is useful for large systems that cannot be processed in a
        single pass.

        .. note::

           If you use too few bins and too many batches, the histogram
           counts may be off by a few due to the floating-point nature
           of the cutoffs. However, when the RDF is averaged over a
           long trajectory with many particles, the difference should
           be negligible.

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
        :code:`results.units["results.bins"]`.

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

    results.pmf : `numpy.ndarray`
        Potential of mean force :math:`w(r)`. Only available after
        running :meth:`calculate_pmf`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{kJ/mol}`.
    """

    # TODO: Combine with RDF class.

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: tuple[float] = (0.0, 15.0), *,
            drop_axis: Union[int, str] = 2, norm: str = "rdf", 
            exclusion: tuple[int] = None, 
            groupings: Union[str, tuple[str]] = "atoms",
            reduced: bool = False, n_batches: int = None,
            verbose: bool = True, **kwargs) -> None:

        self.ag1 = ag1
        self.ag2 = ag1 if ag2 is None else ag2
        self.universe = self.ag1.universe
        if self.universe.dimensions is None and self._ts.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        if isinstance(groupings, str):
            if groupings not in {"atoms", "residues", "segments"}:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms', 'residues', and 'segments'.")
                raise ValueError(emsg)
            self._groupings = 2 * [groupings]
        else:
            for g in groupings:
                if g not in {"atoms", "residues", "segments"}:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms', 'residues', and 'segments'.")
                    raise ValueError(emsg)
            self._groupings = 2 * groupings if len(groupings) == 1 else groupings

        self._drop_axis = (ord(drop_axis) - 120 if isinstance(drop_axis, str) 
                           else drop_axis)
        self._n_bins = n_bins
        self._range = range
        self._norm = norm
        self._exclusion = exclusion
        self._reduced = reduced
        self._n_batches = n_batches
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to store neighbor counts
        self.results.edges = np.linspace(*self._range, self._n_bins + 1)
        self.results.bins = (self.results.edges[:-1] 
                             + self.results.edges[1:]) / 2
        self.results.counts = np.zeros(self._n_bins, dtype=int)
        if not self._reduced:
            self.results.units = {"results.bins": unit.angstrom,
                                  "results.edges": unit.angstrom}

        # Preallocate floating-point number for total area analyzed
        # (for when system dimensions can change, such as during NpT
        # equilibration)
        if self._norm == "rdf":
            self._area = 0.0

    def _single_frame(self) -> None:

        dims = self._ts.dimensions
        pos1 = self.ag1.positions if self._groupings[0] == "atoms" \
                else molecule.center_of_mass(self.ag1, self._groupings[0])
        pos2 = self.ag2.positions if self._groupings[1] == "atoms" \
                else molecule.center_of_mass(self.ag2, self._groupings[1])
        
        # Apply corrections to avoid including periodic images in the
        # dimension to exclude
        pos1[:, self._drop_axis] = pos2[:, self._drop_axis] = 0
        dims[self._drop_axis] = dims[:3].max()

        # Tally counts in each pair separation distance bin
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0]) 
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]), 
                          *((a[-1], b[-1]) 
                            for a, b in zip(edges[:-1], edges[1:]))]
            }

            for r, i in ranges_indices.items():
                self.results.counts[i] += radial_histogram(
                    pos1=pos1, pos2=pos2, n_bins=i.shape[0], range=r, 
                    dims=dims, exclusion=self._exclusion
                )
        else:
            self.results.counts += radial_histogram(
                pos1=pos1, pos2=pos2, n_bins=self._n_bins, range=self._range,
                dims=dims, exclusion=self._exclusion
            )
        
        # Add area analyzed
        if self._norm == "rdf":
            self._area += np.delete(dims[:3], self._drop_axis).prod() 
            
    def _conclude(self):

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            norm *= np.pi * np.diff(self.results.edges ** 2)
            if self._norm == "rdf":
                _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self.ag1, f"n_{self._groupings[0]}") * _N2
                         * self.n_frames / self._area)
        
        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm

    def _get_rdf(self) -> np.ndarray[float]:

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
            return self._area * self.results.counts / (
                np.pi * self.n_frames ** 2 * _N2
                * getattr(self.ag1, f"n_{self._groupings[0]}") 
                * np.diff(self.results.edges ** 2)
            )

    def calculate_coordination_numbers(
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

        self.results.coordination_numbers = coordination_numbers(
            self.results.bins, self._get_rdf(), rho, n=n, threshold=threshold
        )

    def calculate_pmf(self, temp: Union[float, "unit.Quantity", Q_]) -> None:

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

        self.results.units = {"results.pmf": ureg.kilojoule / ureg.mole}

        if self._reduced:
            if isinstance(temp, (int, np.integer, float, np.floating)):
                kBT = temp
            else:
                emsg = ("'temp' has units, but the rest of the data is "
                        "or should be reduced.")
                raise ValueError(emsg)
        else:
            if isinstance(temp, (int, np.integer, float, np.floating)):
                temp *= ureg.kelvin
            elif temp.__module__ == "openmm.unit.quantity":
                temp = temp.value_in_unit(unit.kelvin) * ureg.kelvin
            kBT = (ureg.avogadro_constant * ureg.boltzmann_constant 
                   * temp).m_as(self.results.units["results.pmf"])

        self.results.pmf = -kBT * np.log(self._get_rdf())

class ParallelRDF2D(RDF2D, ParallelAnalysisBase):

    """
    A multithreaded implementation to calculate the two-dimensional 
    radial distribution function (RDF) :math:`g_{ij}(r)` between types
    :math:`i` and :math:`j` and its related properties.
    
    .. note::
       For a theoretical background and a complete list of
       parameters, attributes, and available methods, see :class:`RDF2D`.
    """

    # TODO: Combine with ParallelRDF class.

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: tuple[float] = (0.0, 15.0), *,
            drop_axis: Union[int, str] = 2, norm: str = "rdf", 
            exclusion: tuple[int] = None, 
            groupings: Union[str, tuple[str]] = "atoms",
            reduced: bool = False, n_batches: int = None,
            verbose: bool = True, **kwargs) -> None:
        
        RDF2D.__init__(self, ag1, ag2, n_bins, range, drop_axis=drop_axis,
                       norm=norm, exclusion=exclusion, groupings=groupings,
                       reduced=reduced, n_batches=n_batches, verbose=verbose,
                       **kwargs)
        ParallelAnalysisBase.__init__(self, ag1.universe.trajectory,
                                      verbose=verbose, **kwargs)
        
    def _single_frame(self, frame: int, index: int) -> np.ndarray[float]:

        _ts = self._trajectory[frame]
        result = np.empty(1 + self._n_bins, dtype=float)

        dims = _ts.dimensions
        pos1 = self.ag1.positions if self._groupings[0] == "atoms" \
               else molecule.center_of_mass(self.ag1, self._groupings[0])
        pos2 = self.ag2.positions if self._groupings[1] == "atoms" \
               else molecule.center_of_mass(self.ag2, self._groupings[1])
        
        # Apply corrections to avoid including periodic images in the
        # dimension to exclude
        pos1[:, self._drop_axis] = pos2[:, self._drop_axis] = 0
        dims[self._drop_axis] = dims[:3].max()

        # Compute radial histogram for a single frame
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0]) 
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]), 
                          *((a[-1], b[-1]) 
                            for a, b in zip(edges[:-1], edges[1:]))]
            }

            for r, i in ranges_indices.items():
                result[i] = radial_histogram(
                    pos1=pos1, pos2=pos2, n_bins=i.shape[0], range=r, 
                    dims=dims, exclusion=self._exclusion
                )
        else:
            result[:self._n_bins] = radial_histogram(
                pos1=pos1, pos2=pos2, n_bins=self._n_bins, range=self._range,
                dims=dims, exclusion=self._exclusion
            )
        
        # Store area analyzed in the current frame in the last slot of
        # the results array
        result[self._n_bins] = np.delete(dims[:3], self._drop_axis).prod()

        return result

    def _conclude(self):

        # Tally counts in each pair separation distance bin over all
        # frames
        self._results = np.vstack(self._results).sum(axis=0)
        self.results.counts[:] = self._results[:self._n_bins]
        self._area = self._results[self._n_bins]

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            norm *= np.pi * np.diff(self.results.edges ** 2)
            if self._norm == "rdf":
                _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self.ag1, f"n_{self._groupings[0]}") * _N2
                         * self.n_frames / self._area)
        
        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm

class StructureFactor(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the static structure factor
    :math:`S(q)` or partial structure factor :math:`S_{\alpha\beta}(q)`
    for species :math:`\alpha` and :math:`\beta`.

    The static structure factor is a measure of how a material scatters
    incident radiation, and can be computed directly from a molecular 
    dynamics trajectory using

    .. math::

        S(\mathbf{q})&=\frac{1}{N}\left\langle\sum_{j=1}^N\sum_{k=1}^N
        \exp{[-i\mathbf{q}\cdot(\mathbf{r}_j-\mathbf{r}_k)]}\right\rangle\\
        &=\frac{1}{N}\left\langle\left[
        \sum_{j=1}^N\sin{(\mathbf{q}\cdot\mathbf{r}_j)}\right]^2+\left[
        \sum_{j=1}^N\cos{(\mathbf{q}\cdot\mathbf{r}_j)}\right]^2\right\rangle

    where :math:`N` is the number of particles, :math:`\mathbf{q}` is
    the scattering wavevector, and :math:`\mathbf{r}_i` is the position 
    of the :math:`i`-th particle.

    For multicomponent systems, the equation above can be generalized to
    get the partial structure factor

    .. math::

       S_{\alpha\beta}(\mathbf{q})=\frac{1}{\sqrt{N_\alpha N_\beta}}
       \left\langle\sum_{j=1}^{N_\alpha}\cos{(\mathbf{q}\cdot\mathbf{r}_j)}
       \sum_{k=1}^{N_\beta}\cos{(\mathbf{q}\cdot\mathbf{r}_k)}
       +\sum_{j=1}^{N_\alpha}\sin{(\mathbf{q}\cdot\mathbf{r}_j)}
       \sum_{k=1}^{N_\beta}\sin{(\mathbf{q}\cdot\mathbf{r}_k)}\right\rangle

    where :math:`N_\alpha` and :math:`N_\beta` are the numbers of
    particles for species :math:`\alpha` and :math:`\beta`.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms that share the same grouping type. If `mode`
        is not specified, all atoms in the universe must be assigned to
        a group. If :code:`mode="pair"`, there must be exactly one or 
        two groups.

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
        Number of points in the scattering wavevector grid. Additional
        wavevectors can be introduced via `n_surfaces` and 
        `n_surface_points` for more accurate structure factor values.
        Alternatively, the desired wavevectors can be specified directly
        in `wavevectors`.

    mode : `str`, optional
        Evaluation mode.

        .. container::

           **Valid values**:

           * :code:`None`: The overall structure factor is computed.
           * :code:`"pair"`: The partial structure factor is computed
             between the group(s) in `groups`.
           * :code:`"partial"`: The partial structure factors for all 
             unique pairs from `groups` is computed.

    n_surfaces : `int`, keyword-only, optional
        Number of spherical surfaces in the first octant that intersect 
        with the grid wavevectors along the three coordinate axes for
        which to introduce extra wavevectors for more accurate structure
        factor values. Only available if the system is perfectly cubic.

    n_surface_points : `int`, keyword-only, default: :code:`8`
        Number of extra wavevectors to introduce per spherical surface. 
        Has no effect if `n_surfaces` is not specified.

    wavevectors : `numpy.ndarray`, keyword-only, optional
        Scattering wavevectors for which to compute the structure factor.
        Has precedence over `n_points` if specified.

    n_batches : `int`, keyword-only, optional
        Number of batches to divide the structure factor calculation 
        into. This is useful for large systems that cannot be processed 
        in a single pass.

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
        :code:`results.units["results.wavenumbers"]`.

    results.pairs : `tuple`
        All unique pairs of indices of the groups of atoms in `groups`.
        The ordering coincides with the column indices in
        `results.ssf`.

    results.wavenumbers : `numpy.ndarray`
        :math:`N_\mathrm{w}` unique scattering wavenumbers :math:`q`.

        **Shape**: :math:`(N_\mathrm{w},)`.

    results.ssf : `numpy.ndarray`
        Static structure factor :math:`S(q)` or partial structure
        factor(s) :math:`S_{\alpha\beta}(q)`.

        **Shape**: :math:`(N_\mathrm{w},)`, :math:`(1,\,N_\mathrm{w})`,
        or :math:`(C(N_\mathrm{g}+1,\,2),\,N_\mathrm{w})`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", n_points: int = 32,
            mode: str = None, *, n_surfaces: int = None, 
            n_surface_points: int = 8, wavevectors: np.ndarray[float] = None, 
            n_batches: int = None, verbose: bool = True, **kwargs):

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe

        self._mode = mode
        if self._mode == "pair" and not 1 <= len(self._groups) <= 2:
            emsg = "There must be exactly one or two groups when mode='pair'."
            raise ValueError(emsg)
        elif self._mode is None:
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

        # Determine the wavevectors and their corresponding magnitudes
        if wavevectors is not None:
            self._wavevectors = wavevectors
        elif np.allclose(self._dims, self._dims[0]):
            grid = 2 * np.pi * np.arange(n_points) / self._dims[0]
            self._wavevectors = np.stack(
                np.meshgrid(grid, grid, grid), -1
            ).reshape(-1, 3)
            if n_surfaces:
                n_theta, n_phi = utility.closest_factors(n_surface_points, 2, 
                                                         reverse=True)
                theta = np.linspace(np.pi / (2 * n_theta + 4), 
                                    np.pi / 2 - np.pi / (2 * n_theta + 4), 
                                    n_theta)
                phi = np.linspace(np.pi / (2 * n_phi + 4), 
                                  np.pi / 2 - np.pi / (2 * n_phi + 4), 
                                  n_phi)
                self._wavevectors = np.vstack((
                    self._wavevectors,
                    np.einsum(
                        "o,tpd->otpd", 
                        grid[1:n_surfaces + 1], 
                        np.stack((
                            np.sin(theta) * np.cos(phi)[:, None],
                            np.sin(theta) * np.sin(phi)[:, None],
                            np.tile(np.cos(theta)[None, :], (n_phi, 1))
                        ), axis=-1)
                    ).reshape((n_surfaces * n_surface_points, 3))
                ))
        else:
            self._wavevectors = np.stack(
                np.meshgrid(
                    *[2 * np.pi * np.arange(n_points) / L
                      for L in self._dims]
                ), -1
            ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        self._n_batches = n_batches
        self._verbose = verbose

    def _prepare(self) -> None:

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))

        # Preallocate arrays to store results
        self._positions = np.empty((self._N, 3), dtype=float)
        if self._mode is None:
            self.results.ssf = np.zeros(len(self._wavenumbers), dtype=float)
        else:
            self.results.pairs = tuple(
                combinations_with_replacement(range(self._n_groups), 2)
            ) if self._mode == "partial" else ((0, self._n_groups - 1),)
            self.results.ssf = np.zeros(
                (len(self.results.pairs), len(self._wavenumbers)), 
                dtype=float
            )

        # Create dictionary to hold reference units
        self.results.units = {"_dims": unit.angstrom,
                              "results.wavenumbers": unit.angstrom ** -1}
        
    def _single_frame(self) -> None:

        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            
            # Store atom or center-of-mass positions in the current frame
            self._positions[s] = g.positions if gr == "atoms" \
                                 else molecule.center_of_mass(g, gr)

        # Compute the structure factor by multiplying the cosine and
        # sine terms and adding them together
        if self._mode is None:
            if self._n_batches:
                start = 0
                for w in np.array_split(self._wavevectors, self._n_batches, 
                                        axis=0):
                    arg = np.einsum("wd,pd->pw", w, self._positions)
                    self.results.ssf[start:start + w.shape[0]] += \
                        np.sin(arg).sum(axis=0) ** 2 \
                        + np.cos(arg).sum(axis=0) ** 2
                    start += w.shape[0]
            else:
                arg = np.einsum("wd,pd->pw", 
                                self._wavevectors, self._positions)
                self.results.ssf += np.sin(arg).sum(axis=0) ** 2 \
                                    + np.cos(arg).sum(axis=0) ** 2
        else:
            for i, (j, k) in enumerate(self.results.pairs):
                if self._n_batches:
                    start = 0
                    for w in np.array_split(self._wavevectors, self._n_batches, 
                                            axis=0):
                        arg_j = np.einsum("wd,pd->pw", w, 
                                          self._positions[self._slices[j]])
                        if j == k:
                            self.results.ssf[i, start:start + w.shape[0]] += \
                                np.sin(arg_j).sum(axis=0) ** 2 \
                                + np.cos(arg_j).sum(axis=0) ** 2
                        else:
                            arg_k = np.einsum("wd,pd->pw", w, 
                                              self._positions[self._slices[k]])
                            self.results.ssf[i, start:start + w.shape[0]] += (
                                np.sin(arg_j).sum(axis=0) 
                                * np.sin(arg_k).sum(axis=0)
                                + np.cos(arg_j).sum(axis=0) 
                                * np.cos(arg_k).sum(axis=0)
                            )
                        start += w.shape[0]
                else:
                    arg_j = np.einsum("wd,pd->pw", self._wavevectors, 
                                      self._positions[self._slices[j]])
                    if j == k:
                        self.results.ssf[i] += np.sin(arg_j).sum(axis=0) ** 2 \
                                               + np.cos(arg_j).sum(axis=0) ** 2
                    else:
                        arg_k = np.einsum("wd,pd->pw", self._wavevectors, 
                                        self._positions[self._slices[k]])
                        self.results.ssf[i] += (
                            np.sin(arg_j).sum(axis=0) 
                            * np.sin(arg_k).sum(axis=0)
                            + np.cos(arg_j).sum(axis=0) 
                            * np.cos(arg_k).sum(axis=0)
                        )

    def _conclude(self) -> None:

        # Normalize the static structure factor by the number of 
        # particles and timesteps, and flatten the array by combining
        # values sharing the same wavevector magnitude
        if self._mode is None:
            self.results.ssf /= self.n_frames * self._N
            self.results.ssf = np.fromiter(
                (self.results.ssf[np.isclose(q, self._wavenumbers)].mean() 
                 for q in self.results.wavenumbers),
                dtype=float, 
                count=len(self.results.wavenumbers)
            )
        else:
            self.results.ssf /= (
                self.n_frames *
                np.fromiter((np.sqrt(self._Ns[i] * self._Ns[j])
                             for i, j in self.results.pairs),
                            dtype=float, 
                            count=len(self.results.pairs))[:, None]
            )
            self.results.ssf = np.hstack(
                [self.results.ssf[:, np.isclose(q, self._wavenumbers)]
                 .mean(axis=1, keepdims=True) 
                 for q in self.results.wavenumbers]
            )

class ParallelStructureFactor(StructureFactor, ParallelAnalysisBase):

    r"""
    A multithreaded implementation to calculate the static structure 
    factor :math:`S(q)` or partial structure factor 
    :math:`S_{\alpha\beta}(q)` for species :math:`\alpha` and 
    :math:`\beta`.
    
    .. note::

       For a theoretical background and a complete list of parameters,
       attributes, and available methods, see :class:`StructureFactor`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", n_points: int = 32,
            mode: str = None, *, n_surfaces: int = None, 
            n_surface_points: int = 8, wavevectors: np.ndarray[float] = None, 
            n_batches: int = None, verbose: bool = True, **kwargs):
        
        StructureFactor.__init__(self, groups, groupings, n_points, mode,
                                 n_surfaces=n_surfaces,
                                 n_surface_points=n_surface_points, 
                                 wavevectors=wavevectors, n_batches=n_batches,
                                 verbose=verbose, **kwargs)
        ParallelAnalysisBase.__init__(self, self.universe.trajectory, 
                                      verbose=verbose, **kwargs)
        
    def _prepare(self) -> None:
               
        # Determine pairs
        if self._mode is not None:
            self.results.pairs = tuple(
                combinations_with_replacement(range(self._n_groups), 2)
            ) if self._mode == "partial" else ((0, self._n_groups - 1),)

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))

        # Create dictionary to hold reference units
        self.results.units = {"_dims": unit.angstrom,
                              "results.wavenumbers": unit.angstrom ** -1}

    def _single_frame(self, frame: int, index: int) -> np.ndarray[float]:

        self._trajectory[frame]
        positions = np.empty((self._N, 3), dtype=float)
        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            
            # Store atom or center-of-mass positions in the current frame
            positions[s] = g.positions if gr == "atoms" \
                           else molecule.center_of_mass(g, gr)

        # Compute the structure factor by multiplying the cosine and
        # sine terms and adding them together
        if self._mode is None:
            if self._n_batches:
                start = 0
                ssf = np.empty(len(self._wavenumbers), dtype=float)
                for w in np.array_split(self._wavevectors, self._n_batches, 
                                        axis=0):
                    arg = np.einsum("wd,pd->pw", w, positions)
                    ssf[start:start + w.shape[0]] = \
                        np.sin(arg).sum(axis=0) ** 2 \
                        + np.cos(arg).sum(axis=0) ** 2
                    start += w.shape[0]
                return ssf
            else:
                arg = np.einsum("wd,pd->pw", self._wavevectors, positions)
                return np.sin(arg).sum(axis=0) ** 2 \
                       + np.cos(arg).sum(axis=0) ** 2
        else:
            ssf = np.empty((len(self.results.pairs), 
                            len(self._wavenumbers)), dtype=float)
            for i, (j, k) in enumerate(self.results.pairs):
                if self._n_batches:
                    start = 0
                    for w in np.array_split(self._wavevectors, self._n_batches, 
                                            axis=0):
                        arg_j = np.einsum(
                            "wd,pd->pw", w, positions[self._slices[j]]
                        )
                        if j == k:
                            ssf[i, start:start + w.shape[0]] = \
                                np.sin(arg_j).sum(axis=0) ** 2 \
                                + np.cos(arg_j).sum(axis=0) ** 2
                        else:
                            arg_k = np.einsum(
                                "wd,pd->pw", w, positions[self._slices[k]]
                            )
                            ssf[i, start:start + w.shape[0]] = (
                                np.sin(arg_j).sum(axis=0) 
                                * np.sin(arg_k).sum(axis=0)
                                + np.cos(arg_j).sum(axis=0) 
                                * np.cos(arg_k).sum(axis=0)
                            )
                        start += w.shape[0]
                else:
                    arg_j = np.einsum("wd,pd->pw", self._wavevectors, 
                                      positions[self._slices[j]])
                    if j == k:
                        ssf[i] = np.sin(arg_j).sum(axis=0) ** 2 \
                                 + np.cos(arg_j).sum(axis=0) ** 2
                    else:
                        arg_k = np.einsum("wd,pd->pw", self._wavevectors, 
                                          positions[self._slices[k]])
                        ssf[i] = (
                            np.sin(arg_j).sum(axis=0) 
                            * np.sin(arg_k).sum(axis=0)
                            + np.cos(arg_j).sum(axis=0) 
                            * np.cos(arg_k).sum(axis=0)
                        )
            return ssf
    
    def _conclude(self) -> None:

        # Normalize the static structure factor by the number of 
        # particles and timesteps, and flatten the array by combining
        # values sharing the same wavevector magnitude
        if self._mode is None:
            self.results.ssf = np.vstack(self._results).sum(axis=0) \
                               / (self.n_frames * self._N)
            self.results.ssf = np.fromiter(
                (self.results.ssf[np.isclose(q, self._wavenumbers)].mean() 
                 for q in self.results.wavenumbers),
                dtype=float, 
                count=len(self.results.wavenumbers)
            )
        else:    
            self.results.ssf = np.stack(self._results).sum(axis=0) / (
                self.n_frames *
                np.fromiter((np.sqrt(self._Ns[i] * self._Ns[j])
                             for i, j in self.results.pairs),
                            dtype=float, 
                            count=len(self.results.pairs))[:, None]
            )
            self.results.ssf = np.hstack(
                [self.results.ssf[:, np.isclose(q, self._wavenumbers)]
                 .mean(axis=1, keepdims=True) 
                 for q in self.results.wavenumbers]
            )

class IncoherentIntermediateScatteringFunction(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the incoherent (or self)
    intermediate scattering function :math:`F_\mathrm{s}(q,\,t)`.

    The incoherent intermediate scattering function characterizes the 
    mean relaxation time of a system, and its spatial fluctuations 
    provide information about dynamic heterogeneities. It is defined as

    .. math::

        F_\mathrm{s}(\mathbf{q},t)=\frac{1}{N}\left\langle\sum_{j=1}^N
        \exp\left[i\mathbf{q}\cdot\left(\mathbf{r}_j(t_0+t)
        -\mathbf{r}_j(t_0)\right)\right]\right\rangle
    
    where :math:`N` is the number of particles, :math:`\mathbf{q}` is
    the wavevector, :math:`t_0` and :math:`t` are the initial and lag 
    times, and :math:`\mathbf{r}_j` is the position of particle 
    :math:`j`.

    .. note::

       The simulation must have been run with a constant timestep
       :math:`\Delta t` and the frames to be analyzed must be evenly
       spaced and proceed forward in time for this analysis module to
       function correctly.

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
        Number of points in the scattering wavevector grid. Additional
        wavevectors can be introduced via `n_surfaces` and 
        `n_surface_points` for more accurate structure factor values.
        Alternatively, the desired wavevectors can be specified directly
        in `wavevectors`.

    n_surfaces : `int`, keyword-only, optional
        Number of spherical surfaces in the first octant that intersect 
        with the grid wavevectors along the three coordinate axes for
        which to introduce extra wavevectors for more accurate incoherent 
        scattering function values. Only available if the system is 
        perfectly cubic.

    n_surface_points : `int`, keyword-only, default: :code:`8`
        Number of extra wavevectors to introduce per spherical surface. 
        Has no effect if `n_surfaces` is not specified.

    wavevectors : `numpy.ndarray`, keyword-only, optional
        Scattering wavevectors for which to compute the incoherent 
        scattering function. Has precedence over `n_points` if 
        specified.

    n_batches : `int`, keyword-only, optional
        Number of batches to divide the incoherent scattering function 
        calculation into. This is useful for large systems that cannot 
        be processed in a single pass.

    fft : `bool`, keyword-only, default: :code:`True`
        Determines whether fast Fourier transforms (FFT) are used to 
        evaluate the auto- and cross-correlations.

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
        :code:`results.units["results.wavenumbers"]`.

    results.wavenumbers : `numpy.ndarray`
        Scattering wavenumbers :math:`q`.

    results.isf : `numpy.ndarray`
        Incoherent (self) intermediate scattering function
        :math:`F_\mathrm{s}(q,\,t)`.

        **Shape**: Same as `results.wavenumbers`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", n_points: int = 32,
            *, n_surfaces: int = None, n_surface_points: int = 8, 
            wavevectors: np.ndarray[float] = None, n_batches: int = None, 
            fft: bool = True, verbose: bool = True, **kwargs):
        
        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        
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

        if wavevectors is not None:
            self._wavevectors = wavevectors
        elif np.allclose(self._dims, self._dims[0]):
            grid = 2 * np.pi * np.arange(n_points) / self._dims[0]
            self._wavevectors = np.stack(
                np.meshgrid(grid, grid, grid), -1
            ).reshape(-1, 3)
            if n_surfaces:
                n_theta, n_phi = utility.closest_factors(n_surface_points, 2, 
                                                         reverse=True)
                theta = np.linspace(np.pi / (2 * n_theta + 4), 
                                    np.pi / 2 - np.pi / (2 * n_theta + 4), 
                                    n_theta)
                phi = np.linspace(np.pi / (2 * n_phi + 4), 
                                  np.pi / 2 - np.pi / (2 * n_phi + 4), 
                                  n_phi)
                self._wavevectors = np.vstack((
                    self._wavevectors,
                    np.einsum(
                        "o,tpd->otpd", 
                        grid[1:n_surfaces + 1], 
                        np.stack((
                            np.sin(theta) * np.cos(phi)[:, None],
                            np.sin(theta) * np.sin(phi)[:, None],
                            np.tile(np.cos(theta)[None, :], (n_phi, 1))
                        ), axis=-1)
                    ).reshape((n_surfaces * n_surface_points, 3))
                ))
        else:
            self._wavevectors = np.stack(
                np.meshgrid(
                    *[2 * np.pi * np.arange(n_points) / L
                      for L in self._dims]
                ), -1
            ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        self._n_batches = n_batches
        self._fft = fft
        self._verbose = verbose

    def _prepare(self) -> None:

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            df = np.diff(self._sliced_trajectory.frames)
            if df[0] <= 0 or not np.allclose(df, df[0]):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
        elif hasattr(self._sliced_trajectory, "step") \
                and self._sliced_trajectory.step <= 0:
            raise ValueError("The analysis must proceed forward in time.")

        # Determine the unique wavenumbers and store the lag times
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))
        self.results.time = self._trajectory.dt * np.arange(self.n_frames)

        # Preallocate arrays to store results
        self._positions = np.empty((self._N, 3), dtype=float)
        self._cos_sum = np.zeros((self.n_frames, len(self._wavenumbers)), dtype=float)
        self._sin_sum = np.zeros((self.n_frames, len(self._wavenumbers)), dtype=float)

        # Create dictionary to hold reference units
        self.results.units = {"_dims": unit.angstrom,
                              "results.time": unit.picosecond,
                              "results.wavenumbers": unit.angstrom ** -1}

    def _single_frame(self) -> None:

        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            
            # Store atom or center-of-mass positions in the current frame
            self._positions[s] = g.positions if gr == "atoms" \
                                 else molecule.center_of_mass(g, gr)

        # Compute the sum of cosines and sines of the dot product of the
        # wavevectors and positions
        if self._n_batches:
            start = 0
            for w in np.array_split(self._wavevectors, self._n_batches, axis=0):
                arg = np.einsum("wd,pd->pw", w, self._positions)
                self._cos_sum[self._frame_index, start:start + w.shape[0]] \
                    = np.cos(arg).sum(axis=0)
                self._sin_sum[self._frame_index, start:start + w.shape[0]] \
                    = np.sin(arg).sum(axis=0)
                start += w.shape[0]
        else:
            arg = np.einsum("wd,pd->pw", self._wavevectors, self._positions)
            self._cos_sum[self._frame_index] = np.cos(arg).sum(axis=0)
            self._sin_sum[self._frame_index] = np.sin(arg).sum(axis=0)
    
    def _conclude(self) -> None:

        # Tally intermediate scattering function for each wavevector 
        # over all frames and normalize by the number of particles and 
        # timesteps
        corr = correlation.correlation_fft if self._fft \
               else correlation.correlation_shift
        self.results.isf = (
            corr(self._cos_sum, axis=0) + corr(self._sin_sum, axis=0)
        ) / self._N
        
        # Flatten the array by combining values sharing the same
        # wavevector magnitude
        self.results.isf = np.hstack(
            [
                self.results.isf[
                    :, np.isclose(q, self._wavenumbers)
                ].mean(axis=1, keepdims=True)
                for q in self.results.wavenumbers
            ]
        )

class ParallelIncoherentIntermediateScatteringFunction(
        IncoherentIntermediateScatteringFunction, ParallelAnalysisBase
    ):

    r"""
    A multithreaded implementation to calculate the incoherent (self)
    intermediate scattering function :math:`F_\mathrm{s}(q,\,t)`.
    
    .. note::

       For a theoretical background and a complete list of parameters,
       attributes, and available methods, see 
       :class:`IncoherentIntermediateScatteringFunction`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", n_points: int = 32,
            *, n_surfaces: int = None, n_surface_points: int = 8, 
            wavevectors: np.ndarray[float] = None, n_batches: int = None, 
            fft: bool = True, verbose: bool = True, **kwargs):
        
        IncoherentIntermediateScatteringFunction.__init__(
            self, groups, groupings, n_points, n_surfaces=n_surfaces,
            n_surface_points=n_surface_points, wavevectors=wavevectors, 
            n_batches=n_batches, fft=fft, verbose=verbose, **kwargs)
        ParallelAnalysisBase.__init__(self, self.universe.trajectory, 
                                      verbose=verbose, **kwargs)
        
    def _prepare(self) -> None:

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            df = np.diff(self._sliced_trajectory.frames)
            if df[0] <= 0 or not np.allclose(df, df[0]):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
        elif hasattr(self._sliced_trajectory, "step") \
                and self._sliced_trajectory.step <= 0:
            raise ValueError("The analysis must proceed forward in time.")

        # Determine the unique wavenumbers and store the lag times
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))
        self.results.time = self._trajectory.dt * np.arange(self.n_frames)

        # Create dictionary to hold reference units
        self.results.units = {"_dims": unit.angstrom,
                              "results.time": unit.picosecond,
                              "results.wavenumbers": unit.angstrom ** -1}

    def _single_frame(self, frame: int, index: int) -> None:

        self._trajectory[frame]
        results = np.zeros((2, len(self._wavenumbers)), dtype=float)

        positions = np.empty((self._N, 3), dtype=float)
        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            
            # Store atom or center-of-mass positions in the current frame
            positions[s] = g.positions if gr == "atoms" \
                           else molecule.center_of_mass(g, gr)

        # Compute the sum of cosines and sines of the dot product of the
        # wavevectors and positions
        if self._n_batches:
            start = 0
            for w in np.array_split(self._wavevectors, self._n_batches, axis=0):
                arg = np.einsum("wd,pd->pw", w, positions)
                results[0, start:start + w.shape[0]] = np.cos(arg).sum(axis=0)
                results[1, start:start + w.shape[0]] = np.sin(arg).sum(axis=0)
                start += w.shape[0]
        else:
            arg = np.einsum("wd,pd->pw", self._wavevectors, positions)
            results[0] = np.cos(arg).sum(axis=0)
            results[1] = np.sin(arg).sum(axis=0)

        return results
    
    def _conclude(self) -> None:

        trig_sums = np.stack(self._results)

        # Tally intermediate scattering function for each wavevector 
        # over all frames and normalize by the number of particles and 
        # timesteps
        corr = correlation.correlation_fft if self._fft \
               else correlation.correlation_shift
        self.results.isf = (
            corr(trig_sums[:, 0], axis=0) + corr(trig_sums[:, 1], axis=0)
        ) / self._N
        
        # Flatten the array by combining values sharing the same
        # wavevector magnitude
        self.results.isf = np.hstack(
            [
                self.results.isf[
                    :, np.isclose(q, self._wavenumbers)
                ].mean(axis=1, keepdims=True)
                for q in self.results.wavenumbers
            ]
        )