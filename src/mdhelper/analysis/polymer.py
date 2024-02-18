"""
Polymeric analysis
==================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to determine the structural
and dynamical properties of polymeric systems.
"""

from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
import numpy as np
from scipy import optimize, special

from .base import SerialAnalysisBase, ParallelAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm import correlation
from ..algorithm.molecule import center_of_mass, radius_of_gyration
from ..algorithm.topology import unwrap
from ..algorithm.utility import strip_unit
from ..fit.exponential import stretched_exp

if FOUND_OPENMM:
    from openmm import unit

def correlation_fft(*args, **kwargs):

    """
    Evaluates the autocorrelation function (ACF) or cross-correlation
    function (CCF) of a time series using fast Fourier transforms (FFT).

    .. note::

       This is an alias function. For more information, see
       :func:`mdhelper.algorithm.correlation.correlation_fft`.
    """

    return correlation.correlation_fft(*args, **kwargs)

def correlation_shift(*args, **kwargs) -> np.ndarray[float]:

    """
    Evaluates the autocorrelation function (ACF) or cross-correlation
    function (CCF) of a time series directly by using sliding windows
    along the time axis.

    .. note::

       This is an alias function. For more information, see
       :func:`mdhelper.algorithm.correlation.correlation_shift`.
    """

    return correlation.correlation_shift(*args, **kwargs)

def calculate_relaxation_time(
        time: np.ndarray[float], acf: np.ndarray[float]) -> float:

    r"""
    Calculates the orientational relaxation time :math:`\tau_\mathrm{r}`
    of a polymer using the end-to-end vector autocorrelation function
    (ACF) time series :math:`C_\mathrm{ee}`.

    A stretched exponential function with :math:`\tau` and :math:`\beta`
    as coefficients,

    .. math::

       C_\mathrm{ee}=\exp{\left[-(t/\tau)^\beta\right]}

    is fitted to the ACF time series, and the relaxation time is estimated
    using

    .. math::

       \tau_\mathrm{r}=\int_0^\infty C_\mathrm{ee}\,dt=\tau\Gamma(1/\beta)

    Parameters
    ----------
    time : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\textrm{ps}`.

    acf : `numpy.ndarray`
        End-to-end vector ACFs for the :math:`N_\textrm{g}` groups over
        :math:`N_\textrm{b}` blocks of :math:`N_t` trajectory frames
        each.

        **Shape**:
        :math:`(N_\textrm{g},\,N_\textrm{b},\,N_t)`.

    Returns
    -------
    relaxation_time : `float`
        Average orientational relaxation time.

        **Reference unit**: :math:`\textrm{ps}`.
    """

    tau_r, beta = optimize.curve_fit(stretched_exp, time / time[1], acf,
                                     bounds=(0, np.inf))[0]
    return tau_r * time[1] * special.gamma(1 + beta ** -1)

class _PolymerAnalysisBase(SerialAnalysisBase):

    r"""
    An analysis base object for polymer systems.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of polymers to be analyzed. All polymers in each group
        must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (for coarse-grained polymer
             simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic polymer simulations).

    n_chains : `int` or array-like, optional
        Number of chains in each polymer group. Must be provided if the
        trajectory does not adhere to the standard container heirarchy
        (see Notes).

    n_monomers : `int` or array-like, optional
        Number of monomers in each chain in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy (see Notes).

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
        reference units for :code:`results.time`, call
        :code:`results.units["results.time"]`.

    Notes
    -----
    In a standard trajectory file, segments (or chains) contain
    residues, and residues contain atoms. This heirarchy must be adhered
    to for this analysis module to function correctly. If your
    trajectory file does not contain the correct segment or residue
    information, provide the number of chains and chain lengths in
    `n_chains` and `n_monomers`, respectively.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *, unwrap: bool = False,
            verbose: bool = True, **kwargs):

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)
        self._dimensions = self.universe.dimensions
        if self._dimensions is not None:
            self._dimensions = self._dimensions[:3].copy()

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

        if hasattr(self.universe, "segments"):
            self._n_chains = np.zeros(self._n_groups, dtype=int)
            self._n_monomers = np.zeros(self._n_groups, dtype=int)
            for i, g in enumerate(self._groups):
                self._n_chains[i] = g.segments.n_segments
                self._n_monomers[i] = g.n_atoms // self._n_chains[i]
        else:
            if isinstance(n_chains, np.integer):
                self._n_chains = n_chains * np.ones(self._n_groups, dtype=int)
            elif self._n_groups == len(n_chains):
                self._n_chains = n_chains
            else:
                emsg = ("The number of polymer counts is not equal to the "
                        "number of groups.")
                raise ValueError(emsg)
            if isinstance(n_monomers, np.integer):
                self._n_monomers = n_monomers * np.ones(n_monomers, dtype=int)
            elif self._n_groups == len(n_monomers):
                self._n_monomers = n_monomers
            else:
                emsg = ("The number of chain lengths is not equal to the "
                        "number of groups.")
                raise ValueError(emsg)

        self._unwrap = unwrap
        self._verbose = verbose

        self.results.units = {"_dimensions": ureg.angstrom}

class Gyradius(_PolymerAnalysisBase):

    r"""
    A serial implementation to calculate the radius of gyration
    :math:`R_\mathrm{g}` of a polymer.

    The radius of gyration is used to describe the dimensions of a
    polymer chain, and is defined as

    .. math::

        R_\mathrm{g}=\sqrt{
        \frac{\sum_i^N m_i\|\mathbf{r}_i-\mathbf{R}_\mathrm{com}\|^2}
        {\sum_i^N m_i}}

    where :math:`m_i` and :math:`\mathbf{r}_i` are the mass and
    position, respectively, of particle :math:`i`, and
    :math:`\mathbf{R}_\mathrm{com}` is the center of mass.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or `array_like`
        Group(s) of polymers to be analyzed. All polymers in each group
        must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (for coarse-grained polymer
             simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic polymer simulations).

    n_chains : `int` or array-like, optional
        Number of chains in each polymer group. Must be provided if the
        trajectory does not adhere to the standard container heirarchy
        (see Notes).

    n_monomers : `int` or array-like, optional
        Number of monomers in each chain in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy (see Notes).

    components : `bool`, keyword-only, default: :code:`False`
        Specifies whether the components of the radii of gyration are
        calculated and returned instead.

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
        reference units for :code:`results.time`, call
        :code:`results.units["results.time"]`.

    results.gyradius : `numpy.ndarray`
        Radii of gyration for the :math:`N_\textrm{g}` groups over
        :math:`N_t` trajectory frames.

        **Shape**: :math:`(N_\textrm{g},\,N_t)`
        (:code:`components=False`) or :math:`(N_\textrm{g},\,N_t,\,3)`
        (:code:`components=True`).

        **Reference unit**: :math:`\textrm{Å}`.

    Notes
    -----
    In a standard trajectory file, segments (or chains) contain
    residues, and residues contain atoms. This heirarchy must be adhered
    to for this analysis module to function correctly. If your
    trajectory file does not contain the correct segment or residue
    information, provide the number of chains and chain lengths in
    `n_chains` and `n_monomers`, respectively.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *,
            components: bool = False, unwrap: bool = False,
            verbose: bool = True, **kwargs):

        super().__init__(groups, groupings, n_chains, n_monomers,
                         unwrap=unwrap, verbose=verbose, **kwargs)
        self._components = components

    def _prepare(self) -> None:

        # Unwrap particle positions if necessary
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0]
                if hasattr(self._sliced_trajectory, "frames")
                else (self.start or 0)
            ]
            self._positions_old = [np.empty((M * N_p, 3))
                                   for M, N_p in zip(self._n_chains,
                                                     self._n_monomers)]
            for i, (g, gr) in enumerate(zip(self._groups, self._groupings)):
                self._positions_old[i] = (g.positions if gr == "atoms"
                                          else center_of_mass(g, gr))
            self._images = [np.zeros(p.shape, dtype=int)
                            for p in self._positions_old]
            self._thresholds = self._dimensions / 2

        # Preallocate arrays to store results
        shape = ((self._n_groups, self.n_frames, 3) if self._components
                 else (self._n_groups, self.n_frames))
        self.results.gyradius = np.empty(shape)

        # Store reference units
        self.results.units = {"results.gyradius": ureg.angstrom}

    def _single_frame(self) -> None:

        for i, (g, gr, M, N_p) in enumerate(
                zip(self._groups, self._groupings, self._n_chains,
                    self._n_monomers)):

            # Store atom or center-of-mass positions in the current frame
            positions = (g.positions if gr == "atoms"
                         else center_of_mass(g, gr))

            # Unwrap particle positions if necessary
            if self._unwrap:
                unwrap(
                    positions,
                    self._positions_old[i],
                    self._dimensions,
                    thresholds=self._thresholds,
                    images=self._images[i]
                )

            # Calculate radius of gyration
            self.results.gyradius[i, self._frame_index] \
                = radius_of_gyration(
                    grouping="segments",
                    positions=positions.reshape((M, N_p, 3)),
                    masses=g.masses.reshape((M, N_p)),
                    components=self._components
                )

class Relaxation(_PolymerAnalysisBase):

    r"""
    A serial implementation to calculate the end-to-end vector
    autocorrelation function (ACF) :math:`C_\mathrm{ee}(t)` and
    estimate the orientational relaxation time :math:`\tau_\mathrm{r}`
    of a polymer.

    The end-to-end vector ACF is defined as

    .. math::

       C_\mathrm{ee}(t)=\frac{\langle\mathbf{R}_\mathrm{ee}(t)
       \cdot\mathbf{R}_\mathrm{ee}(0)\rangle}
       {\langle\mathbf{R}_\mathrm{ee}^2\rangle}

    where :math:`\mathbf{R}_\mathrm{ee}=\mathbf{r}_N-\mathbf{r}_1`
    is the end-to-end vector.

    The orientational relaxation time can then be estimated by fitting
    a stretched exponential function

    .. math::

       C_\mathrm{ee}=\exp{\left[-(t/\tau)^\beta\right]}

    to the end-to-end vector ACF and evaluating

    .. math::

       \tau_\mathrm{r}=\int_0^\infty C_\mathrm{ee}\,dt
       =\tau\Gamma(\frac{1}{\beta}+1)

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or `array_like`
        Group(s) of polymers to be analyzed. All polymers in each group
        must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (for coarse-grained polymer
             simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic polymer simulations).

    n_chains : `int` or array-like, optional
        Number of chains in each polymer group. Must be provided if the
        trajectory does not adhere to the standard container heirarchy
        (see Notes).

    n_monomers : `int` or array-like, optional
        Number of monomers in each chain in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy (see Notes).

    n_blocks : `int`, keyword-only, default: :code:`1`
        Number of blocks to split the trajectory into.

    fft : `bool`, keyword-only, default: :code:`True`
        Determines whether fast Fourier transforms (FFT) are used to
        evaluate the ACFs.

    dt : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Time between frames :math:`\Delta t`. While this is normally
        determined from the trajectory, the trajectory may not have the
        correct information if the data is in reduced units. For
        example, if your reduced timestep is :math:`0.01` and you output
        trajectory data every :math:`10,000` timesteps, then
        :math:`\Delta t=100`.

        **Reference unit**: :math:`\mathrm{ps}`.

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
        reference units for :code:`results.time`, call
        :code:`results.units["results.time"]`.

    results.time : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\textrm{ps}`.

    results.acf : `numpy.ndarray`
        End-to-end vector ACFs for the :math:`N_\textrm{g}` groups over
        :math:`N_\textrm{b}` blocks with :math:`N_t` trajectory frames
        each.

        **Shape**:
        :math:`(N_\textrm{g},\,N_\textrm{b},\,N_t)`.

    results.relaxation_time : `numpy.ndarray`
        Average orientational relaxation times for the
        :math:`N_\textrm{g}` groups over :math:`N_t` trajectory frames
        split into :math:`N_\textrm{b}` blocks.

    Notes
    -----
    In a standard trajectory file, segments (or chains) contain
    residues, and residues contain atoms. This heirarchy must be adhered
    to for this analysis module to function correctly. If your
    trajectory file does not contain the correct segment or residue
    information, provide the number of chains and chain lengths in
    `n_chains` and `n_monomers`, respectively.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *, n_blocks: int = 1,
            fft: bool = True, dt: Union[float, "unit.Quantity", Q_] = None,
            unwrap: bool = False, verbose: bool = True, **kwargs) -> None:

        super().__init__(groups, groupings, n_chains, n_monomers,
                         unwrap=unwrap, verbose=verbose, **kwargs)

        self._n_blocks = n_blocks
        self._fft = fft
        self._dt = strip_unit(dt or self._trajectory.dt, "picosecond")

    def _prepare(self) -> None:

        # Determine number of frames used when the trajectory is split
        # into blocks
        self._n_frames_block = self.n_frames // self._n_blocks
        self._n_frames = self._n_blocks * self._n_frames_block
        extra_frames = self.n_frames - self._n_frames
        if extra_frames > 0:
            wmsg = (f"The trajectory is not divisible into {self._n_blocks:,} "
                    f"blocks, so the last {extra_frames:,} frame(s) will be "
                    "discarded. To maximize performance, set appropriate "
                    "starting and ending frames in run() so that the number "
                    "of frames to be analyzed is divisible by the number of "
                    "blocks.")
            warnings.warn(wmsg)

        # Preallocate array(s) to store positions of first and last
        # monomers of each polymer chain (and number of boundary
        # crossings)
        self._positions_end = [np.empty((self.n_frames, M, 2, 3))
                               for M in self._n_chains]

        # Unwrap particle positions if necessary
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0]
                if hasattr(self._sliced_trajectory, "frames")
                else (self.start or 0)
            ]
            self._positions_end_prev = [np.empty((M, 2, 3)) 
                                        for M in self._n_chains]
            for i, (g, gr, N_p) in enumerate(zip(self._groups, self._groupings,
                                                 self._n_monomers)):
                self._positions_end_prev[i] = (
                    g.positions if gr == "atoms" else center_of_mass(g, gr)
                ).reshape(-1, N_p, 3)[:, (0, -1)]
            self._images = [np.zeros(p.shape, dtype=int)
                            for p in self._positions_end_prev]
            self._thresholds = self._dimensions / 2

        # Preallocate arrays to store results
        self.results.time = self.step * self._dt * np.arange(self._n_frames //
                                                             self._n_blocks)
        self.results.acf = np.empty(
            (self._n_groups, self._n_blocks, self._n_frames_block)
        )

    def _single_frame(self) -> None:

        for i, (g, gr, N_p) in enumerate(zip(self._groups, self._groupings,
                                             self._n_monomers)):

            # Store atom or center-of-mass positions in the current frame
            self._positions_end[i][self._frame_index] = (
                g.positions if gr == "atoms" else center_of_mass(g, gr)
            ).reshape(-1, N_p, 3)[:, (0, -1)]

            # Unwrap particle positions if necessary
            if self._unwrap:
                unwrap(
                    self._positions_end[i][self._frame_index],
                    self._positions_end_prev[i],
                    self._dimensions,
                    thresholds=self._thresholds,
                    images=self._images[i],
                )

    def _conclude(self) -> None:

        # Compute the end-to-end vector autocorrelation function
        _acf = correlation_fft if self._fft else correlation_shift
        for i, M in ProgressBar(enumerate(self._n_chains)):
            e2e = np.diff(self._positions_end[i], axis=2)[:, :, 0]
            self.results.acf[i] = _acf(
                (e2e / np.linalg.norm(e2e, axis=2, keepdims=True))
                .reshape(self._n_blocks, -1, M, 3)
            )

    def calculate_relaxation_time(self) -> None:

        """
        Calculates the orientational relaxation time.
        """

        if not hasattr(self.results, "acf"):
            emsg = ("Call Relaxation.run() before "
                    "Relaxation.calculate_relaxation_time().")
            raise RuntimeError(emsg)

        self.results.relaxation_time = np.empty(
            (self._n_groups, self._n_blocks)
        )
        self.results.units["results.relaxation_time"] = ureg.picosecond

        for i, g in enumerate(self.results.acf):
            for j, acf in enumerate(g):
                valid = np.where(acf >= 0)[0]
                self.results.relaxation_time[i, j] = calculate_relaxation_time(
                    self.results.time[valid], acf[valid]
                )

class SingleChainStructureFactor(SerialAnalysisBase):

    r"""
    A serial implementation to calculate the single-chain structure
    factor :math:`S_\mathrm{sc}(q)` of a homopolymer.

    It is defined as

    .. math::

       S_{\mathrm{sc}}(\mathbf{q})=\frac{1}{MN_\mathrm{p}}
       \sum_{m=1}^M\sum_{i,j=1}^{N_\mathrm{p}}\left\langle
       \exp{[i\mathbf{q}\cdot(\mathbf{r}_i-\mathbf{r}_j)]}\right\rangle

    where :math:`M` is the number of chains, :math:`N_\mathrm{p}` is the
    chain length, :math:`\mathbf{q}` is the scattering wavevector, and
    :math:`\mathbf{r}_i` is the position of the :math:`i`-th monomer.

    .. container::

       The single-chain structure factor reveals information about the
       characteristic length scales of the polymer:

       * In the Guinier regime (:math:`qR_g\ll1`),
         :math:`S_{\mathrm{sc}}(q)^{-1}\approx N_\mathrm{p}(1-(qR_g)^2/3)`
         can be used to determine the radius of gyration :math:`R_g`.
       * In the Porod regime (:math:`qR_g\gg1`),
         :math:`S_{\mathrm{sc}}(q)=1` since the only contribution is the
         self-scattering of the monomers.
       * In the intermediate regime, the slope :math:`s` of the log-log
         plot of :math:`S_{\mathrm{sc}}(q)` is related to the scaling
         exponent :math:`\nu` via :math:`\nu=-1/s`.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`
        Group of polymers to be analyzed.

    grouping : `str`, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (for coarse-grained polymer
             simulations).
           * :code:`"residues"`: Residues' centers of mass (for 
             atomistic polymer simulations).

    n_points : `int`, default: :code:`32`
        Number of points to sample the wavevector space.

    n_chains : `int`, optional
        Number of chains in `group`. Must be provided if the topology 
        does not contain segment information.

    n_monomers : `int`, optional
        Number of monomers per chain. Must be provided if the topology
        does not contain segment information.

    dimensions : `numpy.ndarray` or `openmm.unit.Quantity`, optional
        System dimensions. If the 
        :class:`MDAnalysis.core.universe.Universe` object that `group`
        belongs to does not contain dimensionality information, provide
        it here.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\textrm{Å}`.

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
        reference units for :code:`results.time`, call
        :code:`results.units["results.time"]`.

    results.wavenumbers : `numpy.ndarray`
        Unique wavenumbers.

        **Shape**: :math:`(N_q,)`.

        **Reference unit**: :math:`\textrm{Å}^{-1}`.

    results.scsf : `numpy.ndarray`
        Single-chain structure factors for the unique wavenumbers.

        **Shape**: :math:`(N_q,)`.

    Notes
    -----
    In a standard trajectory file, segments (or chains) contain
    residues, and residues contain atoms. This heirarchy must be adhered
    to for this analysis module to function correctly. If your
    trajectory file does not contain the correct segment or residue
    information, provide the number of chains and chain lengths in
    `n_chains` and `n_monomers`, respectively.
    """

    def __init__(
            self, group: mda.AtomGroup, grouping: str = "atoms",
            n_points: int = 32, *, n_chains: int = None, n_monomers: int = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            unwrap: bool = False, verbose: bool = True, **kwargs):

        self._group = group
        self.universe = group.universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        self.results.units = {"_dimensions": ureg.angstrom}

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            self._dimensions = np.asarray(strip_unit(dimensions, "angstrom")[0])
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        self._grouping = grouping
        if hasattr(self.universe, "segments"):
            self._n_chains = self._group.segments.n_segments
            self._n_monomers = self._group.n_atoms // self._n_chains
        else:
            if isinstance(n_chains, np.integer):
                self._n_chains = n_chains
            else:
                emsg = ("The number of chains must be specified when "
                        "the universe does not contain segment "
                        "information.")
                raise ValueError(emsg)
            if isinstance(n_monomers, np.integer):
                self._n_monomers = n_monomers
            else:
                emsg = ("The number of monomers per chain must be "
                        "specified when the universe does not contain "
                        "segment information.")
                raise ValueError(emsg)

        self._n_points = n_points
        self._wavevectors = np.stack(
            np.meshgrid(
                *[2 * np.pi * np.arange(self._n_points) / L
                 for L in self._dimensions]
            ), -1
        ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        # Unwrap particle positions if necessary
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0]
                if hasattr(self._sliced_trajectory, "frames")
                else (self.start or 0)
            ]
            self._positions_old = (
                self._group.positions if self._grouping == "atoms"
                else center_of_mass(self._group, self._grouping)
            )
            self._images = np.zeros(self._positions_old.shape, dtype=int)
            self._thresholds = self._dimensions / 2

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))
        self.results.units["results.wavenumbers"] = ureg.angstrom ** -1

        # Preallocate arrays to store results
        self.results.scsf = np.zeros(len(self._wavevectors))

    def _single_frame(self) -> None:

        # Get atom or center-of-mass positions in the current frame
        positions = (self._group.positions if self._grouping == "atoms"
                     else center_of_mass(self._group, self._grouping))

        # Unwrap particle positions if necessary
        if self._unwrap:
            unwrap(
                positions,
                self._positions_old,
                self._dimensions,
                thresholds=self._thresholds,
                images=self._images
            )

        # Calculate single-chain structure factor contributions
        for chain in positions.reshape((self._n_chains, self._n_monomers, 3)):
            arg = np.einsum("ij,kj->ki", self._wavevectors, chain)
            self.results.scsf += (np.sin(arg).sum(axis=0) ** 2
                                  + np.cos(arg).sum(axis=0) ** 2)

    def _conclude(self) -> None:

        # Normalize the single-chain structure factor by
        # dividing by the total number of monomers and timesteps
        self.results.scsf /= self._n_chains * self._n_monomers * self.n_frames

        # Flatten the array for each combination by combining
        # values sharing the same wavevector magnitude
        self.results.scsf = np.fromiter(
            (self.results.scsf[np.isclose(q, self._wavenumbers)].mean()
             for q in self.results.wavenumbers),
            dtype=float,
            count=len(self.results.wavenumbers)
        )

class ParallelSingleChainStructureFactor(
        SingleChainStructureFactor, ParallelAnalysisBase):

    r"""
    A multithreaded implementation to calculate the single-chain
    structure factor :math:`S_\mathrm{sc}(q)` of a homopolymer.

    .. note::

       For a theoretical background and a complete list of parameters,
       attributes, and available methods, see
       :class:`SingleChainStructureFactor`.
    """

    def __init__(
            self, group: mda.AtomGroup, grouping: str = "atoms",
            n_points: int = 32, *, n_chains: int = None, n_monomers: int = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            unwrap: bool = False, verbose: bool = True, **kwargs):

        SingleChainStructureFactor.__init__(
            self, group, grouping, n_points, n_chains=n_chains,
            n_monomers=n_monomers, dimensions=dimensions, unwrap=unwrap,
            verbose=verbose, **kwargs
        )
        ParallelAnalysisBase.__init__(self, self.universe.trajectory,
                                      verbose=verbose, **kwargs)

    def _prepare(self) -> None:

        # Unwrap particle positions if necessary
        self._positions = np.empty(
            (self.n_frames, self._n_chains * self._n_monomers, 3)
        )
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0]
                if hasattr(self._sliced_trajectory, "frames")
                else (self.start or 0)
            ]
            positions_old = (
                self._group.positions if self._grouping == "atoms"
                else center_of_mass(self._group, self._grouping)
            )
            images = np.zeros(positions_old.shape, dtype=int)
            threshold = self._dimensions / 2

        # Store particle positions in a shared memory array
        for i, _ in enumerate(self.universe.trajectory[
                list(self._sliced_trajectory.frames)
                if hasattr(self._sliced_trajectory, "frames")
                else slice(self.start, self.stop, self.step)
            ]):

            # Store atom or center-of-mass positions in the current frame
            self._positions[i] = (
                self._group.positions if self._grouping == "atoms"
                else center_of_mass(self._group, self._grouping)
            )

            # Unwrap particle positions if necessary
            if self._unwrap:
                unwrap(
                    self._positions[i],
                    positions_old,
                    self._dimensions,
                    thresholds=threshold,
                    images=images
                )

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))

    def _single_frame(self, frame: int, index: int) -> np.ndarray[float]:

        # Compute the single-chain structure factor by squaring the
        # cosine and sine terms and adding them together
        scsf = np.zeros(len(self._wavevectors), dtype=float)
        for chain in self._positions[index].reshape((self._n_chains, -1, 3)):
            arg = np.einsum("ij,kj->ki", self._wavevectors, chain)
            scsf += np.sin(arg).sum(axis=0) ** 2 + np.cos(arg).sum(axis=0) ** 2
        return scsf

    def _conclude(self) -> None:

        # Tally single-chain structure factor for each wavevector over
        # all frames and normalize by the number of particles and timesteps
        self.results.scsf = (
            np.vstack(self._results).sum(axis=0) 
            / (self._n_chains * self._n_monomers * self.n_frames)
        )

        # Flatten the array by combining values sharing the same
        # wavevector magnitude
        self.results.scsf = np.fromiter(
            (self.results.scsf[np.isclose(q, self._wavenumbers)].mean()
             for q in self.results.wavenumbers),
            dtype=float,
            count=len(self.results.wavenumbers)
        )