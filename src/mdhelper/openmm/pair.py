"""
Custom OpenMM pair potentials
=============================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains implementations of commonly used pair potentials
that are not available in OpenMM, such as the Gaussian and Yukawa
potentials. Generally, the pair potentials are named after their LAMMPS
:code:`pair_style` counterparts, if available.
"""

from typing import Iterable, Union

import numpy as np
import openmm
from openmm import unit

from .unit import VACUUM_PERMITTIVITY

def _setup_pair(
        cnbforce: openmm.CustomNonbondedForce,
        cutoff: Union[float, unit.Quantity],
        global_params: dict[str, Union[float, unit.Quantity]],
        per_params: list,
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]]
    ) -> None:

    r"""
    Sets up a :class:`openmm.CustomNonbondedForce` object.

    Parameters
    ----------
    cnbforce : `openmm.CustomNonbondedForce`
        Custom nonbonded force object.

    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    global_params : `dict`, optional
        Global parameters.

    per_params : `list`, optional
        Per-particle parameters.

    tab_funcs : `dict`, optional
        Optional tabulated functions.
    """

    for param in global_params.items():
        cnbforce.addGlobalParameter(*param)
    for param in per_params:
        cnbforce.addPerParticleParameter(param)
    for name, func in tab_funcs.items():
        if not isinstance(func, openmm.Discrete2DFunction):
            func = openmm.Discrete2DFunction(*func.shape, func.ravel().tolist())
        cnbforce.addTabulatedFunction(name, func)
    cnbforce.setCutoffDistance(cutoff)
    cnbforce.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

def coul_gauss(
        cutoff: Union[float, unit.Quantity], tol: float = 1e-4, *,
        g_ewald: Union[float, unit.Quantity] = None,
        dims: Union[Iterable, unit.Quantity] = None,
        mix: str = "default", per_params: list = [],
        global_params: dict[str, Union[float, unit.Quantity]] = {},
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = {}
    ) -> tuple[openmm.CustomNonbondedForce, openmm.NonbondedForce]:

    r"""
    Implements the smeared Coulomb pair potential.
    
    The charges have spherical Gaussian-distributed charge 
    distributions [1]_ [2]_

    .. math::

       u_\textrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0 r_{12}}
       \textrm{erf}(\alpha_{12}r_{12})
    
    where :math:`q_1` and :math:`q_2` are the particle charges in e,
    :math:`\varepsilon_0` is the vacuum permittivity, and
    :math:`\alpha_{12}` is an inverse screening length or damping
    parameter. Effectively, electrostatic interactions are dampened for
    high charges at small separation distances.

    :math:`q_1`, :math:`q_2`, and :math:`\alpha_{12}` are determined
    from the per-particle parameters `charge` and `alpha`, respectively,
    which are set in the main script using
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    To account for a solvent implicitly using its relative permittivity
    :math:`\varepsilon_\textrm{r}`, scale the particle charges
    :math:`q_i` by :math:`1/\sqrt{\varepsilon_\textrm{r}}`.

    In addition to the default mixing rule for :math:`\alpha_{12}`, an
    implementation of the Gaussian core model is available:

    .. math::

       u_\textrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0 r_{12}}
       \textrm{erf}\left(\frac{\pi^{1/2}}{2^{1/2}a_{12}}r_{12}\right)

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    tol : `float`, default: :code:`1e-4`
        Error tolerance :math:`\delta` for Ewald summation.

    g_ewald : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Ewald splitting parameter :math:`g_\textrm{Ewald}`. If not
        provided, `g_ewald` is computed using

        .. math::

           g_\textrm{Ewald}=\frac{\sqrt{-\log(2\delta)}}{r_\textrm{cut}}

        **Reference unit**: :math:`\textrm{nm}^{-1}`.

    dims : `array_like` or `openmm.unit.Quantity`, keyword-only, optional
        Simulation system dimensions. Must be provided with `g_ewald` to
        calculate the number of mesh nodes :math:`n_\textrm{mesh}`.
        
        **Reference unit**: :math:`\textrm{nm}`.

    mix : `str`, keyword-only, default: :code:`"default"`
        Mixing rule for :math:`\alpha_{12}`.

        .. container::

           **Valid values**:
        
           * :code:`"default"`: Default mixing rule.

             .. math::
                \alpha_{12}=\frac{\alpha_1\alpha_2}
                {\sqrt{\alpha_1^2+\alpha_2^2}}

           * :code:`"core"`: Gaussian core model.

             .. math::
                \begin{gather*}
                  a_{12}^2=a_1^2+a_2^2\\ 
                  \alpha_{12}=\sqrt{\frac{\pi}{2a_{12}^2}}
                \end{gather*}

             This is equivalent to setting 
             :math:`\alpha_i=\sqrt{\pi/(2a_i^2)}`. :math:`a_{12}`, which 
             has units of :math:`\textrm{nm}`, is determined from the
             per-particle parameter `a`. `a` is automatically added to
             `per_params` and should be set in the main script using
             :meth:`openmm.openmm.NonbondedForce.addParticle`.

        Alternatively, a custom mixing rule can be provided as a string
        with valid C++ syntax. The expression must have the form
        :code:`"alpha12 = ...;"`. Unless custom global or per-particle
        parameters are defined, the expression should contain
        :code:`"alpha1"` and :code:`"alpha2"`. If a mixing rule is not
        provided, the default mixing rule is used.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\alpha_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\alpha_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}`.

    Returns
    -------
    pair_coul_gauss_dir : `openmm.CustomNonbondedForce`
        Short-range electrostatic contribution evaluated in real space.
        
    pair_coul_gauss_recip : `openmm.NonbondedForce`
        Long-range electrostatic contribution evaluated in Fourier
        (reciprocal) space.

    References
    ----------
    .. [1] Kiss, P. T.; Sega, M.; Baranyai, A. Efficient Handling of
       Gaussian Charge Distributions: An Application to Polarizable
       Molecular Models. *J. Chem. Theory Comput.* **2014**, *10* (12),
       5513–5519. https://doi.org/10.1021/ct5009069.

    .. [2] Eslami, H.; Khani, M.; Müller-Plathe, F. Gaussian Charge
       Distributions for Incorporation of Electrostatic Interactions in
       Dissipative Particle Dynamics: Application to Self-Assembly of
       Surfactants. *J. Chem. Theory Comput.* **2019**, *15* (7),
       4197–4207. https://doi.org/10.1021/acs.jctc.9b00174.
    """

    if g_ewald is None:
        g_ewald = np.sqrt(-np.log(2 * tol)) / cutoff

    global_params = {
        "G_EWALD": g_ewald,
        "ONE_4PI_EPS0": unit.AVOGADRO_CONSTANT_NA / (4 * np.pi * VACUUM_PERMITTIVITY)
    } | global_params
    if mix == "default":
        mix = "alpha12=alpha1*alpha2/sqrt(alpha1^2+alpha2^2);"
        per_params = ["alpha"] + per_params
    elif mix == "core":
        mix = f"alpha12=sqrt({np.pi}/(2*(a1^2+a2^2)));"
        per_params = ["a"] + per_params

    pair_coul_gauss_dir = openmm.CustomNonbondedForce(
        f"ONE_4PI_EPS0*q1*q2*(erf(alpha12*r)-erf(G_EWALD*r))/r;{mix}"
    )
    pair_coul_gauss_dir.addPerParticleParameter("q")
    _setup_pair(pair_coul_gauss_dir, cutoff, global_params, per_params, tab_funcs)

    pair_coul_gauss_recip = lj_coul(cutoff, tol, g_ewald=g_ewald, dims=dims)
    pair_coul_gauss_recip.setIncludeDirectSpace(False)

    return pair_coul_gauss_dir, pair_coul_gauss_recip

def gauss(
        cutoff: Union[float, openmm.unit.Quantity],
        cutoff_gauss: Union[float, openmm.unit.Quantity] = None, *,
        shift: bool = True, mix: str = "geometric",
        global_params: dict[str, Union[float, openmm.unit.Quantity]] = {},
        per_params: list = [],
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = {}
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized intermolecular Gaussian pair potential.

    The potential energy between two Gaussian particles is given by

    .. math::
        
       u_\textrm{Gauss}(r_{12})=\alpha_{12}\exp{(-\beta_{12}r_{12}^2)}

    where :math:`\alpha_{12}` is a magnitude scaling constant in
    :math:`\textrm{kJ/mol}` and :math:`\beta_{12}` is a screening length
    in :math:`\textrm{nm}`.

    :math:`\alpha_{12}` and :math:`\beta_{12}` are determined from
    per-particle parameters `alpha` and `beta`, respectively, which are
    set in the main script using
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    In addition to arithmetic and geometric mixing rules for
    :math:`\alpha_{12}` and :math:`\beta_{12}`, an implementation of the
    Gaussian core model for a melt is available:

    .. math::

        u_\textrm{Gauss}(r_{12})=A\left(\frac{3}{2\pi\sigma_{12}^2}
        \right)^{3/2}\exp{\left(-\frac{3}{2\sigma_{12}^2}r_{12}^2\right)}

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`         
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    cutoff_gauss : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance for the Gaussian potential. Must be less than
        `cutoff`. If not provided, it is set to the shared cutoff
        distance for the nonbonded interactions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the Gaussian potential is shifted at its
        cutoff to :math:`0\;\textrm{kJ/mol}`.
        
    mix : `str`, keyword-only, default: :code:`"geometric"`
        Mixing rule for :math:`\alpha_{12}` and :math:`\beta_{12}`. 
        
        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Arithmetic combining rule.

             .. math::
               
                \begin{gather*}
                  \alpha_{12}=\sqrt{\alpha_1\alpha_2}\\
                  \beta_{12}=\frac{2}{\beta_1^{-1}+\beta_2^{-1}}
                \end{gather*}

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::
             
                \begin{gather*}
                  \alpha_{12}=\sqrt{\alpha_1\alpha_2}\\
                  \beta_{12}=\sqrt{\beta_1\beta_2}
                \end{gather*}

           * :code:`"core"`: Gaussian core model.

             .. math::
                
                \begin{gather*}
                  \sigma_{12}^2=\sigma_1^2+\sigma_2^2\\
                  \beta_{12}=\frac{3}{2\sigma_{12}^2}\\
                  \alpha_{12}=A\left(
                  \frac{\beta_{12}}{\pi}\right)^{3/2}
                \end{gather*}

           :math:`\sigma_{12}`, which has units of :math:`\textrm{nm}`,
           is determined from the per-particle parameter `sigma`.
           `sigma` is automatically added to `per_params` and should be
           set in the main script using
           :meth:`openmm.openmm.NonbondedForce.addParticle`.
           The global parameter :math:`A`, which has units of
           :math:`\textrm{nm}^3 \cdot \textrm{kJ/mol}`, should be
           provided to this function via 
           :code:`global_params={"A": ...}`.

        Alternatively, a custom mixing rule can be provided as a string
        with valid C++ syntax. The expression must have the form
        :code:`"alpha12 = ...; beta12 = ...;"`. Unless custom global or
        per-particle parameters are defined, the expression should
        contain :code:`"alpha1"`, :code:`"alpha2"`, :code:`"beta1"`, and
        :code:`"beta2"`. If a mixing rule is not provided, the default
        geometric mixing rule is used.
        
    global_params : `dict`, keyword-only, optional
        Optional global parameters for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.
        
    per_params : `list`, keyword-only, optional
        Optional per-particle parameters for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.

    Returns
    -------
    pair_gauss : `openmm.CustomNonbondedForce`
        Gaussian pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff /= unit.nanometer
    if cutoff_gauss is None:
        cutoff_gauss = cutoff
    else:
        if isinstance(cutoff_gauss):
            cutoff_gauss /= unit.nanometer
        if cutoff_gauss > cutoff:
            emsg = ("The cutoff distance for the Gaussian potential "
                    "must be less than the shared cutoff distance.")
            raise ValueError(emsg)

    prefix = f"step({cutoff_gauss}-r)*(" if cutoff != cutoff_gauss else "("
    root = "alpha12*exp(-beta12*r^2)"
    suffix = f"-ucut);ucut=alpha12*exp(-beta12*{cutoff_gauss}^2);" if shift else ");"
    if mix == "arithmetic":
        per_params = ["alpha", "beta"] + per_params
        mix = "alpha12=sqrt(alpha1*alpha2);beta12=2/(1/beta1+1/beta2);"
    elif mix == "geometric":
        per_params = ["alpha", "beta"] + per_params
        mix = "alpha12=sqrt(alpha1*alpha2);beta12=sqrt(beta1*beta2);"
    elif mix == "core":
        per_params = ["sigma"] + per_params
        mix = (f"alpha12=A*(beta12/{np.pi})^(3/2);beta12=3/(2*sigma12sq);"
               "sigma12sq=sigma1^2+sigma2^2;")

    pair_gauss = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(pair_gauss, cutoff, global_params, per_params, tab_funcs)
    
    return pair_gauss

def lj_coul(
        cutoff: Union[float,unit.Quantity], tol: float = 1e-4, *,
        g_ewald: Union[float, unit.Quantity] = None,
        dims: Union[float, unit.Quantity] = None) -> openmm.NonbondedForce:

    r"""
    Implements the standard Lennard-Jones (LJ) and Coulomb pair
    potentials.

    The potential energy between two charged LJ particles is given by

    .. math::

       \begin{gather*}
         u_\textrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0r_{12}}\\
         u_\textrm{LJ}(r_{12})=4\epsilon_{12}\left[\left(
         \frac{\sigma_{12}}{r_{12}}\right)^{12}
         -\left(\frac{\sigma_{12}}{r_{12}} \right)^6\right]
       \end{gather*}
    
    where :math:`q_1` and :math:`q_2` are the particle charges in e,
    :math:`\varepsilon_0` is the vacuum permittivity,
    :math:`\sigma_{12}` is the size of the particle in
    :math:`\textrm{nm}`, and :math:`\epsilon_{12}` is the dispersion
    energy in :math:`\textrm{kJ/mol}`. :math:`q_1`, :math:`q_2`,
    :math:`\sigma_{12}`, and :math:`\epsilon_{12}` are determined from
    per-particle parameters `charge`, `sigma`, and `epsilon`,
    respectively, which are set in the main script using
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    The mixing rule for :math:`\sigma_{12}` and :math:`\epsilon_{12}` is
    the Lorentz–Berthelot combining rule:

    .. math:: 

       \begin{gather*}
         \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
         \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
       \end{gather*}
    
    To disable the LJ potential and use a different excluded volume
    interaction potential but keep the Coulomb potential, set
    :math:`\sigma_{12}=0\;\textrm{nm}` and
    :math:`\epsilon_{12}=0\;\textrm{kJ/mol}` for all particles.

    To account for a solvent implicitly using its relative permittivity
    :math:`\varepsilon_\textrm{r}`, scale the particle charges
    :math:`q_i` by :math:`1/\sqrt{\varepsilon_\textrm{r}}`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`         
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    tol : `float`, default: :code:`1e-4`
        Error tolerance :math:`\delta` for Ewald summation.

    g_ewald : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Ewald splitting parameter :math:`g_\textrm{Ewald}`. If not
        provided, `g_ewald` is computed using

        .. math::

           g_\textrm{Ewald}=\frac{\sqrt{-\log(2\delta)}}{r_\textrm{cut}}

        **Reference unit**: :math:`\textrm{nm}^{-1}`.
        
    dims : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Simulation system dimensions. Must be provided with `g_ewald` to
        calculate the number of mesh nodes :math:`n_\textrm{mesh}`.
        Both `dims` and `g_ewald` must either have units or be unitless.
        
        **Reference unit**: :math:`\textrm{nm}`.
    
    Returns
    -------
    pair_lj_coul : `openmm.NonbondedForce`        
        Hybrid LJ and Coulomb pair potentials.
    """

    pair_lj_coul = openmm.NonbondedForce()
    pair_lj_coul.setCutoffDistance(cutoff)
    pair_lj_coul.setNonbondedMethod(openmm.NonbondedForce.PME)
    if g_ewald is None or dims is None:
        pair_lj_coul.setEwaldErrorTolerance(tol)
    else:
        n_mesh = np.ceil(2 * g_ewald * dims / (3 * tol ** (1 / 5)))
        pair_lj_coul.setPMEParameters(g_ewald, *n_mesh)

    return pair_lj_coul

def ljts(
        cutoff: Union[float, unit.Quantity],
        cutoff_ljts: Union[float, unit.Quantity] = None, *,
        shift: bool = True, mix: str = "arithmetic", wca: bool = False,
        global_params: dict[str, Union[float, unit.Quantity]] = {},
        per_params: list = [],
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = {}
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the Lennard-Jones truncated and shifted (LJTS) pair
    potential.

    The potential energy between two LJ particles is given by

    .. math::

       \begin{gather*}
         u_\textrm{LJTS}(r_{12})=\begin{cases}
           u_\textrm{LJ}(r_{12})-u_\textrm{LJ}(r_\textrm{cut}),
           &\textrm{if}\,r_{12}<r_\textrm{cut}\\
           0,&\textrm{if}\,r_{12}\geq r_\textrm{cut}
        \end{cases}\\
        u_\textrm{LJ}(r_{12})=4\epsilon_{12}
        \left[\left(\frac{\sigma_{12}}{r_{12}}\right)^{12}
        -\left(\frac{\sigma_{12}}{r_{12}}\right)^6\right]
       \end{gather*}
    
    where :math:`\sigma_{12}` is the size of the particle in
    :math:`\textrm{nm}` and :math:`\epsilon_{12}` is the dispersion
    energy in :math:`\textrm{kJ/mol}`. :math:`\sigma_{12}` and
    :math:`\epsilon_{12}` are determined from per-particle parameters
    `sigma` and `epsilon`, respectively, which are set in the main
    script using :meth:`openmm.openmm.NonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`         
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    cutoff_ljts : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance for the LJTS potential. Must be less than the
        shared cutoff distance. If not provided, it is set to the shared
        cutoff distance for the nonbonded interactions. If the
        Weeks–Chander–Andersen (WCA) potential is used, a dynamic cutoff
        of :math:`2^{1/6}\sigma_{12}` is used instead. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the LJTS potential is shifted at its cutoff
        to :math:`0\;\textrm{kJ/mol}`. If the WCA potential is used, a
        dynamic shift of :math:`\epsilon_{12}` is used instead.

    wca : `bool`, keyword-only, default: :code:`False`
        Determines whether to use the WCA potential.

    mix : `str`, keyword-only, default: :code:`"arithmetic"`
        Mixing rule for :math:`\sigma_{12}` and :math:`\epsilon_{12}`.
        
        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Lorentz-Berthelot combining rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::
               
                \begin{gather*}
                  \sigma_{12}=\sqrt{\sigma_1\sigma_2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}
            
           * :code:`"sixthpower"`: Sixth-power mixing rule.
        
             .. math::
                
                \begin{gather*}
                  \sigma_{12}=\left(\frac{\sigma_1^6
                  +\sigma_2^6}{2}\right)^{1/6}\\
                  \epsilon_{12}=\frac{2\sqrt{\epsilon_1\epsilon_2}
                  \sigma_1^3\sigma_2^3}{\sigma_1^6+\sigma_2^6}
                \end{gather*}

        Alternatively, a custom mixing rule can be provided as a string
        with valid C++ syntax. The expression must have the form
        :code:`"sigma12 = ...; epsilon12 = ...;"`. Unless custom global
        or per-particle parameters are defined, the expression should
        contain :code:`"sigma1"`, :code:`"sigma2"`, :code:`"epsilon1"`,
        and :code:`"epsilon2"`. If a mixing rule is not provided, the
        default arithmetic mixing rule is used.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    Returns
    -------
    pair_ljts : `openmm.CustomNonbondedForce`
        LJTS pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff /= unit.nanometer
    if cutoff_ljts is None:
        cutoff_ljts = cutoff
    else:
        if isinstance(cutoff_ljts):
            cutoff_ljts /= unit.nanometer
        if cutoff_ljts > cutoff:
            emsg = ("The cutoff distance for the LJTS potential must be "
                    "less than the shared cutoff distance.")
            raise ValueError(emsg)

    if wca:
        prefix = "step(2^(1/6)*sigma12-r)*("
    elif cutoff != cutoff_ljts:
        prefix = f"step({cutoff_ljts}-r)*("
    else:
        prefix = "("
    root = "4*epsilon12*((sigma12/r)^12-(sigma12/r)^6)"
    if wca:
        suffix = "+epsilon12);"
    elif shift:
        suffix = (f"-ucut);ucut=4*epsilon12*((sigma12/{cutoff_ljts})^12"
                  f"-(sigma12/{cutoff_ljts})^6));")
    else:
        suffix = ");"
    if mix == "arithmetic":
        mix = "sigma12=(sigma1+sigma2)/2;epsilon12=sqrt(epsilon1*epsilon2);"
        per_params = ["sigma", "epsilon"] + per_params
    elif mix == "geometric":
        mix = "sigma12=sqrt(sigma1*sigma2);epsilon12=sqrt(epsilon1*epsilon2);"
        per_params = ["sigma", "epsilon"] + per_params
    elif mix == "sixthpower":
        mix = ("sigma12=((sigma1^6+sigma2^6)/2)^(1/6);"
               "epsilon12=2*sqrt(epsilon1*epsilon2)*sigma1^3*sigma2^3"
               "/(sigma1^6+sigma2^6);")
        per_params = ["sigma", "epsilon"] + per_params

    pair_ljts = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(pair_ljts, cutoff, global_params, per_params, tab_funcs)

    return pair_ljts

def solvation(
        cutoff: Union[float, unit.Quantity],
        cutoff_solvation: Union[float, unit.Quantity] = None, *,
        mix: str = "arithmetic", per_params: list = [],
        global_params: dict[str, Union[float, unit.Quantity]] = {},
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = {}
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the solvation pair potential.

    The solvation energy between particles is given by

    .. math::

       u_\textrm{solv}(r_{12})=-S_{12}\left[\left(
       \frac{\sigma_{12}}{r_{12}}\right)^4
       -\left(\frac{\sigma_{12}}{r_\mathrm{cut}}\right)^4\right] 

    where :math:`\sigma_{12}` is the size of the particle in
    :math:`\textrm{nm}` and :math:`S_{12}` is the solvation strength
    in :math:`\textrm{kJ/mol}`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`         
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    cutoff_solvation : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance for the solvation potential. Must be less than the
        shared cutoff. If not provided, it is set to the shared cutoff
        distance for the nonbonded interactions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    mix : `str`, keyword-only, default: :code:`"arithmetic"` 
        Mixing rule for :math:`\sigma_{12}` and :math:`S_{12}`.

        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Arithmetic mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
                  S_{12}=\sqrt{S_1S_2}
                \end{gather*}

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::
               
                \begin{gather*}
                  \sigma_{12}=\sqrt{\sigma_1\sigma_2}\\
                  S_{12}=\sqrt{S_1S_2}
                \end{gather*}

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.
        
    Returns
    -------
    pair_solv : `openmm.CustomNonbondedForce`
        Solvation pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff /= unit.nanometer
    if cutoff_solvation is None:
        cutoff_solvation = cutoff
    else:
        if isinstance(cutoff_solvation):
            cutoff_solvation /= unit.nanometer
        if cutoff_solvation > cutoff:
            emsg = ("The cutoff distance for the solvation potential "
                    "must be less than the shared cutoff distance.")
            raise ValueError(emsg)

    root = "-S12*((sigma12/r)^4-(sigma12/cut)^4)"

    if mix == "arithmetic":
        mix = "sigma12=(sigma1+sigma2)/2;S12=sqrt(S1*S2);"
        per_params = ["sigma", "S"] + per_params
    elif mix == "geometric":
        mix = "sigma12=sqrt(sigma1*sigma2);S12=sqrt(S1*S2);"
        per_params = ["sigma", "S"] + per_params

    pair_solv = openmm.CustomNonbondedForce(f"{root}{mix}")
    _setup_pair(pair_solv, cutoff, global_params, per_params, tab_funcs)

    return pair_solv

def yukawa(
        cutoff: Union[float, unit.Quantity],
        cutoff_yukawa: Union[float, unit.Quantity] = None, *,
        shift: bool = True, mix: str = "geometric", per_params: list = [],
        global_params: dict[str, Union[float, unit.Quantity]] = {},
        tab_funcs: dict[str, Union[np.ndarray, openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = {}
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized intermolecular Yukawa pair potential.

    The potential energy between two Yukawa particles is given by

    .. math::

       u_\textrm{Yukawa}(r_{12})=\frac{\alpha_{12}
       \exp(-\kappa r_{12})}{r_{12}}
    
    where :math:`\alpha_{12}` is a magnitude scaling constant in
    :math:`\textrm{nm}\cdot\textrm{kJ/mol}` and :math:`\kappa` is a
    screening length in :math:`\textrm{nm}^{-1}`. 
    
    :math:`\kappa` is a global parameter that must be provided in
    `global_params`, while :math:`\alpha_{12}` is determined from the
    per-particle parameter `alpha` that must be set in the main script
    using :meth:`openmm.openmm.NonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`         
        Shared cutoff distance :math:`r_\textrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    cutoff_yukawa : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance for the Yukawa potential. Must be less than the
        shared cutoff. If not provided, it is set to the shared cutoff
        distance for the nonbonded interactions. 
        
        **Reference unit**: :math:`\textrm{nm}`.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the Yukawa potential is shifted at its
        cutoff to :math:`0\;\textrm{kJ/mol}`.

    mix : `str`, keyword-only, default: :code:`"geometric"` 
        Mixing rule for :math:`\alpha_{12}`.

        .. container::

           **Valid values**:

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \alpha_{12}=\sqrt{\alpha_1\alpha_2}

        Alternatively, a custom mixing rule can be provided as a string
        with valid C++ syntax. The expression must have the form
        :code:`"alpha12 = ...;"`. Unless custom global or per-particle
        parameters  are defined, the expression should contain
        :code:`"alpha1"` and :code:`"alpha2"`. If a mixing rule is not
        provided, the default geometric mixing rule is used.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\alpha_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\alpha_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}`.

    Returns
    -------
    pair_yukawa : `openmm.CustomNonbondedForce`
        Yukawa pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff /= unit.nanometer
    if cutoff_yukawa is None:
        cutoff_yukawa = cutoff
    else:
        if isinstance(cutoff_yukawa):
            cutoff_yukawa /= unit.nanometer
        if cutoff_yukawa > cutoff:
            emsg = ("The cutoff distance for the LJTS potential must be "
                    "less than the shared cutoff distance.")
            raise ValueError(emsg)

    prefix = f"step({cutoff_yukawa}-r)*(" if cutoff != cutoff_yukawa else "("
    root = "alpha12*exp(-kappa*r)/r"
    suffix = f"-ucut);ucut=alpha12*exp(-kappa*{cutoff_yukawa})/{cutoff_yukawa};" \
             if shift else ");"
    if mix == "geometric":
        per_params = ["alpha"] + per_params
        mix = "alpha12=sqrt(alpha1*alpha2);"

    Yukawa = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(Yukawa, cutoff, global_params, per_params, tab_funcs)

    return Yukawa