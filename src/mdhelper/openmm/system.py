"""
OpenMM system transformations
=============================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains implementations of common OpenMM topology 
transformations, like the Yeh–Berkowitz slab correction, the method of
image charges, and adding an external electric field.
"""

from typing import Any, Iterable, Union
import warnings

try:
    from constvplugin import ConstVLangevinIntegrator
    FOUND_CONSTVPLUGIN = True
except ImportError:
    FOUND_CONSTVPLUGIN = False
import numpy as np
import openmm
from openmm import app, unit

from .unit import VACUUM_PERMITTIVITY
from .. import ArrayLike

def register_particles(
        system: openmm.System = None, topology: app.Topology = None,
        N: int = 0, mass: Union[float, unit.Quantity] = 0.0, *,
        chain: app.Chain = None, element: app.Element = None, name: str = "",
        resname: str = "", nbforce: openmm.NonbondedForce = None,
        charge: Union[float, unit.Quantity] = 0.0,
        sigma: Union[float, unit.Quantity] = 0.0,
        epsilon: Union[float, unit.Quantity] = 0.0,
        cnbforces: Union[ArrayLike, dict[openmm.CustomNonbondedForce, 
                                         Iterable[Any]]] = {}
    ) -> None:

    """
    Sequentially register particles of the same type to the simulation
    system, topology, and nonbonded pair potentials.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.

    N : `int`, default: :code:`0`
        Number of atom(s). If not provided, no particles are added.

    mass : `float` or `openmm.unit.Quantity`, default: :code:`0`
        Molar mass. If not provided, particles have no mass and will not
        move. 
        
        **Reference unit**: :math:`\mathrm{g/mol}`.

    chain : `openmm.app.Chain`, keyword-only, optional
        Chain that the atom(s) should be added to. If not provided,
        a new chain is created for each atom.

    element : `openmm.app.Element`, keyword-only, optional
        Chemical element of the atom(s) to add.
    
    name : `str`, keyword-only, optional
        Name of the atom(s) to add.
    
    resname : `str`, keyword-only, optional
        Name of the residue(s) to add.

    nbforce : `openmm.NonbondedForce`, keyword-only, optional
        Built-in hybrid pair potential object containing the
        Lennard-Jones and Coulomb potentials.
    
    charge : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        Charge :math:`q` of the atoms for use in the Coulomb potential
        in `nbforce`. 
        
        **Reference unit**: :math:`\mathrm{e}`.

    sigma : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        :math:`\sigma` parameter of the Lennard-Jones potential in
        `nbforce`. 
        
        **Reference unit**: :math:`\mathrm{nm}`.

    epsilon : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        :math:`\epsilon` parameter of the Lennard-Jones potential in
        `nbforce`. 
        
        **Reference unit**: :math:`\mathrm{kJ/mol}`.

    cnbforces : `dict`, keyword-only, optional
        Custom pair potential objects implementing other non-standard
        pair potentials and their corresponding per-particle parameters. 

        **Example**: :code:`{gauss: (0.3 * unit.nanometer,)}`, where 
        `gauss` is a custom Gaussian potential 
        :func:`mdhelper.openmm.pair.gauss`.
    """

    per_chain = chain is None
    for _ in range(N):
        if system:
            system.addParticle(mass)
        if per_chain:
            chain = topology.addChain()
        residue = topology.addResidue(resname, chain)
        topology.addAtom(name, element, residue)
        if nbforce is not None:
            nbforce.addParticle(charge, sigma, epsilon)
        for cnbforce, param in cnbforces.items():
            cnbforce.addParticle(param)

def slab_correction(
        system: openmm.System, topology: app.Topology,
        nbforce: Union[openmm.NonbondedForce, openmm.CustomNonbondedForce],
        temp: Union[float, unit.Quantity], fric: Union[float, unit.Quantity],
        dt: Union[float, unit.Quantity], axis: int = 2, *, 
        charge_index: int = 0, z_scale: float = 3, method: str = "force"
    ) -> openmm.Integrator:

    r"""
    Implements a slab correction so that efficient three-dimensional
    Ewald methods can continue to be used to evaluate the electrostatics
    for systems that are periodic in the :math:`x`- and
    :math:`y`-directions but not the :math:`z`-direction. Effectively,
    the system is treated as if it were periodic in the
    :math:`z`-direction, but with empty volume added between the slabs
    and the slab–slab interactions removed.

    For electroneutral systems, the Yeh–Berkowitz correction [1]_ is
    applied:

    .. math::

       \begin{gather*}
         U^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}M_z^2\\
         u_i^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}q_i\left(z_iM_z
         -\frac{\sum_i q_iz_i^2}{2}\right)\\
         f_{i,z}^\mathrm{corr}=-\frac{N_A}{\varepsilon_0V}q_iM_z
       \end{gather*}

    For systems with a net electric charge, the Ballenegger–Arnold–Cerdà
    correction [2]_ is applied instead:

    .. math::

       \begin{gather*}
         U^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}
         \left(M_z^2-q_\mathrm{tot}\sum_i q_iz_i^2
         -\frac{q_\mathrm{tot}^2L_z^2}{12}\right)\\
         u_i^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}q_i
         \left(z_iM_z-\frac{\sum_i q_iz_i^2+q_\mathrm{tot}z_i^2}{2}
         -\frac{q_\mathrm{tot}L_z^2}{12}\right)\\
         f_{i,z}^\mathrm{corr}=-\frac{N_A}{\varepsilon_0V}q_i
         \left(M_z-q_\mathrm{tot}z_i\right)
       \end{gather*}

    Note that the the relative permittivity
    :math:`\varepsilon_\mathrm{r}` does not appear in the equations
    above because it is accounted for by scaling the particle charges in
    the Coulomb potential.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.

    nbforce : `openmm.NonbondedForce` or `openmm.CustomNonbondedForce`
        Pair potential containing particle charge information. 
        
        .. note::
           
           It is assumed that the charge :math:`q` information is the
           first per-particle parameter stored in `nbforce`. If not, the
           index can be specified in `charge_index`.

    temp : `float` or `openmm.unit.Quantity`
        System temperature :math:`T`. 
        
        **Reference unit**: :math:`\mathrm{K}`.
    
    fric : `float` or `openmm.unit.Quantity`
        Friction coefficient :math:`\gamma` that couples the system to
        the heat bath.
        
        **Reference unit**: :math:`\mathrm{ps}^{-1}`.
    
    dt : `float` or `openmm.unit.Quantity`
        Integration step size :math:`\Delta t`. 
        
        **Reference unit**: :math:`\mathrm{ps}`.
    
    axis : `int`, default: :code:`2`
        Axis along which to apply the slab correction, with :math:`x`
        being :code:`0`, :math:`y` being :code:`1`, and :math:`z` being
        :code:`2`.

    charge_index : `int`, keyword-only, default: :code:`0`
        Index of charge :math:`q` information in the per-particle
        parameters stored in `nbforce`.
    
    z_scale : `float`, keyword-only, default: :code:`3`
        :math:`z`-dimension scaling factor.

    method : `str`, keyword-only, default: :code:`"force"`
        Slab correction methodology.

        .. container::

           **Valid values**:
            
           * :code:`"force"`: Collective implementation via 
             :class:`openmm.openmm.CustomExternalForce` and 
             :class:`openmm.openmm.CustomCVForce` [3]_. This is 
             generally the most efficient.
           * :code:`"integrator"`: Per-particle implementation via an
             :class:`openmm.openmm.CustomIntegrator`.

    Returns
    -------
    integrator : `openmm.Integrator` or `openmm.CustomIntegrator`
        Integrator that simulates a system using Langevin dynamics, with
        the LFMiddle discretization.

    References
    ----------
    .. [1] Yeh, I.-C.; Berkowitz, M. L. Ewald Summation for Systems with
       Slab Geometry. *J. Chem. Phys.* **1999**, *111* (7), 3155–3162.
       https://doi.org/10.1063/1.479595.

    .. [2] Ballenegger, V.; Arnold, A.; Cerdà, J. J. Simulations of
       Non-Neutral Slab Systems with Long-Range Electrostatic
       Interactions in Two-Dimensional Periodic Boundary Conditions.
       *J. Chem. Phys.* **2009**, *131* (9), 094107.
       https://doi.org/10.1063/1.3216473.

    .. [3] Son, C. Y.; Wang, Z.-G. Image-Charge Effects on Ion
       Adsorption near Aqueous Interfaces. *Proc. Natl. Acad. Sci.
       U.S.A.* **2021**, *118* (19), e2020615118.
       https://doi.org/10.1073/pnas.2020615118.
    """

    # Get system dimensions
    dims = np.array(topology.getUnitCellDimensions() / unit.nanometer) \
           * unit.nanometer
    pbv = system.getDefaultPeriodicBoxVectors()

    # Scale system z-dimension by specified z-scale
    if z_scale < 2:
        wmsg = ("A z-scaling factor that is less than 2 may introduce "
                "unwanted slab–slab interactions. The recommended "
                "value is 3.")
        warnings.warn(wmsg)
    elif z_scale > 5:
        wmsg = ("A z-scaling factor that is greater than 5 may "
                "penalize performance. The recommended value is 3.")
        warnings.warn(wmsg)
    dims[axis] *= z_scale
    pbv[axis] *= z_scale

    # Set new system dimensions
    topology.setUnitCellDimensions(dims)
    system.setDefaultPeriodicBoxVectors(*pbv)

    # Obtain particle charge information
    dtype = type(nbforce.getParticleParameters(0)[charge_index])
    qs = np.fromiter(
        (nbforce.getParticleParameters(i)[charge_index] / unit.elementary_charge
         for i in range(nbforce.getNumParticles())) if dtype == unit.Quantity \
        else (nbforce.getParticleParameters(i)[charge_index]
              for i in range(nbforce.getNumParticles())),
        dtype=float
    )
    neutral = qs.min() == qs.max()
    if not neutral:
        q_tot = qs.sum()
        electroneutral = np.isclose(q_tot, 0)

    # Calculate coefficient for slab correction
    coef = unit.AVOGADRO_CONSTANT_NA / \
           (2 * VACUUM_PERMITTIVITY * dims[0] * dims[1] * dims[2])

    # Get letter representation of axis for formula
    x = chr(120 + axis)

    if neutral:

        # Instantiate an integrator that simulates a system using
        # Langevin dynamics, with the LFMiddle discretization
        integrator = openmm.LangevinMiddleIntegrator(temp, fric, dt)

    else:
        if method == "integrator":

            # Implement an integrator that simulates a system using Langevin
            # dynamics, with the LFMiddle discretization
            integrator = openmm.CustomIntegrator(dt)
            integrator.addGlobalVariable("a", np.exp(-fric * dt))
            integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * fric * dt)))
            integrator.addGlobalVariable(
                "kT", unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temp
            )
            integrator.addPerDofVariable("x1", 0)
            integrator.addUpdateContextState()
            integrator.addComputePerDof("v", "v+dt*f/m")
            integrator.addConstrainVelocities()
            integrator.addComputePerDof("x", "x+dt*v/2")
            integrator.addComputePerDof("v", "a*v+b*sqrt(kT/m)*gaussian")
            integrator.addComputePerDof("x", "x+dt*v/2")
            integrator.addComputePerDof("x1", "x")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "v+(x-x1)/dt")

            # Initialize per-degree-of-freedom variable q for charge
            integrator.addPerDofVariable("q", 0)

            # Add global dipole moment computation to integrator
            integrator.addComputeSum("M_z", "q*x")
            integrator.addComputeSum("M_zz", "q*x^2")

            # Give particle charge information to integrator
            q_vectors = np.zeros((len(qs), 3), dtype=float)
            q_vectors[:, axis] = qs
            integrator.setPerDofVariableByName("q", q_vectors)

            # Implement per-particle slab correction
            if electroneutral:
                slab_corr = openmm.CustomExternalForce(
                    f"coef*q*({x}*M_z-M_zz/2)"
                )
            else:
                slab_corr = openmm.CustomExternalForce(
                    f"coef*q*({x}*M_z-(M_zz+q_tot*{x}^2)/2-q_tot*dim_z^2/12)"
                )
                slab_corr.addGlobalParameter("dim_z", dims[axis])
                slab_corr.addGlobalParameter("q_tot", q_tot)
            slab_corr.addGlobalParameter("M_z", 0)
            slab_corr.addGlobalParameter("M_zz", 0)
            slab_corr.addGlobalParameter("coef", coef)
            slab_corr.addPerParticleParameter("q")

            # Register real particles to the slab correction
            for i, q in enumerate(qs):
                slab_corr.addParticle(i, (q,))

        elif method == "force":
            
            # Instantiate an integrator that simulates a system using
            # Langevin dynamics, with the LFMiddle discretization
            integrator = openmm.LangevinMiddleIntegrator(temp, fric, dt)

            # Calculate instantaneous system dipole
            M_z = openmm.CustomExternalForce(f"q*{x}")
            M_z.addPerParticleParameter("q")

            # Implement collective slab correction
            if electroneutral:
                slab_corr = openmm.CustomCVForce("coef*M_z^2")
            else:
                M_zz = openmm.CustomExternalForce(f"q*{x}^2")
                M_zz.addPerParticleParameter("q")
                slab_corr = openmm.CustomCVForce(
                    "coef*(M_z^2-q_tot*M_zz-q_tot^2*dim_z^2/12)"
                )
                slab_corr.addCollectiveVariable("M_zz", M_zz)
                slab_corr.addGlobalParameter("dim_z", dims[axis])
                slab_corr.addGlobalParameter("q_tot", q_tot)
            slab_corr.addCollectiveVariable("M_z", M_z)
            slab_corr.addGlobalParameter("coef", coef)
        
            # Register real particles to the slab correction
            for i, q in enumerate(qs):
                M_z.addParticle(i, (q,))
                if not electroneutral:
                    M_zz.addParticle(i, (q,))

        # Register slab correction to the system
        system.addForce(slab_corr)

    return integrator

def image_charges(
        system: openmm.System, topology: app.Topology,
        positions: Union[np.ndarray, unit.Quantity],
        temp: Union[float, unit.Quantity], fric: Union[float, unit.Quantity],
        dt: Union[float, unit.Quantity], axis: int = 2, *,
        wall_indices: np.ndarray = None,
        nbforce: openmm.NonbondedForce = None,
        cnbforces: Union[Iterable[openmm.CustomNonbondedForce],
                         dict[openmm.CustomNonbondedForce, Iterable[Any]]] = {},
        params: Iterable[Iterable[Any]] = ()
    ) -> tuple[unit.Quantity, openmm.Integrator]:

    r"""
    Implements the method of image charges for perfectly conducting
    boundaries (with a relative permittivity of
    :math:`\varepsilon_\mathrm{r}=\infty`).

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.
    
    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the real system. 

        **Shape**: :math:`(N,/,3)`.
        
        **Reference unit**: :math:`\mathrm{nm}`.

    temp : `float` or `openmm.unit.Quantity`
        System temperature :math:`T`. 
        
        **Reference unit**: :math:`\mathrm{K}`.

    fric : `float` or `openmm.unit.Quantity`
        Friction coefficient :math:`\gamma` that couples the system to
        the heat bath.
        
        **Reference unit**: :math:`\mathrm{ps}^{-1}`.
    
    dt : `float` or `openmm.unit.Quantity`
        Integration step size :math:`\Delta t`. 
        
        **Reference unit**: :math:`\mathrm{ps}`.

    axis : `int`, default: :code:`2`
        Axis along which to apply the method of image charges, with 
        :math:`x` being :code:`0`, :math:`y` being :code:`1`, and 
        :math:`z` being :code:`2`.

    nbforce : `openmm.NonbondedForce`, keyword-only, optional
        Standard Lennard-Jones (LJ) and Coulomb pair potential used in
        OpenMM. For the image charges, the charges :math:`q` are flipped
        while the LJ parameters :math:`\sigma` and :math:`\epsilon` are
        set to :math:`0`.

    cnbforces : `dict`, keyword-only, optional
        Custom pair potential objects implementing other non-standard
        pair potentials. The keys are the potentials and the values are
        the corresponding per-particle parameters to use, which can be
        provided as follows:

        .. container::

           * `None`: All per-particle parameters set to :math:`0`.
           * `dict`: Information on how to transform the real particle
             parameters into those for the image charges. The valid
             (optional) key-value pairs are:

             * :code:`"charge"`: Index (`int`) where the charge
               :math:`q` information is stored.
             * :code:`"zero"`: Index (`int`) or indices (`list`, 
               `slice`, or `numpy.ndarray`) of parameters to zero out.
             * :code:`"replace"`: `dict` containing key-value pairs of
               indices (`int`) of parameters to change and their
               replacement value(s). If the value is an `int`, all
               particles receive that value for the current pair
               potential. If the value is another `dict`, the new
               parameter value (value of inner `dict`) is determined by
               the current parameter value (key of inner `dict`).

        To prevent unintended behavior, ensure that each parameter index
        for a custom pair potential is used only once across the keys
        listed above. If one does appear more than once, the order
        in which parameter values are updated follows that of the list
        above.

        See the Examples section below for a simple illustration of the
        possibilities outlined above.

    wall_indices : `numpy.ndarray`, keyword-only, optional
        Atom indices corresponding to wall particles. If not provided,
        the wall particles are guessed using the system dimensions and
        `axis`.

    Returns
    -------
    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the real system and 
        their images.

        **Shape**: :math:`(2N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.
    
    integrator : `openmm.Integrator`
        Integrator that simulates a system using Langevin dynamics, with
        the LFMiddle discretization.

    Examples
    --------
    Prepare the `cnbforces` `dict` argument for a system with the
    following custom pair potentials:

    * :code:`PairGauss`: Gaussian potential with per-particle
      parameters `type` and `beta`.
        
      * `type` is the particle type, and is used to look up a
        tabulated function for parameter values. As such, the new
        values will vary and depend on the old ones. For example, for
        a system with :math:`2` types of particles, :math:`0` becomes
        :math:`2` and :math:`1` becomes :math:`3`.
      * `beta` is assumed to be constant, with the value for the real
        particles differing from that for image charges. For this
        example, the new value is :math:`\beta = 42` (arbitrary
        units).

    * :code:`PairWCA`: Weeks–Chander–Andersen potential with per-
      particle parameters `sigma` and `epsilon`, both of which
      should be set to :math:`0` to turn off this potential for
      image charges.
    * :code:`PairCoulRecip`: Smeared Coulomb potential with per-
      particle parameter `charge`, which should be flipped for image
      charges, and `dummy`, a value that should be set to :math:`0`.

    .. code::

       NEW_BETA_VALUE = 42
       cnbforces = {
           PairGauss: {"replace": {0: {0: 2, 1: 3}, 1: NEW_BETA_VALUE}},
           PairWCA: None,
           PairCoulRecip: {"charge": 0, "zero": 1}
       }
    """

    if not FOUND_CONSTVPLUGIN:
        emsg = ("The required constant potential MD integrator plugin "
                "(https://github.com/scychon/openmm_constV) was not found. "
                "As such, the method of image charges is unavailable.")
        raise ImportError(emsg)

    # Get system information
    dims = np.array(topology.getUnitCellDimensions() / unit.nanometer) \
           * unit.nanometer
    pbv = system.getDefaultPeriodicBoxVectors()
    N_real = positions.shape[0]
    if isinstance(positions, unit.Quantity):
        positions /= unit.nanometer

    # Guess indices of left and right walls if not provided
    if wall_indices is None:
        wall_indices = np.concatenate(
            ((positions[:, axis] == 0).nonzero()[0],
            (positions[:, axis] == dims[axis] / unit.nanometer).nonzero()[0])
        )

    # Mirror particle positions
    flip = np.ones(3, dtype=int)
    flip[axis] *= -1
    positions = np.concatenate((positions, positions * flip)) * unit.nanometer

    # Update and set new system dimensions
    dims[axis] *= 2
    topology.setUnitCellDimensions(dims)
    pbv[axis] *= 2
    system.setDefaultPeriodicBoxVectors(*pbv)

    # Instantiate an integrator that simulates a system using
    # Langevin dynamics and updates the image charge positions
    integrator = ConstVLangevinIntegrator(temp, fric, dt)

    # Register image charges to the system, topology, and force field
    chains_ic = [topology.addChain() for _ in range(topology.getNumChains())]
    residues_ic = [topology.addResidue(f"IC_{r.name}", 
                                       chains_ic[r.chain.index])
                   for r in list(topology.residues())]
    for i, atom in enumerate(list(topology.atoms())):
        system.addParticle(0)
        topology.addAtom(f"IC_{atom.name}", atom.element,
                         residues_ic[atom.residue.index])
        if nbforce is not None:
            nbforce.addParticle(
                0 if i in wall_indices
                else -nbforce.getParticleParameters(i)[0], 0, 0)
        for force, kwargs in cnbforces.items():
            params = np.array(force.getParticleParameters(i))
            if kwargs is None:
                params[:] = 0
            else:
                if "charge" in kwargs:
                    params[kwargs["charge"]] *= 0 if i in wall_indices else -1
                if "zero" in kwargs:
                    params[kwargs["zero"]] = 0
                if "replace" in kwargs:
                    for index, value in kwargs["replace"].items():
                        params[index] = value[params[index]] \
                                        if isinstance(value, dict) \
                                        else value
            force.addParticle(params)

    # Add existing particle exclusions to mirrored image charges
    for i in range(nbforce.getNumExceptions()):
        i1, i2, qq = nbforce.getExceptionParameters(i)[:3]
        nbforce.addException(N_real + i1, N_real + i2, qq, 0, 0)
        for force in cnbforces:
            i1, i2 = force.getExclusionParticles(i)
            force.addExclusion(N_real + i1, N_real + i2)

    # Prevent wall particles from interacting with their mirrors
    for i in wall_indices:
        nbforce.addException(i, N_real + i, 0, 0, 0)
        for force in cnbforces:
            force.addExclusion(i, N_real + i)

    return positions, integrator

def electric_field(
        system: openmm.System, nbforce: openmm.NonbondedForce,
        efield: Union[float, unit.Quantity,], *, axis: int = 2,
        charge_index: int = 0, atom_indices: Union[int, Iterable] = None
    ) -> None:

    r"""
    Adds an electric field to all charged particles by adding a force
    :math:`f_i=q_iE` in the axis specified in `axis`, where :math:`q_i`
    is the per-particle charge and :math:`E` is the electric field.

    .. hint::

       The following schematic shows how directionality is handled:

       .. code::

          |-| (-) ---> |+|
          |-| <-- E -- |+|
          |-| <--- (+) |+|

       With a positive potential difference 
       (:math:`\Delta V>0\;\mathrm{V}`), the electric field is negative
       (:math:`E<0\;\mathrm{V/m}`) such that it is pointing from the 
       left (positive) plate to the right (negative) plate. If an ion 
       has a positive charge (:math:`q_i>0\;\mathrm{e}`), the force will
       be negative, indicating that it will be drawn to the left plate,
       and vice versa.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    nbforce : `openmm.NonbondedForce` or `openmm.CustomNonbondedForce`
        Pair potential containing particle charge information. 
        
        .. note::
           
           It is assumed that the charge :math:`q` information is the
           first per-particle parameter stored in `nbforce`. If not, the
           index can be specified in `charge_index`.
    
    efield : `float` or `openmm.unit.Quantity`
        Electric field :math:`E`.

        **Reference unit**: :math:`\mathrm{kJ/(mol\cdot nm\cdot e)}`.
    
    axis : `int`, keyword-only, default: :code:`2`
        Axis along which the walls are placed. :code:`0`, :code:`1`, and
        :code:`2` correspond to :math:`x`, :math:`y`, and :math:`z`,
        respectively.

    charge_index : `int`, keyword-only, default: :code:`0`
        Index of charge :math:`q` information in the per-particle
        parameters stored in `nbforce`.

    atom_indices : `int` or array-like, keyword-only, optional
        Indices of atoms to apply the electric field to. By default,
        the electric field is applied to all atoms, but this can be
        computationally expensive when there are charged particles that
        do not need to be included, such as image charges.
    """

    # Get letter representation of axis for formula
    x = chr(120 + axis)

    # Get indices of atoms that are affected by the electric field
    if atom_indices is None:
        atom_indices = range(nbforce.getNumParticles())
    elif isinstance(atom_indices, int):
        atom_indices = range(atom_indices)

    # Create and register particles to the electric field
    efield = openmm.CustomExternalForce(f"-q*E*{x}")
    efield.addGlobalParameter("E", efield)
    efield.addPerParticleParameter('q')

    for i in atom_indices:
        q = nbforce.getParticleParameters(i)[charge_index]
        if isinstance(q, unit.Quantity):
            q /= unit.elementary_charge
        if not np.isclose(q, 0):
            efield.addParticle(i, (q,))
    
    system.addForce(efield)