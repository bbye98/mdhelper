"""
Molecular structure
===================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains algorithms for computing structural information
for a collection of particles.
"""

from typing import Union

import MDAnalysis as mda
import numpy as np

from .. import ArrayLike

def center_of_mass(
        group: mda.AtomGroup = None, grouping: str = None, *,
        masses: ArrayLike = None, positions: ArrayLike = None,
        images: np.ndarray = None, dims: np.ndarray = None,
        n_groups: int = None, raw: bool = False
    ) -> np.ndarray:
    
    r"""
    Computes the center(s) of mass for a collection of particles.

    For a group with :math:`N` particles with masses :math:`m_i` and
    positions :math:`\mathbf{r}_i`, the center of mass is defined as

    .. math::

       \mathbf{R}_\mathrm{com}=\dfrac{\sum_{i=1}^N m_i
       \mathbf{r}_i}{\sum_{i=1}^N m_i}

    .. note::

       This function supports a wide variety of inputs, depending on 
       how the particle information is provided and what should be 
       calculated.

       When an :class:`MDAnalysis.core.groups.AtomGroup` object is 
       provided, the particle masses and positions are extracted from 
       it. If the :code:`AtomGroup` abides by the standard topological 
       heirarchy, you can specify the desired grouping and the 
       appropriate center(s) of mass will be calculated. Otherwise, if
       and only if the :code:`AtomGroup` contains equisized or identical
       groups corresponding to the desired grouping, you can provide the 
       total number of groups and the particle masses and positions will
       be distributed accordingly. If the :code:`AtomGroup` does not 
       have the correct structural information and the residues or 
       segments do not contain the same number of atoms, see the final
       paragraph.

       If the trajectory is not unwrapped, the number of periodic 
       boundary crossings (and optionally, the system dimensions if they
       are not embedded in the :code:`AtomGroup`) can be provided.

       Alternatively, the *unwrapped* particle masses and positions can 
       be provided directly as a :class:`numpy.ndarray`, list, or tuple.
       To calculate the overall center of mass, the array-like object
       holding the masses should be one-dimensional, while that 
       containing the positions should be two-dimensional. To calculate
       center(s) of mass for group(s), the array-like object holding the
       masses should be two-dimensional (indexing: group, particle 
       mass), while that containing the positions should be 
       three-dimensional (indexing: group, particle position, 
       dimension). When a list or tuple is used, the inner arrays do not
       have to be homogeneously shaped, thus allowing you to calculate 
       the centers of mass for different residues or segments.

       You may also provide only one of the particle masses or 
       positions, in which case the missing information will be
       retrieved from the :code:`AtomGroup`. This is generally not
       recommended since the shapes of the provided and retrieved
       arrays may be incompatible.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`, optional
        Collection of atoms to compute the center(s) of mass for. If not
        provided, the particle masses and posititions must be provided
        directly in `masses` and `positions`.

    grouping : `str`, optional
        Determines which center of mass is calculated if particle 
        massses and positions are retrieved from `group`.

        .. container::

           **Valid values**:

           * :code:`None`: Center of mass of all particles in `group`.
           * :code:`"residues"`: Centers of mass for each residue in
             `group`.
           * :code:`"segments"`: Centers of mass for each chain in 
             `group`.

    masses : array-like, keyword-only, optional
        Particle masses.

        .. container::

           **Shape**:

           The general shape is :math:`(N,)`.

           For equisized or identical groups,

           * :math:`(N,)` for the overall center of mass,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for the residue
             center(s) of mass, where :math:`N_\mathrm{res}` is
             the number of residues, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg}` for the segment
             center(s) of mass, where :math:`N_\mathrm{seg}` is
             the number of segments.

           For groups with different numbers of atoms, the array-like
           object should contain inner lists or tuples holding the
           masses of the particles in each group. 

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : array-like, keyword-only, optional
        Unwrapped particle positions.

        .. container::

           **Shape**:

           The general shape is :math:`(N,\,3)`.

           For equisized or identical groups,

           * :math:`(N,\,3)` for the overall center of mass,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for the 
             residue center(s) of mass, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for the 
             segment center(s) of mass.

           For groups with different numbers of atoms, the array-like
           object should contain inner lists or tuples holding the
           coordinates of the particles in each group. 

        **Reference unit**: :math:`\mathrm{Å}`.

    images : `numpy.ndarray`, keyword-only, optional
        Image flags for the particles in `group`. Must be provided for 
        correct results if particle positions are wrapped.

        **Shape**: :math:`(N,\,3)`.

    dims : `numpy.ndarray`, keyword-only, optional
        System dimensions. Must be provided if `images` is provided and
        `group` does not contain the system dimensions.

        **Shape**: :math:`(3,)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    n_groups : `int`, keyword-only, optional
        Number of residues or segments. Must be provided if `group` has
        an irregular topological heirarchy or the `masses` and 
        `positions` arrays have the generic shapes.
    
    raw : `bool`, keyword-only, default: :code:`False`
        Determines whether particle masses and positions are returned if
        they were retrieved from `group`.

    Returns
    -------
    com : `numpy.ndarray`
        Center(s) of mass.

        .. container::

           **Shape**:

           * :math:`(3,)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,3)` for 
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,3)` for 
             :code:`grouping="segments"`.

    masses : `numpy.ndarray`
        Particle masses. Only returned if `group` was provided
        and contains equisized or identical groups, and 
        :code:`raw=True`.

        .. container::

           **Shape**:

           * :math:`(N,)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for 
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg})` for 
             :code:`grouping="segments"`.

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : `numpy.ndarray`
        Unwrapped particle positions. Only returned if `group` was 
        provided and contains equisized or identical groups, and 
        :code:`raw=True`.

        .. container::

           **Shape**:

           * :math:`(N,\,3)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for 
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for 
             :code:`grouping="segments"`.

        **Reference unit**: :math:`\mathrm{Å}`.
    """
    
    # Check whether grouping is valid
    if grouping not in {None, "residues", "segments"}:
        emsg = (f"Invalid grouping: '{grouping}'. Valid options are "
                "None, 'residues', and 'segments'.")
        raise ValueError(emsg)
    
    # Get particle masses and positions from the trajectory, if necessary
    missing = (masses is None, positions is None)
    if any(missing):

        # Ensure a trajectory is available
        if group is None:
            emsg = ("Either a group of atoms or atom positions and "
                    "masses must be provided.")
            raise ValueError(emsg)

        # Check whether the groups are identical
        if grouping:
            groups = getattr(group, grouping)
            same = all(r.atoms.n_atoms == groups[0].atoms.n_atoms 
                        for r in groups)
        else:
            same = True

        # Calculate and return the centers of mass for different groups
        # here if unwrapping and the mass and position arrays are not 
        # needed
        if not same and images is None and not raw:
            return np.array([g.atoms.center_of_mass() for g in groups])

        # Get particle positions, if necessary
        if missing[1]:
            positions = group.positions

            # Unwrap particle positions, if necessary
            if images is not None:
                if dims is None:
                    try:
                        dims = group.dimensions[:3]
                    except TypeError:
                        emsg = ("The number of periodic boundary "
                                "crossings was provided, but no "
                                "system dimensions were provided or "
                                "found in the trajectory.")
                        raise ValueError(emsg)
                positions += images * dims[:3]

        if same:

            # Get particle masses, if necessary
            if missing[0]:
                masses = group.masses

            # Reshape the mass and position arrays, if necessary
            if grouping or n_groups:
                shape = (n_groups, -1, 3) if n_groups \
                        else (getattr(group, f"n_{grouping}"), -1, 3)
                masses = masses.reshape(shape[:-1])
                positions = positions.reshape(shape)
        else:

            # Get particle masses, if necessary
            if missing[0]:
                masses = [g.atoms.masses for g in groups]

            # Reshape the position array, if necessary
            if missing[1]:
                positions = [positions[g.atoms.ix] for g in groups]

    else:

        # Try to convert arrays to NumPy arrays if they are not already
        # to take advantage of vectorized operations later
        if not isinstance(positions, np.ndarray):
            try:
                positions = np.array(positions)
            except ValueError:
                pass
        if not isinstance(masses, np.ndarray):
            try:
                masses = np.array(masses)
            except ValueError:
                pass
        if type(masses) != type(positions):
            emsg = ("The shapes of the arrays containing the particle "
                    "masses and positions are incompatible.")
            raise ValueError(emsg)

        # Reshape the mass and position arrays based on the specified
        # number of groups
        if n_groups and isinstance(positions, np.ndarray):
            masses = masses.reshape((n_groups, -1))
            positions = positions.reshape((n_groups, -1, 3))

    # Calculate the center(s) of mass for the specified grouping
    if isinstance(positions, np.ndarray):
        com = np.einsum("...a,...ad->...d", masses, positions) \
                  / masses.sum(axis=-1, keepdims=True)
    else:
        com = np.array([np.dot(m, p) / m.sum() 
                       for m, p in zip(masses, positions)], dtype=float)
    
    # Return the center(s) of mass
    if raw and any(missing):

        # Also return the particle masses and positions, if desired
        return com, masses, positions
    return com

def radius_of_gyration(
        group: mda.AtomGroup = None, grouping: str = None, *,
        positions: ArrayLike = None, masses: ArrayLike = None,
        com: ArrayLike = None, images: np.ndarray = None, 
        dims: np.ndarray = None, n_groups: int = None, components: bool = False
    ) -> Union[float, np.ndarray]:
    
    r"""
    Computes the radii of gyration for a collection of particles.

    For a group with :math:`N` particles with masses :math:`m_i` and
    positions :math:`\mathbf{r}_i`, the radius of gyration is defined as

    .. math::

        R_\mathrm{g}=\sqrt{
        \frac{\sum_i^N m_i\|\mathbf{r}_i-\mathbf{R}_\mathrm{com}\|^2}
        {\sum_i^N m_i}}

    where :math:`\mathbf{R}_\mathrm{com}` is the center of mass.

    Alternatively, the radii of gyration around the coordinate axes can 
    be calculated by only summing the radii components orthogonal to
    each axis. For example, the radius of gyration around the 
    :math:`x`-axis is

    .. math::
     
       R_{\mathrm{g},x}=\sqrt{
       \frac{\sum_i^N m_i[(\mathbf{r}_{i,y}-\mathbf{R}_{\mathrm{com},y})^2
       +(\mathbf{r}_{i,z}-\mathbf{R}_{\mathrm{com},z})^2]}{\sum_i^N m_i}}

    .. note::

       This function supports a wide variety of inputs, depending on 
       how the particle information is provided and what should be 
       calculated.

       When an :class:`MDAnalysis.core.groups.AtomGroup` object is 
       provided, the particle masses and positions are extracted from 
       it. If the :code:`AtomGroup` abides by the standard topological 
       heirarchy, you can specify the desired grouping and the 
       appropriate radii of gyration will be calculated. Otherwise, if
       and only if the :code:`AtomGroup` contains equisized or identical
       groups corresponding to the desired grouping, you can provide the 
       total number of groups and the particle masses and positions will
       be distributed accordingly. If the :code:`AtomGroup` does not 
       have the correct structural information and the residues or 
       segments do not contain the same number of atoms, see the final
       paragraph.

       If the trajectory is not unwrapped, the number of periodic 
       boundary crossings (and optionally, the system dimensions if they
       are not embedded in the :code:`AtomGroup`) can be provided.

       Alternatively, the *unwrapped* particle masses and positions can 
       be provided directly as a :class:`numpy.ndarray`, list, or tuple.
       To calculate the overall radius of gyration, the array-like object
       holding the masses should be one-dimensional, while that 
       containing the positions should be two-dimensional. To calculate
       radii of gyration for group(s), the array-like object holding the
       masses should be two-dimensional (indexing: group, particle 
       mass), while that containing the positions should be 
       three-dimensional (indexing: group, particle position, 
       dimension). When a list or tuple is used, the inner arrays do not
       have to be homogeneously shaped, thus allowing you to calculate 
       the radii of gyration for different residues or segments.

       You may also provide only one of the particle masses or 
       positions, in which case the missing information will be
       retrieved from the :code:`AtomGroup`. This is generally not
       recommended since the shapes of the provided and retrieved
       arrays may be incompatible.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`, optional
        Collection of atoms to compute the radii of gyration for. If not
        provided, the particle masses and posititions must be provided
        directly in `masses` and `positions`.

    grouping : `str`, optional
        Determines which radius of gyration is calculated if particle 
        massses and positions are retrieved from `group`.

        .. container::

           **Valid values**:

           * :code:`None`: Radius of gyration of all particles in 
             `group`.
           * :code:`"residues"`: Radius of gyration for each residue in
             `group`.
           * :code:`"segments"`: Radius of gyration for each chain in 
             `group`.
    
    masses : array-like, keyword-only, optional
        Particle masses.

        .. container::

           **Shape**:

           The general shape is :math:`(N,)`.

           For equisized or identical groups,

           * :math:`(N,)` for the overall radius of gyration,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for the residue
             radii of gyration, where :math:`N_\mathrm{res}` is
             the number of residues, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg}` for the segment
             radii of gyration, where :math:`N_\mathrm{seg}` is
             the number of segments.

           For groups with different numbers of atoms, the array-like
           object should contain inner lists or tuples holding the
           masses of the particles in each group. 

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : array-like, keyword-only, optional
        Unwrapped particle positions.

        .. container::

           **Shape**:

           The general shape is :math:`(N,\,3)`.

           For equisized or identical groups,

           * :math:`(N,\,3)` for the overall radius of gyration,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for the 
             residue radii of gyration, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for the 
             segment radii of gyration.

           For groups with different numbers of atoms, the array-like
           object should contain inner lists or tuples holding the
           coordinates of the particles in each group. 

        **Reference unit**: :math:`\mathrm{Å}`.

    com : `numpy.ndarray`, keyword-only, optional
        Center(s) of mass.

        .. container::

           **Shape**:

           * :math:`(3,)` for the overall radius of gyration.
           * :math:`(N_\mathrm{res},\,3)` for the residue radii of 
             gyration.
           * :math:`(N_\mathrm{seg},\,3)` for the segment radii of
             gyration.

    images : `numpy.ndarray`, keyword-only, optional
        Image flags for the particles in `group`. Must be provided for 
        correct results if particle positions are wrapped.

        **Shape**: :math:`(N,\,3)`.

    dims : `numpy.ndarray`, keyword-only, optional
        System dimensions. Must be provided if `images` is provided and
        `group` does not contain the system dimensions.

        **Shape**: :math:`(3,)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    n_groups : `int`, keyword-only, optional
        Number of residues or segments. Must be provided if `group` has
        an irregular topological heirarchy or the `masses` and 
        `positions` arrays have the generic shapes.

    components : `bool`, keyword-only, default: :code:`False`
        Specifies whether the components of the radii of gyration are
        calculated and returned instead.

    Returns
    -------
    gyradii : `float` or `numpy.ndarray`
        Radii of gyration.

        .. container::

           **Shape**:

           * Scalar for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},)` for :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},)` for :code:`grouping="segments"`.

        **Reference unit**: :math:`\mathrm{Å}`.
    """

    # Check whether grouping is valid
    if grouping not in {None, "residues", "segments"}:
        emsg = (f"Invalid grouping: '{grouping}'. Valid options are "
                "None, 'residues', and 'segments'.")
        raise ValueError(emsg)

    # Get particle masses and positions from the trajectory and the
    # center(s) of mass, if necessary
    missing = (masses is None, positions is None, com is None)
    if any(missing[:2]):
        com, masses, positions = center_of_mass(group, grouping, masses=masses, 
                                                positions=positions, raw=True,
                                                images=images, dims=dims)
    elif missing[2]:
        com = center_of_mass(masses=masses, positions=positions, n_groups=n_groups)

    if isinstance(positions, np.ndarray):
        if components:
            cpos = (positions - np.expand_dims(com, axis=positions.ndim - 2)) ** 2
            if grouping or n_groups:

                # Compute the radii of gyration in each direction for 
                # equisized or identical group(s)
                return np.sqrt(
                    np.einsum("ga,gad->gd", masses, 
                              np.stack((cpos[:, :, (1, 2)].sum(axis=2), 
                                        cpos[:, :, (0, 2)].sum(axis=2), 
                                        cpos[:, :, (0, 1)].sum(axis=2)), axis=2))
                    / masses.sum(axis=1, keepdims=True)
                )
            
            # Compute the radius of gyration in each direction for all
            # atoms
            return np.sqrt(
                np.dot(masses, np.hstack(
                    (cpos[:, (1, 2)].sum(axis=1, keepdims=True), 
                     cpos[:, (0, 2)].sum(axis=1, keepdims=True), 
                     cpos[:, (0, 1)].sum(axis=1, keepdims=True))
                )) / masses.sum()
            )
        elif grouping or n_groups:

            # Compute the overall radii of gyration for equisized or 
            # identical group(s)
            return np.sqrt(np.einsum("ga,gad->gd", masses,
                                     (positions - com[:, None]) ** 2).sum(axis=1) \
                   / masses.sum(axis=1))
        
        # Compute the overall radius of gyration for all atoms
        return np.sqrt(np.dot(masses, (positions - com) ** 2).sum() \
               / masses.sum())
    if components:

        # Compute the radii of gyration in each direction for asymmetric
        # groups
        gyradii = np.empty(com.shape, dtype=float)
        for i, (m, p, c) in enumerate(zip(masses, positions, com)):
            cpos = (p - c) ** 2
            gyradii[i] = np.array(
                (np.dot(m, cpos[:, (1, 2)].sum(axis=1)),
                 np.dot(m, cpos[:, (0, 2)].sum(axis=1)),
                 np.dot(m, cpos[:, (0, 1)].sum(axis=1)))
            ) / m.sum()
        return np.sqrt(gyradii)
    
    # Compute the overall radii of gyration for asymmetric groups
    return np.sqrt(
        [np.einsum("a,ad->d", m, (p - c) ** 2).sum() / m.sum() 
            for m, p, c in zip(masses, positions, com)]
    )
                