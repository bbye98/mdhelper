"""
Topology transformations
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of common topology
transformations, like the generation of initial particle positions.
"""

from typing import Any, Union

import numpy as np

from .. import FOUND_OPENMM
from . import utility

if FOUND_OPENMM:
    from openmm import app, unit

def create_atoms(
        dims: Union[np.ndarray[float], "unit.Quantity", "app.Topology"],
        N: int = None, N_p: int = 1, *, lattice: str = None,
        length: Union[float, "unit.Quantity"] = 0.34,
        flexible: bool = False, connectivity: bool = False,
        randomize: bool = False, length_unit: "unit.Unit" = None,
        wrap: bool = False) -> Any:

    r"""
    Generates initial particle positions for coarse-grained simulations.

    Parameters
    ----------
    dims : `numpy.ndarray`, `openmm.unit.Quantity`, or `openmm.app.Topology`
        System dimensions, provided as an array or obtained from an
        OpenMM topology.

        **Reference unit**: :math:`\mathrm{nm}`.

    N : `int`, optional
        Total number of particles (monomers). Must be provided for
        random melts or polymers.

    N_p : `int`, default: :code:`1`
        Number of particles (monomers) :math:`N_p` in each
        segment (polymer chain).

        **Valid values**: :math:`1\leq N_\mathrm{p}\leq N`, with
        :math:`N` divisible by :math:`N_\mathrm{p}`.

    lattice : `str`, keyword-only, optional
        Lattice type, with the relevant length scale specified in
        `length`. If `lattice` is not specified, particle positions will
        be assigned randomly.

        .. tip::

           To build walls with the correct periodicity, set the
           :math:`z`-dimension to :code:`0` in `dims` and
           :code:`flexible=True`. This function will then return
           the wall particle positions and the :math:`x`- and
           :math:`y`-dimensions closest to those specified in `dims`
           that also satisfy the lattice periodicity.

           Keep in mind that walls should only be built in the
           :math:`z`-direction.

        .. container::

           **Valid values**:

           * :code:`"fcc"`: Face-centered cubic (FCC) lattice,
             determined by the particle size :math:`a`.
           * :code:`"hcp"`: Hexagonal close-packed (HCP) lattice,
             determined by the particle size :math:`a`.
           * :code:`"cubic"`: Cubic crystal system, determined by the
             particle size :math:`a`.
           * :code:`"honeycomb"`: Honeycomb lattice (e.g., graphene),
             determined by the bond length :math:`b`.

    length : `float` or `openmm.unit.Quantity`, default: :code:`0.34`
        For random polymer positions, `length` is the bond length used in
        the random walk. For lattice systems, `length` is either the
        particle size or the bond length, depending on the lattice type.
        Has no effect if :code:`N_p=1` or :code:`lattice=None`.

        **Reference unit**: :math:`\mathrm{nm}`.

    flexible : `bool`, default: :code:`False`
        Specifies whether the dimensions provided in `dims` can be
        changed to satisfy the lattice periodicity, if applicable.
        For example, if the provided :math:`z`-dimension can hold a
        non-integer 19.7 replicas of a lattice, then it is updated
        to reflect the width of 20 replicas. To ignore a direction (and
        make that dimension constant), such as when creating walls, set
        that dimension to :code:`0` in `dims`.

    connectivity : `bool`, default: :code:`False`
        Determines whether bond information is returned for polymeric
        systems. Has no effect if :code:`N_p=1`.

    randomize : `bool`, default: :code:`False`
        Determines whether the order of the replicated polymer positions
        are randomized. Has no effect if :code:`N_p=1`.

    length_unit : `openmm.unit.Unit`, optional
        Length unit. If not specified, it is determined automatically
        from `dims` or `length`.

    wrap : `bool`, default: :code:`False`
        Determines whether particles outside the simulation box are
        wrapped back into the main unit cell.

    Returns
    -------
    positions : `numpy.ndarray` or `openmm.unit.Quantity`
        Generated particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    bonds : `numpy.ndarray`
        Pairs of all bonded particle indices. Only returned if
        :code:`connectivity=True`.

    dims : `numpy.ndarray` or `openmm.unit.Quantity`
        Dimensions for lattice systems. Only returned if `lattice` is
        not :code:`None`.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.
    """

    # Get raw numerical dimensions and length
    if isinstance(dims, app.Topology):
        dims = dims.getUnitCellDimensions()
    dims, length_unit = utility.strip_unit(dims, length_unit)
    length, length_unit = utility.strip_unit(length, length_unit)

    if lattice is None:

        # Ensure the user-specified values are valid
        if N is None:
            raise ValueError("The number of particles N must be specified.")
        if not isinstance(N, (int, np.integer)):
            raise ValueError("The number of particles N must be an integer.")
        if not (1 <= N_p <= N and isinstance(N_p, (int, np.integer))):
            emsg = ("The number of particles N_p in each segment must "
                    "be an integer between 1 and N.")
            raise ValueError(emsg)
        if N_p > 1 and N % N_p != 0:
            emsg = (f"{N=} particles cannot be evenly divided into segments "
                    f"with {N_p=} particles.")
            raise ValueError(emsg)

        # Generate particle positions for a random melt
        if N_p == 1:
            pos = np.random.rand(N, 3) * dims
        else:

            # Determine unit cell information for each segment
            segments = N // N_p
            n_cells = utility.get_closest_factors(segments, 3)
            cell_dims = dims / n_cells

            # Randomly generate a segment within the unit cell
            cell_pos = np.zeros((N_p, 3))
            cell_pos[0] = cell_dims / 4
            rng = np.random.default_rng()
            for i in range(1, N_p):
                vec = rng.random(3) * 2 - 1
                cell_pos[i] = cell_pos[i - 1] + length * vec / np.linalg.norm(vec)

            # Replicate unit cell in x-, y-, and z-directions (and
            # randomize, if desired)
            pos = utility.replicate(cell_dims, cell_pos, n_cells)
            if randomize:
                pos = np.vstack(rng.permutation(pos.reshape((segments, -1, 3))))

            # Wrap particles past the system boundaries
            if wrap:
                for i in range(3):
                    pos[pos[:, i] < 0, i] += dims[i]
                    pos[pos[:, i] > dims[i], i] -= dims[i]

            # Determine all bonds
            if connectivity:
                bonds = np.array([(i * N_p + j, i * N_p + j + 1)
                                  for i in range(segments)
                                  for j in range(N_p - 1)])
                return pos if length_unit is None else pos * length_unit, bonds

        return pos if length_unit is None else pos * length_unit
    else:
        around = np.around if flexible else np.floor

        # Set unit cell information
        if lattice == "cubic":
            _dims = np.array(dims)
            _dims[dims == 0] = 1
            n_cells = around(_dims / length).astype(int)
            cell_dims = length * np.array((1, 1, 1))
            x, y, z = (length * np.arange(n) for n in n_cells)
            pos = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        else:
            if lattice == "fcc":
                cell_dims = length * np.array((1, np.sqrt(3), 3 * np.sqrt(6) / 3))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0.5, np.sqrt(3) / 2, 0),
                    (0.5, np.sqrt(3) / 6, np.sqrt(6) / 3),
                    (0, 2 * np.sqrt(3) / 3, np.sqrt(6) / 3),
                    (0, np.sqrt(3) / 3, 2 * np.sqrt(6) / 3),
                    (0.5, 5 * np.sqrt(3) / 6, 2 * np.sqrt(6) / 3),
                ))
            elif lattice == "hcp":
                cell_dims = length * np.array((1, np.sqrt(3), 2 * np.sqrt(6) / 3))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0.5, np.sqrt(3) / 2, 0),
                    (0.5, np.sqrt(3) / 6, np.sqrt(6) / 3),
                    (0, 2 * np.sqrt(3) / 3, np.sqrt(6) / 3)
                ))
            elif lattice == "honeycomb":
                cell_dims = length * np.array((np.sqrt(3), 3, np.inf))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0, 1, 0),
                    (np.sqrt(3) / 2, 1.5, 0),
                    (np.sqrt(3) / 2, 2.5, 0)
                ))

            # Determine unit cell multiples
            n_cells = around(dims / cell_dims).astype(int)
            n_cells[n_cells == 0] = 1
            cell_dims[np.isinf(cell_dims)] = 0

            # Replicate unit cell in x-, y-, and z-directions
            pos = utility.replicate(cell_dims, cell_pos, n_cells)

        # Remove particles outside of system boundaries
        if flexible:
            n_cells[dims == 0] = 0
            pos = pos[~np.any(pos[:, dims == 0] > 0, axis=1)]
        else:
            pos = pos[~np.any(pos > dims, axis=1)]

    if length_unit is None:
        return pos, n_cells * cell_dims
    return pos * length_unit, n_cells * cell_dims * length_unit

def unwrap(
        positions: np.ndarray[float], positions_old: np.ndarray[float],
        dimensions: np.ndarray[float], *, thresholds: float = None,
        images: np.ndarray[int] = None, in_place: bool = True
    ) -> Union[None, tuple[np.ndarray[float], np.ndarray[int]]]:

    """
    Unwraps particle positions.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    positions_old : `numpy.ndarray`
        Previous particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        System dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    thresholds : `float`, keyword-only, optional
        Maximum distances in each direction a particle can move before
        it is considered to have crossed a boundary.

        **Reference unit**: :math:`\mathrm{nm}`.

    images : `numpy.ndarray`, keyword-only, optional
        Current image flags (or number of boundary crossings).

        **Shape**: :math:`(N,\,3)`.

    in_place : `bool`, keyword-only, default: :code:`False`
        Determines whether the input array is modified in-place.

    Returns
    -------
    positions : `numpy.ndarray`
        Unwrapped particle positions. Only returned if
        :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    images : `numpy.ndarray`
        Updated image flags (or number of boundary crossings). Only
        returned if :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.
    """

    if thresholds is None:
        thresholds = np.min(dimensions) / 2
    if images is None:
        images = np.zeros_like(positions, dtype=int)

    dpos = positions - positions_old
    mask = np.abs(dpos) >= thresholds
    if in_place:
        images[mask] -= np.sign(dpos[mask]).astype(int)
        positions_old[:] = positions[:]
        positions += images * dimensions
    else:
        positions = positions.copy()
        positions_old = positions.copy()
        images = images.copy()
        images[mask] -= np.sign(dpos[mask]).astype(int)
        positions += images * dimensions
        return positions, positions_old, images