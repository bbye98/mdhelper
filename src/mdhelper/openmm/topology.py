"""
OpenMM topology transformations
===============================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains implementations of common OpenMM topology 
transformations, like the generation of initial particle positions and
subsetting existing topology objects.
"""

from typing import Any, Iterable, Union

import numpy as np
from openmm import app

from ..algorithm import topology as t

def create_atoms(*args, **kwargs) -> Any:

    """
    Generates initial particle positions.

    This is an alias function. For more information, see
    :func:`mdhelper.algorithm.topology.create_atoms`.
    """

    return t.create_atoms(*args, **kwargs)

def _get_hierarchy_indices(
        item: Union[app.Atom, app.topology.Bond, app.Residue, app.Chain]
    ) -> tuple[set, set, set, set]:

    """
    Get unique indices of all topology items related to the one passed
    to this function.

    For an atom, the indices of itself, the residue it belongs to, and
    the chain that its residue belongs to are returned.

    For a bond, the indices of the two atoms it connects, itself, the
    residue(s) that the two atoms belong to, and the chain(s) that the
    residue(s) belong to are returned.

    For a residue, the indices of all underlying atoms and bonds,
    itself, and the chain it belongs to are returned.

    For a chain, the indices of all underlying atoms, bonds, and
    residues, and itself are returned.

    Parameters
    ----------
    item : `openmm.app.Atom`, `openmm.app.topology.Bond`,
    `openmm.app.Residue`, or `openmm.app.Chain`
        Topology item of interest.

    Returns
    -------
    atoms : `set`
        Indices of all atoms associated with `item`.

    bonds : `set`
        Indices of all bonds associated with `item`.

    residues : `set`
        Indices of all residues associated with `item`.

    chains : `set`
        Indices of all chains associated with `item`.
    """

    if isinstance(item, app.Atom):
        return {item.index}, set(), {item.residue.index}, {item.residue.chain.index}

    elif isinstance(item, app.topology.Bond):
        return {item.atom1.index, item.atom2.index}, {b.index}, \
               {item.atom1.residue.index, item.atom2.residue.index}, \
               {item.atom1.residue.chain.index, item.atom2.residue.chain.index}

    elif isinstance(item, app.Residue):
        return {a.index for a in item.atoms()}, \
               {b.index for b in item.bonds()}, {item.index}, {item.chain.index}

    elif isinstance(item, app.Chain):
        atoms = set()
        bonds = set()
        residues = set()
        for residue in item.residues():
            a, b, r, _ = _get_hierarchy_indices(residue)
            atoms |= a
            bonds |= b
            residues |= r
        return atoms, bonds, residues, {item}

def _is_topology_object(obj: Any):

    """
    Check if the argument is a topology item.

    Parameters
    ----------
    obj : `Any`
        Any object.
    
    Returns
    -------
    is_topology_object : `bool`
        Boolean value indicating whether `obj` is a topology item.
    """
    return isinstance(obj, (app.Atom, app.topology.Bond, app.Residue, app.Chain))

def subset(
        topology: app.Topology, positions: np.ndarray = None, *,
        delete: Iterable[Union[int, app.Atom, app.topology.Bond, app.Residue,
                               app.Chain]] = None,
        keep: Iterable[Union[int, app.Atom, app.topology.Bond, app.Residue,
                             app.Chain]] = None,
        types: Union[str, Iterable[str]] = None
    ) -> Union[app.Topology, np.ndarray]:

    """
    Creates a topology subset and get its corresponding particle positions.

    Parameters
    ----------
    topology : `openmm.app.Topology`
        OpenMM topology.

    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the topology.

        **Shape**: :math:`(N,\,3)`.

    delete : array-like, keyword-only, optional
        `openmm.app.Atom`, `openmm.app.Bond`, `openmm.app.Residue`, 
        and/or `openmm.app.Chain` objects, or the indices of those 
        objects. If indices are provided, their corresponding object 
        types (:code:`"atom"`, :code:`"residue"`, :code:`"chain"`) must
        be provided in `types`. The specified items will be deleted from
        the model. 
        
        .. note::
        
           Only one of `delete` and `keep` can be specified.
    
    keep : array-like, keyword-only, optional
        `openmm.app.Atom`, `openmm.app.Bond`, `openmm.app.Residue`, 
        and/or `openmm.app.Chain` objects, or the indices of those 
        objects. If indices are provided, their corresponding object
        types (:code:`"atom"`, :code:`"residue"`, :code:`"chain"`) must
        be provided in `types`. The specified items will be kept in the
        model.

        .. note::
        
           Only one of `delete` and `keep` can be specified.

    types : `str` or array-like, keyword-only, optional
        Object types corresponding to the indices provided in `delete` 
        or `keep`. If a `str` is provided, all items in the array are
        assumed to have the same object type.

    Returns
    -------
    topology : `openmm.app.Topology`
        OpenMM topology subset.

    positions : `numpy.ndarray`
        Positions of the remaining :math:`N_\mathrm{rem}` particles in
        the topology.

        **Shape**: :math:`(N_\mathrm{rem},\,3)`.
    """

    # Set boolean flags for subroutines below
    fd, fk = delete is not None, keep is not None

    # Check if both delete and keep arguments were provided
    if fd and fk:
        emsg = ("Only specify topology items to either delete or keep. "
                "When both types are specified, the atoms, bonds, "
                "residues, and/or chains to be removed from the topology "
                "become ambiguous.")
        raise ValueError(emsg)

    # Ensure object type(s) are provided
    if (fd or fk) and types is None:
        emsg = ("Object types must be provided for the specified "
                f"topology items to be {'kept' if fk else 'deleted'}.")
        raise ValueError(emsg)

    # Create dictionary with topology subitems
    model = {
        "atom": np.fromiter(topology.atoms(), dtype=object),
        "bond": np.fromiter(topology.bonds(), dtype=object),
        "chain": np.fromiter(topology.chains(), dtype=object),
        "residue": np.fromiter(topology.residues(), dtype=object)
    }

    # Create OpenMM modeller
    Modeller = app.Modeller(topology, positions)

    # If indices and types of objects to be deleted are specified,
    # create an iterable object of corresponding items from the
    # dictionary
    if fd and types is not None:
        delete = (i if _is_topology_object(i)
                    else model[types][i]
                    for i in delete) if isinstance(types, str) \
                 else (i if _is_topology_object(i) else model[t][i]
                         for i, t in zip(delete, types))

    elif fk:
        
        # Preallocate sets to store indices of atoms, residues, and
        # chains to delete
        atoms = set()
        bonds = set()
        residues = set()
        chains = set()

        # Remove items to be kept from the master list of topology
        # subitems to delete
        if isinstance(types, str):
            for item in keep:
                a, b, r, c = _get_hierarchy_indices(
                    item if _is_topology_object(item)
                         else model[types][item]
                )
                atoms |= a
                bonds |= b
                residues |= r
                chains |= c   
        else:
            for item, item_type in zip(keep, types):
                a, b, r, c = _get_hierarchy_indices(
                    item if _is_topology_object(item)
                         else model[item_type][item]
                )
                atoms |= a
                bonds |= b
                residues |= r
                chains |= c

        delete = np.hstack((
            np.delete(model["atom"], list(atoms)),
            np.delete(model["bond"], list(bonds)),
            np.delete(model["residue"], list(residues)),
            np.delete(model["chain"], list(chains))
        ))

    # Create subset by deleting objects from original topology
    if delete is not None:
        Modeller.delete(delete)

    return Modeller.topology, Modeller.positions