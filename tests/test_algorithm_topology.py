import pathlib
import sys

import numpy as np
from openmm import app, unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper.algorithm import topology # noqa: E402

rng = np.random.default_rng()
dims = np.array((10, 10, 10))

def test_func_create_atoms_error():

    # TEST CASE 1: N not specified
    with pytest.raises(ValueError):
        topology.create_atoms(dims)

    # TEST CASE 2: N not an integer
    with pytest.raises(ValueError):
        topology.create_atoms(dims, np.pi)

    # TEST CASE 3: Invalid N_p
    with pytest.raises(ValueError):
        topology.create_atoms(dims, N=9000, N_p=9001)

    # TEST CASE 4: N not divisible by N_p
    with pytest.raises(ValueError):
        topology.create_atoms(dims, N=10, N_p=3)

def test_func_create_atoms_random():

    # TEST CASE 1: Random melt in reduced units
    N = rng.integers(1, 1000)
    pos = topology.create_atoms(dims, N)
    assert pos.shape == (N, 3)

    # TEST CASE 2: Random melt with default length unit
    pos = topology.create_atoms(dims * unit.nanometer, N)
    assert pos.shape == (N, 3) and pos.unit == unit.nanometer

    # TEST CASE 3: Random melt with specific length unit
    pos = topology.create_atoms(dims * unit.nanometer, N, length_unit=unit.angstrom)
    assert pos.shape == (N, 3) and pos.unit == unit.angstrom

    # TEST CASE 4: Topology provided instead of dimensions
    topo = app.Topology()
    topo.setUnitCellDimensions(dims)
    pos = topology.create_atoms(topo, N)
    assert pos.shape == (N, 3)

def test_func_create_atoms_polymer():

    # TEST CASE 1: Random polymer melt
    M = rng.integers(1, 100)
    N_p = rng.integers(2, 100)
    N = M * N_p
    pos = topology.create_atoms(dims, N, N_p)
    assert pos.shape == (N, 3)

    # TEST CASE 2: Random polymer melt with bond information and wrapped
    # positions
    pos, bonds = topology.create_atoms(dims, N, N_p, connectivity=True,
                                       randomize=True, wrap=True)
    assert pos.shape == (N, 3)
    assert bonds.shape[0] == N - M
    assert np.all((pos[:, 0] > 0) & (pos[:, 0] < dims[0]))
    assert np.all((pos[:, 1] > 0) & (pos[:, 2] < dims[1]))
    assert np.all((pos[:, 1] > 0) & (pos[:, 2] < dims[2]))

def test_func_create_atoms_lattice():

    # TEST CASE 1: FCC lattice with flexible dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="fcc", length=0.8,
                                          flexible=True)
    assert np.allclose(pos[4], 0.8 * np.array((0, np.sqrt(3) / 3, 2 * np.sqrt(6) / 3)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 2: HCP lattice with flexible dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="hcp", length=0.8,
                                          flexible=True)
    assert np.allclose(pos[1], 0.8 * np.array((0.5, np.sqrt(3) / 2, 0)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 3: HCP lattice to fill specified dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="hcp", length=0.8)
    assert np.allclose(pos[1], 0.8 * np.array((0.5, np.sqrt(3) / 2, 0)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 4: Graphene wall
    pos, new_dims = topology.create_atoms(dims, lattice="honeycomb",
                                          length=0.142 * unit.nanometer,
                                          flexible=True)
    assert pos[1, 1] == 0.142 * unit.nanometer
    assert np.allclose(dims[:2], new_dims[:2], atol=1)
    assert new_dims[2] == 0 * unit.nanometer