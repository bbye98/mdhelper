import os
import pathlib
import sys

import netCDF4 as nc
import numpy as np
import openmm
from openmm import app, unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper.openmm import file, pair, reporter, system as s, unit as u # noqa: E402

def test_class_netcdfreporter():

    path = os.getcwd()
    if "tests" not in path:
        path += "/tests"
    if not os.path.isdir(f"{path}/data/netcdf"):
        os.makedirs(f"{path}/data/netcdf")
    os.chdir(f"{path}/data/netcdf")

    # Set up a basic OpenMM simulation for a single LJ particle
    temp = 300 * unit.kelvin
    size = 3.4 * unit.angstrom
    mass = 39.948 * unit.amu
    scales = u.lj_scaling({
        "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule),
        "length": size, 
        "mass": mass
    })

    dims = 10 * size * np.ones(3, dtype=float)
    dims_nd = [L / unit.nanometer for L in dims]
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        (dims_nd[0], 0, 0) * unit.nanometer,
        (0, dims_nd[1], 0) * unit.nanometer,
        (0, 0, dims_nd[2]) * unit.nanometer
    )
    topology = app.Topology()
    topology.setUnitCellDimensions(dims)
    pair_lj = pair.lj_coul(2.5 * size)
    s.register_particles(system, topology, 1, mass, nbforce=pair_lj, sigma=size,
                         epsilon=21.285 * unit.kilojoule_per_mole)
    system.addForce(pair_lj)

    plat = openmm.Platform.getPlatformByName("CPU")
    dt = 0.005 * scales["time"]
    integrator = openmm.LangevinMiddleIntegrator(temp, 1e-3 / dt, dt)
    simulation = app.Simulation(topology, system, integrator, plat)
    simulation.context.setPositions(dims[None, :] / 2)

    # TEST CASE 1: Correct headers and data for restart file 
    # (static method, filename)
    state = simulation.context.getState(getPositions=True, getVelocities=True, 
                                        getForces=True)

    file.NetCDFFile.write_file("restart", state)
    ncdf = nc.Dataset("restart.nc", "r")
    assert ncdf.Conventions == "AMBERRESTART"
    assert ncdf.dimensions["frame"].size == 1
    assert np.allclose(ncdf.variables["coordinates"][:], dims / 2)

    # TEST CASE 2: Not a restart file
    f = file.NetCDFFile("restart.nc", "w")
    f._initialize(1, True, True, True)
    with pytest.raises(ValueError):
        f.write_file(state)

    # TEST CASE 3: Correct headers and data for restart file (instance method)
    f = file.NetCDFFile("restart.nc", "w", restart=True)
    f.write_file(state)
    ncdf = nc.Dataset("restart.nc", "r")
    assert ncdf.Conventions == "AMBERRESTART"
    assert ncdf.dimensions["frame"].size == 1
    assert np.allclose(ncdf.variables["coordinates"][:], dims / 2)

    # TEST CASE 4: Correct headers and data for restart file 
    # (static method, NetCDF file)
    f = nc.Dataset("restart.nc", "w")
    file.NetCDFFile.write_file(f, state)
    ncdf = nc.Dataset("restart.nc", "r")
    assert ncdf.Conventions == "AMBERRESTART"
    assert ncdf.dimensions["frame"].size == 1
    assert np.allclose(ncdf.variables["coordinates"][:], dims / 2)

    # TEST CASE 5: Correct headers and data for trajectory file
    simulation.reporters.append(
        reporter.NetCDFReporter("traj", 1, periodic=True, velocities=True, 
                                forces=True)
    )
    simulation.step(5)

    ncdf = nc.Dataset("traj.nc", "r")
    assert ncdf.program == "MDHelper"
    assert np.allclose(ncdf.variables["cell_lengths"][0], dims)
    assert np.allclose(
        ncdf.variables["coordinates"][0, 0] 
        - ncdf.variables["time"][0] * ncdf.variables["velocities"][0, 0], 
        dims / 2, 
        atol=1e-3
    )

    # TEST CASE 6: Correct headers and data for subset trajectory file
    s.register_particles(system, topology, 1, mass, nbforce=pair_lj, 
                         sigma=size, epsilon=21.285 * unit.kilojoule_per_mole)
    integrator = openmm.LangevinMiddleIntegrator(temp, 1e-3 / dt, dt)
    simulation = app.Simulation(topology, system, integrator, plat)
    simulation.context.setPositions(np.vstack((dims / 4, 3 * dims / 4)) 
                                    * unit.angstrom)
    simulation.reporters.append(
        reporter.NetCDFReporter("traj_subset", 1, periodic=True, velocities=True, 
                                forces=True, subset=[0])
    )
    simulation.step(1)

    ncdf = nc.Dataset("traj_subset.nc", "r")
    assert ncdf.dimensions["atom"].size == 1