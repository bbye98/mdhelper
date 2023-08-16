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

def test_classes_netcdffile_netcdfreporter():

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
    pair_lj = pair.lj_coul(dims[0] / 4)
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
    ncdf = file.NetCDFFile("restart", "r")
    assert ncdf._nc.Conventions == "AMBERRESTART"
    assert ncdf.get_num_frames() == 1
    assert np.allclose(ncdf.get_positions(), dims / 2)

    # TEST CASE 2: Not a restart file
    ncdf = file.NetCDFFile.write_header("restart.nc", 1, True, True, True)
    with pytest.raises(ValueError):
        ncdf.write_file(state)

    # TEST CASE 3: Correct headers and data for restart file (instance method)
    ncdf = file.NetCDFFile("restart.nc", "ws", restart=True)
    ncdf.write_file(state)
    ncdf = file.NetCDFFile("restart.nc", "r")
    assert ncdf._nc.Conventions == "AMBERRESTART"
    assert ncdf.get_num_frames() == 1
    assert np.allclose(ncdf.get_positions(), dims / 2)

    # TEST CASE 4: Correct headers and data for restart file 
    # (static method, NetCDF file)
    file.NetCDFFile.write_file(nc.Dataset("restart.nc", "ws"), state)
    ncdf = file.NetCDFFile("restart.nc", "r")
    assert ncdf._nc.Conventions == "AMBERRESTART"
    assert ncdf.get_num_frames() == 1
    assert np.allclose(ncdf.get_positions(), dims / 2)

    # TEST CASE 5: Correct headers and data for trajectory file
    simulation.reporters.append(
        reporter.NetCDFReporter("traj.nc", 1, periodic=True, velocities=True, 
                                forces=True)
    )
    simulation.step(5)

    ncdf = file.NetCDFFile("traj.nc", "r")
    cell_lengths, cell_angles = ncdf.get_dimensions(0)
    assert ncdf._nc.program == "MDHelper"
    assert np.allclose(cell_lengths, dims)
    assert np.allclose(cell_angles, 90 * np.ones(3, dtype=float))
    assert np.allclose(
        ncdf.get_positions(0) - ncdf.get_times(0) * ncdf.get_velocities(0), 
        dims / 2, 
        atol=1e-3
    )
    assert ncdf.get_num_frames() == 5
    assert ncdf.get_velocities().shape == (5, 1, 3)
    assert ncdf.get_forces().shape == (5, 1, 3)

    # TEST CASE 6: Correct number of atoms and lack of velocities and 
    # forces for subset trajectory file
    s.register_particles(system, topology, 1, mass, nbforce=pair_lj, 
                         sigma=size, epsilon=21.285 * unit.kilojoule_per_mole)
    integrator = openmm.LangevinMiddleIntegrator(temp, 1e-3 / dt, dt)
    simulation = app.Simulation(topology, system, integrator, plat)
    simulation.context.setPositions(np.vstack((dims / 4, 3 * dims / 4)) 
                                    * unit.angstrom)
    simulation.reporters.append(
        reporter.NetCDFReporter("traj_subset.nc", 1, periodic=True, subset=[0])
    )
    simulation.step(1)

    ncdf = file.NetCDFFile("traj_subset.nc", "r")
    assert ncdf.get_num_atoms() == 1
    with pytest.warns(UserWarning):
        assert ncdf.get_velocities() is None
    with pytest.warns(UserWarning):
        assert ncdf.get_forces() is None

    # TEST CASE 7: Correct number of atoms for full trajectory file
    state = simulation.context.getState(getPositions=True)
    file.NetCDFFile.write_model(
        "traj_two.nc", 
        state.getTime().value_in_unit(unit.picosecond), 
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    ncdf = file.NetCDFFile("traj_two.nc", "r")
    assert ncdf.get_num_atoms() == 2