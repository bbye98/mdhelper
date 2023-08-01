"""
Custom OpenMM reporters
=======================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module provides custom optimized OpenMM reporters.
"""

from typing import Union
import warnings

import numpy as np
import openmm
from openmm import app, unit

from .file import FOUND_NETCDF, NetCDFFile

class NetCDFReporter():

    """
    A NetCDF trajectory reporter for OpenMM that can report velocities
    and forces in addition to time and coordinates for all particles in
    the simulation or just a subset.

    Parameters
    ----------
    file : `str`
        Filename of NetCDF file to which the data is saved. If `file`
        does not have the :code:`.nc` extension, it will automatically
        be appended.

    interval : `int`
        Interval (in timesteps) at which to write frames.
    
    append : `bool`, keyword-only, default: :code:`False`
        If :code:`True`, the existing NetCDF file is opened for data to
        be appended to. If :code:`False`, a new NetCDF file is opened
        (and will clobber an existing file with the same name).

    periodic : `bool`, keyword-only, optional
        Specifies whether particle positions should be translated so the
        center of every molecule lies in the same periodic box. If
        :code:`None` (the default), it will automatically decide whether
        to translate molecules based on whether the system being
        simulated uses periodic boundary conditions.

    velocities : `bool`, keyword-only, default: :code:`False`
        Specifies whether particle velocities should be written to file.

    forces : `bool`, keyword-only, default: :code:`False`
        Specifies whether forces exerted on particles should be written
        to file.

    subset : `slice`, `numpy.ndarray`, or `openmm.app.Topology`, \
    keyword-only, optional
        Slice or array containing the indices of particles to report
        data for. If an OpenMM topology is provided instead, the indices
        are determined from the atoms found in the topology.
    """

    def __init__(
            self, file: str, interval: int, append: bool = False,
            periodic: bool = None, *, velocities: bool = False,
            forces: bool = False,
            subset: Union[slice, np.ndarray, app.Topology] = None) -> None:

        if not ".nc".endswith(".nc"):
            file += ".nc"
        self._out = NetCDFFile(file, "a" if append else "w")
        if not FOUND_NETCDF:
            wmsg = ("The netCDF4 package was not found, so the NetCDF "
                    "reporter is falling back on scipy.io.netcdf_file. "
                    "netCDF4 writes NetCDF files significantly faster than "
                    "scipy.io.netcdf_file and can make a huge difference when "
                    "used as an OpenMM trajectory reporter. Consider "
                    "installing netCDF4 via pip or Conda if possible.")
            warnings.warn(wmsg)
        self._interval = interval
        self._periodic = periodic
        self._subset = np.fromiter((a.index for a in subset.atoms()), dtype=int) \
                       if isinstance(subset, app.Topology) else subset
        self._velocities = velocities
        self._forces = forces

    def __del__(self) -> None:
        self._out._nc.close()

    def describeNextReport(self, simulation: app.Simulation) \
        -> tuple[int, bool, bool, bool, bool, bool]:

        """
        Get information about the next report this NetCDF reporter will
        generate.

        Parameters
        ----------
        simulation : `openmm.app.Simulation`
            OpenMM simulation to generate a report for.

        Returns
        -------
        report : `tuple`
            .. container::

               A six-element tuple containing

               1. the number of steps until the next report,
               2. whether the report will require coordinates,
               3. whether the report will require velocities,
               4. whether the report will require forces,
               5. whether the report will require energies, and
               6. whether coordinates should be wrapped to lie in a 
                  single periodic box.
        """

        return (self._interval - simulation.currentStep % self._interval,
                True, self._velocities, self._forces, False, self._periodic)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:

        """
        Generate a report.

        Parameters
        ----------
        simulation : `openmm.app.Simulation`
            OpenMM simulation to generate a report for.

        state : `openmm.State`
            Current state of the simulation.
        """

        # Get all requested state data from OpenMM State
        data = {}
        if self._subset is None:
            data["coordinates"] = state.getPositions(asNumpy=True) \
                                  / unit.angstrom
            if self._velocities:
                data["velocities"] = state.getVelocities(asNumpy=True) \
                                     / (unit.angstrom / unit.picosecond)
            if self._forces:
                data["forces"] = state.getForces(asNumpy=True) \
                                 / (unit.kilocalorie_per_mole / unit.angstrom)
        else:
            data["coordinates"] = state.getPositions(asNumpy=True)[self._subset] \
                                  / unit.angstrom
            if self._velocities:
                data["velocities"] = state.getVelocities(asNumpy=True)[self._subset] \
                                     / (unit.angstrom / unit.picosecond)
            if self._forces:
                data["forces"] = state.getForces(asNumpy=True)[self._subset] \
                                 / (unit.kilocalorie_per_mole / unit.angstrom)

        # Initialize NetCDF file headers, if not done already
        if not hasattr(self._out._nc, "Conventions"):
            self._out._initialize(
                simulation.topology.getNumAtoms() if self._subset is None \
                    else len(self._subset),
                simulation.topology.getPeriodicBoxVectors() is not None,
                self._velocities, self._forces
            )

        # Get the lengths and angles that define the size and shape of the
        # simulation box
        pbv = state.getPeriodicBoxVectors()
        if pbv is not None:
            (a, b, c, alpha, beta, gamma) = app.internal.unitcell.computeLengthsAndAngles(pbv)
            data["cell_lengths"] = 10 * np.array((a, b, c))
            data["cell_angles"] = 180 * np.array((alpha, beta, gamma)) / np.pi

        # Write current frame
        self._out.write_model(state.getTime() / unit.picosecond, **data)