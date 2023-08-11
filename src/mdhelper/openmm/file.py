"""
Custom OpenMM topology and trajectory file readers and writers
==============================================================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module provides custom topology and trajectory file readers and
writers for OpenMM.
"""

import platform
from typing import Any, Union

import numpy as np
import openmm
from openmm import app, unit

try:
    import netCDF4 as nc
    FOUND_NETCDF = True
except ImportError: # pragma: no cover
    from scipy.io import netcdf_file as nc
    FOUND_NETCDF = False

from .. import VERSION, ArrayLike

class NetCDFFile():

    """
    Interface for writing AMBER NetCDF trajectory and restart files.

    Parameters
    ----------
    file : `str`
        Filename of NetCDF file to write to. If `file` does not have the
        :code:`.nc` extension, it will automatically be appended.
    
    mode : `str`
        NetCDF file access mode.

    restart : `bool`, default: `False`
        Specifies whether the NetCDF file is a trajectory or restart file.

    **kwargs
        Keyword arguments to be passed to :code:`netCDF4.Dataset` or
        :code:`scipy.io.netcdf_file()`.
    """

    def __init__(
            self, file: Union[str, nc.Dataset], mode: str, 
            restart: bool = False, **kwargs):

        if isinstance(file, str):
            if not file.endswith(".nc") or not file.endswith(".ncdf"):
                file += ".nc"
            if FOUND_NETCDF:
                self._nc = nc.Dataset(file, mode=mode, 
                                      format="NETCDF3_64BIT_OFFSET", **kwargs)
            else: # pragma: no cover
                self._nc = nc(file, mode=mode, version=2, **kwargs)
        else:
            self._nc = file
        
        self._frame = 0 if mode == "w" else self._nc["time"].shape[0]
        self._restart = restart

    def _initialize(
            self, N: int, cell: bool, velocities: bool, forces: bool,
            remd: str = None, temp0: float = None,
            remd_dimtype: ArrayLike = None,
            remd_indices: ArrayLike = None,
            remd_repidx: int = -1, remd_crdidx: int = -1,
            remd_values: ArrayLike = None) -> None:

        """
        Initialize the NetCDF file according to AMBER NetCDF
        Trajectory/Restart Convention Version 1.0, Revision C
        (https://ambermd.org/netcdf/nctraj.xhtml).

        Parameters
        ----------
        N : `int`
            Number of particles.
        
        cell : `bool`
            Specifies whether simulation box length and angle
            information is available.
        
        velocities : `bool`
            Specifies whether particle velocities should be written.

        forces : `bool`
            Specifies whether forces exerted on particles should be
            written.
        
        remd : `str`, :code:`{"temp", "multi"}`, optional
            Specifies whether information about a replica exchange
            molecular dynamics (REMD) simulation is written. 
            
            .. container::

               **Valid values**:

               * :code:`"temp"` for regular REMD.
               * :code:`"multi"` for multi-dimensional REMD.
        
        temp0 : `float`, optional
            Temperature that the thermostat is set to maintain for a
            REMD restart file only. 
            
            **Reference unit**: :math:`\mathrm{K}`.

        remd_dimtype : array-like, optional
            Array specifying the exchange type(s) for the REMD
            dimension(s). Required for a multi-dimensional REMD restart
            file.

        remd_indices : array-like, optional
            Array specifying the position in all dimensions that each
            frame is in. Required for a multi-dimensional REMD restart
            file.
        
        remd_repidx : `int`, optional
            Overall index of the frame in replica space.
        
        remd_crdidx : `int`, optional
            Overall index of the frame in coordinate space.
        
        remd_values : array-like, optional
            Replica value the specified replica dimension has for that
            given frame. Required for a multi-dimensional REMD restart file.
        """

        self._nc.Conventions = "AMBER"
        if self._restart:
            self._nc.Conventions += "RESTART"
        self._nc.ConventionVersion = "1.0"
        self._nc.program = "MDHelper"
        self._nc.programVersion = VERSION
        self._nc.title = (f"OpenMM {openmm.Platform.getOpenMMVersion()} / "
                          f"{platform.node()}")

        if remd == "multi": # pragma: no cover
            self._nc.createDimension("remd_dimension", len(remd_dimtype))
        self._nc.createDimension("spatial", 3)
        self._nc.createDimension("atom", N)

        if self._restart:
            self._nc.createDimension("frame", 1)
            self._nc.createVariable("coordinates", "d", ("atom", "spatial"))
        else:
            self._nc.createDimension("frame", None)
            self._nc.createVariable("coordinates", "f", 
                                    ("frame", "atom", "spatial"))
        self._nc.variables["coordinates"].units = "angstrom"

        self._nc.createVariable("time", "d", ("frame",))
        self._nc.variables["time"].units = "picosecond"

        if cell:
            self._nc.createDimension("cell_spatial", 3)
            self._nc.createDimension("cell_angular", 3)
            self._nc.createDimension("label", 5)
            self._nc.createVariable("spatial", "c", ("spatial",))
            self._nc.variables["spatial"][:] = list("xyz")
            self._nc.createVariable("cell_spatial", "c", ("cell_spatial",))
            self._nc.variables["cell_spatial"][:] = list("abc")
            self._nc.createVariable("cell_angular", "c", 
                                    ("cell_angular", "label"))
            self._nc.variables["cell_angular"][:] = [list("alpha"), 
                                                     list("beta "), 
                                                     list("gamma")]

            if self._restart:
                self._nc.createVariable("cell_lengths", "d", ("cell_spatial",))
                self._nc.createVariable("cell_angles", "d", ("cell_angular",))
            else:
                self._nc.createVariable("cell_lengths", "f", 
                                        ("frame", "cell_spatial"))
                self._nc.createVariable("cell_angles", "f", 
                                        ("frame", "cell_angular"))
            self._nc.variables["cell_lengths"].units = "angstrom"
            self._nc.variables["cell_angles"].units = "degree"
        
        if velocities:
            if self._restart:
                self._nc.createVariable("velocities", "d", ("atom", "spatial"))
            else:
                self._nc.createVariable("velocities", "f", 
                                        ("frame", "atom", "spatial"))
            self._nc.variables["velocities"].units = "angstrom/picosecond"
            self._nc.variables["velocities"].scale_factor = 20.455

        if forces:
            if self._restart:
                self._nc.createVariable("forces", "d", ("atom", "spatial"))
            else:
                self._nc.createVariable("forces", "f", 
                                        ("frame", "atom", "spatial"))
            self._nc.variables["forces"].units = "kilocalorie/mole/angstrom"
        
        if remd is not None: # pragma: no cover
            if remd == "temp":
                self._nc.createVariable("temp0", "d", ("frame",))
                if self._restart:
                    if temp0 is None:
                        emsg = ("Temperature must be provided for a REMD "
                                "restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["temp0"][0] = temp0
                self._nc.variables["temp0"].units = "kelvin"

            elif remd == "multi":
                self._nc.createVariable("remd_dimtype", "i", 
                                        ("remd_dimension",))
                self._nc.createVariable("remd_repidx", "i", ("frame",))
                self._nc.createVariable("remd_crdidx", "i", ("frame",))
                if self._restart:
                    if remd_dimtype is None:
                        emsg = ("Dimension types must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_dimtype"] = remd_dimtype

                    self._nc.createVariable("remd_indices", "i", 
                                            ("remd_dimension",))
                    if remd_indices is None:
                        emsg = ("Dimension indices must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_indices"] = remd_indices

                    self._nc.variables["remd_repidx"][0] = remd_repidx
                    self._nc.variables["remd_crdidx"][0] = remd_crdidx

                    self._nc.createVariable("remd_values", "d", 
                                            ("remd_dimension",))
                    if remd_values is None:
                        emsg = ("Replica values must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_values"][:] = remd_values

                else:
                    self._nc.createVariable("remd_indices", "i", 
                                            ("frame", "remd_dimension"))
                    self._nc.createVariable("remd_values", "d", 
                                            ("frame", "remd_dimension"))

    def write_file(self: Any, state: openmm.State) -> None:
        
        """
        Write the simulation state to a restart NetCDF file.

        Parameters
        ----------
        self : `str`, `netcdf4.Dataset`, `scipy.io.netcdf_file`, \
        or `mdhelper.openmm.file.NetCDFFile`
            If :meth:`write_file` is called as a static method, you must 
            provide a filename or a NetCDF file object. Otherwise, the
            NetCDF file embedded in the current instance is used.

        state : `openmm.State`
            OpenMM simulation state from which to retrieve cell 
            dimensions and particle positions, velocities, and forces.
        """

        # Collect all available data in the state
        data = {}
        pbv = state.getPeriodicBoxVectors()
        if pbv is not None:
            (a, b, c, alpha, beta, gamma) = \
                app.internal.unitcell.computeLengthsAndAngles(pbv)
            data["cell_lengths"] = 10 * np.array((a, b, c))
            data["cell_angles"] = 180 * np.array((alpha, beta, gamma)) / np.pi
        data["coordinates"] = state.getPositions(asNumpy=True).value_in_unit(
            unit.angstrom
        )
        try:
            data["velocities"] \
                = state.getVelocities(asNumpy=True).value_in_unit(
                    unit.angstrom / unit.picosecond
                )
        except openmm.OpenMMException: # pragma: no cover
            pass
        try:
            data["forces"] \
                = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilocalorie_per_mole / unit.angstrom
                )
        except openmm.OpenMMException: # pragma: no cover
            pass

        # Create NetCDF file if it doesn't already exist
        if not isinstance(self, NetCDFFile):
            self = NetCDFFile(self, "w", restart=True)
        if not hasattr(self._nc, "Conventions"):
            self._initialize(data["coordinates"].shape[0], 
                             "cell_lengths" in data or "cell_angles" in data,
                             "velocities" in data, "forces" in data)
        elif self._nc.Conventions != "AMBERRESTART":
            raise ValueError("The NetCDF file must be a restart file.")
            
        # Write data to NetCDF file
        for k, v in data.items():
            self._nc.variables[k][:] = v
        self._nc.sync()

    def write_model(
            self, time: Union[float, np.ndarray], coordinates: np.ndarray,
            velocities: np.ndarray = None, forces: np.ndarray = None,
            cell_lengths: np.ndarray = None, cell_angles: np.ndarray = None, *,
            restart: bool = False) -> None:

        """
        Write the simulation state(s) to the NetCDF file.

        Parameters
        ----------
        time : `float` or `numpy.ndarray`
            Time(s). The dimensionality determines whether a single or
            multiple frames are written. 
            
            **Reference unit**: :math:`\mathrm{ps}`.
        
        coordinates : `numpy.ndarray`
            Particle coordinates of :math:`N` particles over :math:`N_t`
            frames. The dimensionality depends on whether a single or 
            multiple frames are to be written and must be compatible 
            with that for `time`.

            **Shape**: :math:`(N,\,3)` or :math:`(N_t,\,N,\,3)`.

            **Reference unit**: :math:`\mathrm{Å}`.

        velocities : `numpy.ndarray`, optional
            Particle velocities of :math:`N` particles over :math:`N_t` 
            frames. The dimensionality depends on whether a single or 
            multiple frames are to be written and must be compatible 
            with that for `time`.

            **Shape**: :math:`(N,\,3)` or :math:`(N_t,\,N,\,3)`.

            **Reference unit**: :math:`\mathrm{Å/ps}`.

        forces : `numpy.ndarray`, optional
            Forces exerted on :math:`N` particles over :math:`N_t` 
            frames. The dimensionality depends on whether a single or 
            multiple frames are to be written and must be compatible 
            with that for `time`.

            **Shape**: :math:`(N,\,3)` or :math:`(N_t,\,N,\,3)`.

            **Reference unit**: :math:`\mathrm{Å/ps}`.

        cell_lengths : `numpy.ndarray`, optional
            Simulation box dimensions.

            **Shape**: :math:`(3,)`.

            **Reference unit**: :math:`\mathrm{Å}`.

        cell_angles : `numpy.ndarray`, optional
            Angles that define the shape of the simulation box.

            **Shape**: :math:`(3,)`.

            **Reference unit**: :math:`^\circ`.

        restart : `bool`, keyword-only, default: `False`
            Prevents the frame index from being incremented if writing a
            NetCDF restart file.
        """

        n_frames = len(time) if isinstance(time, (tuple, list, np.ndarray)) \
                   else 1
        frames = slice(self._frame, self._frame + n_frames)

        self._nc.variables["time"][frames] = time
        self._nc.variables["coordinates"][frames] = coordinates
        if velocities is not None:
            self._nc.variables["velocities"][frames] = velocities
        if forces is not None:
            self._nc.variables["forces"][frames] = forces
        if cell_lengths is not None:
            self._nc.variables["cell_lengths"][frames] = cell_lengths
        if cell_angles is not None:
            self._nc.variables["cell_angles"][frames] = cell_angles
        self._nc.sync()
        if not restart:
            self._frame += n_frames