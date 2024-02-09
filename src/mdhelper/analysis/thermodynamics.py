"""
Thermodynamics
==============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to evaluate thermodynamic properties of
systems, such as the constant-volume heat capacity.
"""

from io import StringIO
from pathlib import Path
from typing import Union
import warnings

import numpy as np
import pandas as pd

from .. import ureg, Q_

class ConstantVolumeHeatCapacity:

    r"""
    A serial implementation to caluclate the constant-volume heat
    capacity :math:`C_V` for a canonical (:math:`NVT`) system.

    The constant-volume heat capacity is defined as

    .. math::

       C_V=\frac{\langle U^2\rangle-\langle U\rangle^2}
       {k_\mathrm{B}T^2}

    where :math:`U` is the total potential energy of the system,
    :math:`\langle\cdot\rangle` denotes the ensemble average,
    :math:`k_\mathrm{B}` is the Boltzmann constant, and :math:`T` is the
    system temperature.

    Parameters
    ----------
    log_file : `str` or `pathlib.Path`, optional
        Log file generated by the simulation. If not provided, the
        potential energies must be provided directly in `energy`.

    log_format : `str`, optional
        Format of the log file. If not provided, the format will be
        determined automatically (if possible).

        **Valid values**: `"lammps"`, `"openmm"`.

    energy : `numpy.ndarray` or `pint.Quantity`, optional
        Potential energies. If not provided, the log file must be
        provided in `log_file`.

    temp : `float` or `pint.Quantity`, optional
        System temperature. If not provided, the averaged temperature
        from the log file (if available) is used.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Attributes
    ----------
    results : `dict`
        Analysis results:

        .. container::

           * :code:`"energy"`: Potential energy values.
           * :code:`"heat_capacity"`: Constant-volume heat capacity.
           * :code:`"units"`: Reference units. For example, to get the
             reference units for the heat capacity, use key
             :code:`results.heat_capacity`.
    """

    _COLUMNS = {
        "lammps": {
            "energy": ["TotEng", "KinEng", "PotEng", "E_angle", "E_bond", "E_coul",
                       "E_dihed", "E_impro", "E_long", "E_vdwl"],
            "temp": "Temp"
        },
        "openmm": {
            "energy": ["Total Energy (kJ/mole)", "Kinetic Energy (kJ/mole)",
                       "Potential Energy (kJ/mole)"],
            "temp": "Temperature (K)"
        }
    }

    def __init__(
            self, log_file: Union[str, Path] = None, log_format: str = None, *,
            energy: Union[np.ndarray[float], Q_] = None,
            temp: Union[float, Q_] = None, reduced: bool = False) -> None:

        self.results = {"units": {}}
        self._reduced = reduced

        if energy:
            if isinstance(energy, Q_):
                self.results["energy"] = energy.magnitude
                self.results["units"]["results.energy"] = energy.units
            else:
                self.results["energy"] = energy
                if not reduced:
                    self.results["units"]["results.energy"] = ureg.kilojoule / ureg.mole
        elif log_file:
            self._file = log_file if isinstance(log_file, Path) else Path(log_file)
            with open(self._file, "r") as f:
                log = f.read()

            if log_format is None:
                for f, cs in self._COLUMNS.items():
                    if any(c in log for c in cs["energy"]):
                        log_format = f
                        break
                else:
                    raise ValueError("Could not determine log file format.")
            self._format = log_format

            if self._format == "lammps":
                if "minimize" in log:
                    log = log[log.index("Minimization stats:"):]
                log = log.split("\n")
                for i, line in enumerate(log):
                    if "Step" in line:
                        log = log[i:]
                        break
                log = "\n".join(log)
                log = log[:log.index("Loop time of ")]
                kwargs = {"delim_whitespace": True}
                if not reduced:
                    self.results["units"]["results.energy"] = ureg.kilocalorie / ureg.mole
            elif self._format == "openmm":
                kwargs = {"sep": ","}
                if reduced:
                    warnings.warn("OpenMM simulations always use real units.")
                self.results["units"]["results.energy"] = ureg.kilojoule / ureg.mole

            if self._COLUMNS[self._format]["energy"][0] in log:
                cols = self._COLUMNS[self._format]["energy"][:1]
            elif self._COLUMNS[self._format]["energy"][1] in log:
                cols = self._COLUMNS[self._format]["energy"][1:2]
                if self._COLUMNS[self._format]["energy"][2] in log:
                    cols.append(self._COLUMNS[self._format]["energy"][2])
                elif any(e in log for e in self._COLUMNS[self._format]["energy"][3:]):
                    for e in self._COLUMNS[self._format]["energy"][3:]:
                        if e in log:
                            cols.append(e)
                else:
                    raise ValueError("Potential energy column not found.")
            else:
                emsg = "Total energy and kinetic energy columns not found."
                raise ValueError(emsg)

            df = pd.read_csv(StringIO(log), **kwargs)
            self.results["energy"] = df[cols].sum(axis=1).to_numpy()

            if temp is None:
                self._COLUMNS[self._format]["temp"]
        else:
            raise ValueError("No log file or energy values provided.")

        if isinstance(temp, Q_):
            self._temp = temp.magnitude
            self.results["units"]["_temp"] = self._temp.units
        else:
            if temp is None:
                if log_file is None:
                    raise ValueError("No log file or temperature value provided.")
                self._temp = df[self._COLUMNS[self._format]["temp"]].mean()
            else:
                self._temp = temp
            if not reduced:
                self.results["units"]["_temp"] = ureg.kelvin

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[np.ndarray[int], slice] = None) -> None:

        if frames is None:
            frames = np.arange(start or 0,
                               stop or len(self.results["energy"]),
                               step)
        U = self.results["energy"][frames] * self.results["units"]["results.energy"]
        if self._reduced:
            pass
        else:
            C_V = (
                ((U ** 2).mean() - U.mean() ** 2)
                / (ureg.avogadro_constant ** 2 * ureg.boltzmann_constant
                * (self._temp * self.results["units"]["_temp"]) ** 2)
            ).to(ureg.kilojoule / ureg.kelvin)
            self.results["heat_capacity"] = C_V.magnitude
            self.results["units"]["results.heat_capacity"] = C_V.units