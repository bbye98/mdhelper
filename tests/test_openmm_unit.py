import pathlib
import sys

from openmm import unit

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper.openmm import unit as u # noqa: E402

def test_func_lj_scaling():

    temp = 300 * unit.kelvin

    # TEST CASE 1: Correct units for complex scaling factors
    scales = u.lj_scaling(
        {"mass": 18.0153 * unit.amu, 
         "length": 0.275 * unit.nanometer, 
         "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule)}
    )
    assert scales["molar_energy"].unit == unit.kilojoule_per_mole
    assert scales["velocity"].unit == unit.nanometer / unit.picosecond
    assert scales["electric_field"].unit \
           == unit.kilojoule_per_mole / (unit.nanometer * unit.elementary_charge)

    # TEST CASE 2: No default scaling factors
    scales = u.lj_scaling(
        {"mass": 18.0153 * unit.amu, 
         "length": 0.275 * unit.nanometer, 
         "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule)},
        {"diffusivity": (("length", 2), ("time", -1))},
        default=False
    )
    assert "time" not in scales

    # TEST CASE 3: Custom scaling factors
    assert scales["diffusivity"].unit == unit.nanometer ** 2 / unit.picosecond