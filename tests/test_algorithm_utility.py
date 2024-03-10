import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper import ureg
from mdhelper.algorithm import utility # noqa: E402

rng = np.random.default_rng()

def test_func_closest_factors():

    # TEST CASE 1: Cube root of perfect cube
    factors = utility.get_closest_factors(1000, 3)
    assert np.allclose(factors, 10 * np.ones(3, dtype=int))

    # TEST CASE 2: Three closest factors in ascending order
    factors = utility.get_closest_factors(35904, 3)
    assert factors.tolist() == [32, 33, 34]

    # TEST CASE 3: Four closest factors in descending order
    factors = utility.get_closest_factors(73440, 4, reverse=True)
    assert factors.tolist() == [18, 17, 16, 15]

def test_func_replicate():

    # TEST CASE 1: Replicate two vectors
    dims = rng.integers(1, 5, size=3)
    n_cells = rng.integers(2, 10, size=3)
    pos = utility.replicate(dims, np.array(((0, 0, 0), dims // 2)), n_cells)
    assert pos.shape[0] == 2 * n_cells.prod()
    assert np.allclose(pos[2], (dims[0], 0, 0))

def test_func_rebin():

    # TEST CASE 1: Rebin 1D array
    arr = np.arange(50)
    ref = np.arange(2, 52, 5)
    assert np.allclose(utility.rebin(arr), ref)

    # TEST CASE 2: Rebin 2D array
    assert np.allclose(utility.rebin(np.tile(arr[None, :], (5, 1))),
                       np.tile(ref[None, :], (5, 1)))
    
    # TEST CASE 3: No factor specified and cannot be determined
    with pytest.raises(ValueError):
        utility.rebin(np.empty((17,)))

def test_func_get_lj_scaling_factors():
    
    # TEST CASE 1: Lennard-Jones scaling factors
    pint_factors = utility.get_lj_scaling_factors({
        "mass": 39.948 * ureg.gram / ureg.mole,
        "energy": 3.9520829798737548e-25 * ureg.kilocalorie,
        "length": 3.4 * ureg.angstrom
    })
    openmm_factors = utility.get_lj_scaling_factors({
        "mass": 39.948 * unit.gram / unit.mole,
        "energy": 0.238 * unit.kilocalorie_per_mole / unit.AVOGADRO_CONSTANT_NA,
        "length": 3.4 * unit.angstrom
    })
    for key in openmm_factors.keys():
        value, unit_ = utility.strip_unit(openmm_factors[key])
        assert np.isclose(utility.strip_unit(pint_factors[key], unit_)[0],
                          value)

def test_func_strip_unit():

    # TEST CASE 1: Strip unit from non-Quantity
    assert utility.strip_unit(90.0, "deg") == (90.0, "deg")
    assert utility.strip_unit(90.0, ureg.degree) == (90.0, ureg.degree)
    assert utility.strip_unit(90.0, unit.degree) == (90.0, unit.degree)

    # TEST CASE 2: Strip unit from Quantity
    k_ = 1.380649e-23
    assert utility.strip_unit(k_) == (k_, None)
    assert utility.strip_unit(k_ * ureg.joule * ureg.kelvin ** -1) \
           == (k_, ureg.joule * ureg.kelvin ** -1)
    assert utility.strip_unit(k_ * unit.joule * unit.kelvin ** -1) \
           == (k_, unit.joule * unit.kelvin ** -1)

    # TEST CASE 3: Strip unit from Quantity with compatible unit specified
    g_ = 32.17404855643044
    g = 9.80665 * ureg.meter / ureg.second ** 2
    assert utility.strip_unit(g_, "foot/second**2") \
           == (g_, ureg.foot / ureg.second ** 2)
    assert utility.strip_unit(g, ureg.foot / ureg.second ** 2) \
           == (g_, ureg.foot / ureg.second ** 2)
    g = 9.80665 * unit.meter / unit.second ** 2
    assert utility.strip_unit(g, "foot/second**2") \
           == (g_, unit.foot / unit.second ** 2)
    assert utility.strip_unit(g, unit.foot / unit.second ** 2) \
           == (g_, unit.foot / unit.second ** 2)

    # TEST CASE 4: Strip unit from Quantity with incompatible unit specified
    R_ = 8.31446261815324
    R__ = 8.205736608095969e-05
    assert utility.strip_unit(
        R__ * ureg.meter ** 3 * ureg.atmosphere / (ureg.kelvin * ureg.mole),
        unit.joule / (unit.kelvin * unit.mole)
    ) == (R_, unit.joule / (unit.kelvin * unit.mole))
    assert utility.strip_unit(
        R__ * unit.meter ** 3 * unit.atmosphere / (unit.kelvin * unit.mole),
        ureg.joule / (ureg.kelvin * ureg.mole)
    ) == (R_, ureg.joule / (ureg.kelvin * ureg.mole))

    # TEST CASE 5: Strip unit from Quantity with non-standard 
    # incompatible unit specified
    with pytest.raises(ValueError):
        utility.strip_unit(
            R_ * unit.joule / (unit.kelvin * unit.mole),
            ureg.meter ** 3 * ureg.atmosphere / (ureg.kelvin * ureg.mole)
        )