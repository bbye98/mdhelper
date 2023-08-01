"""
Test cases for mdhelper.analysis.profile module
===============================================
The test cases for :class:`mdhelper.analysis.profile.DensityProfile` are 
adapted from the "Computing mass and charge density on each axis" page 
from the MDAnalysis User Guide
(https://userguide.mdanalysis.org/stable/examples/analysis/volumetric/linear_density.html).
"""

import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import waterDCD, waterPSF
from MDAnalysis.analysis.lineardensity import LinearDensity
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper.analysis import profile

universe = mda.Universe(waterPSF, waterDCD)

def test_class_density_profile():

    """
    The test cases are adapted from the "Computing mass and charge 
    density on each axis" page from the MDAnalysis User Guide
    (https://userguide.mdanalysis.org/stable/examples/analysis/volumetric/linear_density.html).
    """

    density = LinearDensity(universe.atoms, grouping="residues").run()
    density_profile = profile.DensityProfile(universe.atoms, "residues",
                                             n_bins=200).run()

    for i, axis in enumerate("xyz"):
        
        # TEST CASE 1: Number density profiles
        assert(
            np.allclose(
                0.602214076 * getattr(density.results, axis).mass_density 
                / universe.residues.masses[0],
                density_profile.results["number_density"][i]
            )
        )
        
        # TEST CASE 2: Charge density profiles
        assert(
            np.allclose(
                0.602214076 * getattr(density.results, axis).charge_density,
                density_profile.results["charge_density"][i]
            )
        )