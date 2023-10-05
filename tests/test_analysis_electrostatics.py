# import pathlib
# import sys

# import MDAnalysis as mda
# from MDAnalysis.tests.datafiles import PSF_TRICLINIC, DCD_TRICLINIC
# from MDAnalysis.analysis.dielectric import DielectricConstant
# import numpy as np

# sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
# from mdhelper.analysis import electrostatics # noqa: E402

# def test_class_dipole_moment():

#     """
#     The test cases are adapted from the "Dielectric — 
#     :code:`MDAnalysis.analysis.dielectric`" page from the MDAnalysis
#     User Guide (https://docs.mdanalysis.org/stable/documentation_pages/analysis/dielectric.html).
#     """

#     universe = mda.Universe(PSF_TRICLINIC, DCD_TRICLINIC)

#     diel = DielectricConstant(universe.atoms)
#     diel.run()

#     rp = electrostatics.DipoleMoment(universe.atoms, unwrap=True)
#     rp.run()
#     rp.calculate_relative_permittivity(300)

#     # TEST CASE 1: Relative permittivity of water system
#     #   Note: MDAnalysis module forgets to divide by 3. When this is
#     #   fixed upstream, the following line will need to be updated.
#     assert np.allclose((diel.results.eps - 1) / (rp.results.dielectric - 1), 3)

# test_class_dipole_moment()