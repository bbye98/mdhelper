import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.tests.datafiles import TPR, XTC
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdhelper.analysis import structure # noqa: E402

rng = np.random.default_rng()

def test_func_radial_histogram():

    L = 20
    half_L = L // 2
    dims = np.array((L, L, L, 90, 90, 90), dtype=int)
    origin = half_L * np.ones(3)

    N = 1000
    norm = L // 2 * rng.random(N)
    counts = np.histogram(norm, bins=half_L, range=(0, half_L + 1))[0]
    neighbors = rng.random((N, 3))
    neighbors *= norm[:, None] / np.linalg.norm(neighbors, axis=1, keepdims=True)
    neighbors += dims[:3] / 2

    # TEST CASE 1: Correct radial histogram for randomly placed particles
    assert np.array_equal(counts,
                          structure.radial_histogram(origin, neighbors,
                                                     n_bins=half_L,
                                                     range=(0, half_L + 1),
                                                     dims=dims))

def test_func_radial_fourier_transform():

    alpha = 1 + 9 * rng.random()
    r = np.linspace(1e-8, 20, 2000)
    q = 1 / r
    f = np.exp(-alpha * r) / r
    F = 4 * np.pi / (alpha ** 2 + q ** 2)

    # TEST CASE 1: Radial Fourier transform of function exp(-ar)/r,
    # which has analytical form 4*pi/(a^2+q^2)
    assert np.allclose(F, structure.radial_fourier_transform(r, f, q),
                       atol=4e-5)

"""
The following test cases are adapted from the "Average radial
distribution functions" page from the MDAnalysis User Guide
(https://userguide.mdanalysis.org/stable/examples/analysis/structure/average_rdf.html).
"""

universe = mda.Universe(TPR, XTC)
res60 = universe.select_atoms("resid 60")
water = universe.select_atoms("resname SOL")
thr = universe.select_atoms("resname THR")
n_bins = 75

def test_class_rdf_residue60_water():

    rdf = InterRDF(res60, water, nbins=n_bins).run()

    # TEST CASE 1: Batched serial RDF calculation
    serial_rdf = structure.RDF(res60, water, n_bins=n_bins, n_batches=2).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Batched parallel RDF calculation
    parallel_rdf = structure.RDF(res60, water, n_bins=n_bins, n_batches=2, 
                                 parallel=True).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

def test_class_rdf_residue60_exclusion_self():

    exclusion = (1, 1)
    rdf = InterRDF(res60, res60, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RDF(res60, n_bins=n_bins, exclusion=exclusion).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RDF(res60, n_bins=n_bins, exclusion=exclusion, 
                                 parallel=True).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

def test_class_rdf_threonine_exclusion_self():

    exclusion = (14, 14)
    rdf = InterRDF(thr, thr, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RDF(thr, n_bins=n_bins, exclusion=exclusion).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RDF(thr, n_bins=n_bins, exclusion=exclusion, 
                                 parallel=True).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

def test_class_rdf_threonine_exclusion_carbon():

    exclusion = (4, 10)
    rdf = InterRDF(thr, thr, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RDF(thr, n_bins=n_bins, exclusion=exclusion).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RDF(thr, n_bins=n_bins, exclusion=exclusion, 
                                 parallel=True).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)