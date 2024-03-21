<img src="https://raw.githubusercontent.com/bbye98/mdhelper/main/assets/logo.png" align="right" width="256"/>

# MDHelper

[![continuous-integration](https://github.com/bbye98/mdhelper/actions/workflows/ci.yml/badge.svg)](https://github.com/bbye98/mdhelper/actions/workflows/ci.yml)

MDHelper is a toolkit of optimized analysis modules and helper
functions for molecular dynamics (MD) simulations.

* **Documentation**: https://bbye98.github.io/mdhelper/
* **Conda**: https://anaconda.org/bbye98/mdhelper
* **Python Package Index (PyPI)**: https://pypi.org/project/mdhelper/

Note that MDHelper is currently an *experimental* library that has 
only been tested on Linux and may contain bugs and issues. If you come 
across one or would like to request new features, please 
[submit a new issue](https://github.com/bbye98/mdhelper/issues/new).

## Features

* [`algorithm`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/algorithm): 
Efficient NumPy and SciPy algorithms for data wrangling and evaluating 
structural and dynamical properties.
* [`analysis`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/analysis): 
Serial and parallel data analysis tools built on top of the MDAnalysis 
framework.
* [`fit`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/fit): 
Two-dimensional curve fitting models for use with SciPy.
* [`lammps`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/lammps):
Helper functions for setting up LAMMPS simulations.
* [`openmm`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/openmm): 
Extensions to the high-performance OpenMM toolkit, such as custom 
bond/pair potentials, support for NetCDF trajectories, and much more.
* [`plot`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/plot): 
Settings and additional functionality for Matplotlib figures.

## Installation

MDHelper requires Python 3.9 or later.

For the most up-to-date version of MDHelper, clone the repository and install
the package using pip:

    git clone https://github.com/bbye98/mdhelper.git
    cd mdhelper
    python -m pip install -e .

MDHelper is also available on Conda and PyPI:

    conda install -c bbye98 mdhelper
    python -m pip install mdhelper

### Prerequisites

If you use pip to manage your Python packages, you must compile and 
install OpenMM prior to installing MDHelper since OpenMM is not 
available in PyPI. See the 
["Compiling OpenMM from Source Code"](http://docs.openmm.org/latest/userguide/library/02_compiling.html) 
section of the OpenMM User Guide for more information.

If you use Conda, it is recommended that you use the conda-forge 
channel to install dependencies. To make conda-forge the default 
channel, use

    conda config --add channels conda-forge

### Postrequisites

To use the image of method charges 
(`mdhelper.openmm.system.add_image_charges()`) in your OpenMM simulations, you must
compile and install [`constvplugin`](https://github.com/scychon/openmm_constV) or
[`openmm-ic-plugin`](https://github.com/bbye98/mdhelper/tree/main/lib/openmm-ic-plugin).