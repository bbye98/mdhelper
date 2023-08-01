# MDHelper

MDHelper is a batteries-included toolkit of analysis modules and helper
functions for molecular dynamics (MD) simulations.

* **Documentation**: https://bbye98.github.io/mdhelper/
* **Python Package Index (PyPI)**: https://pypi.org/project/mdhelper/

Note that MDHelper is currently an *experimental* library that has 
only been tested on Linux and may contain bugs and issues. If you come 
across one, please 
[submit a new issue](https://github.com/bbye98/mdhelper/issues/new). If 
you would like to contribute to MDHelper, please 
[submit a pull request](https://github.com/bbye98/mdhelper/compare).

## Features

* [`algorithm`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/algorithm): 
Efficient NumPy algorithms for data wrangling and evaluating structural 
and dynamical properties.
* [`analysis`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/analysis): 
Serial and parallel data analysis tools built on top of the MDAnalysis 
framework.
* [`fit`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/fit): 
Two-dimensional curve fitting models for use with SciPy.
* [`openmm`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/openmm): 
Extensions to the high-performance OpenMM toolkit, such as custom 
bond/pair potentials, support for NetCDF trajectories, and much more.
* [`plot`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/plot): 
Settings and additional functionality for Matplotlib figures.
* [`utility`](https://github.com/bbye98/mdhelper/tree/main/src/mdhelper/utility): 
General utility functions.

## Installation and usage

### Prerequisites

If you use pip to manage your Python packages, you must compile and 
install OpenMM prior to installing MDHelper since OpenMM is not 
available in PyPI. See the 
["Compiling OpenMM from Source Code"](http://docs.openmm.org/latest/userguide/library/02_compiling.html) 
for more section of the OpenMM User Guide for more information.

### Virtual environment

It is recommended, but not necessary, that you create a virtual 
environment to prevent dependency conflicts.

If you are using pip to manage Python packages, use

    virtualenv venv
    source venv/bin/activate

where `venv` is the name of the new environment.

If you are using Anaconda or Miniconda, use

    conda create --name venv
    conda activate venv

### Option 1: Install using pip

 1. Install MDHelper and its dependencies using 

        python3 -m pip install mdhelper

 2. To test that MDHelper has been installed correctly, run

        python3 -c "import mdhelper"

### Option 2: Install using Anaconda or Miniconda

 1. Add the conda-forge channel using

        conda config --add channels conda-forge

 2. Install MDHelper and its dependencies using

        conda install mdhelper

 3. To test that MDHelper has been installed correctly, run

        python3 -c "import mdhelper"

### Option 3: Build from source

 1. Ensure that the latest version of the Python build frontend, 
    `build`, is installed:

        python3 -m pip install --upgrade build

 2. Create a local copy of the MDHelper repository on your machine using

        git clone https://github.com/bbye98/mdhelper.git

 3. Enter the root directory of MDHelper using

        cd mdhelper

 4. Build the MDHelper wheel using

        python3 -m build

    The Python wheel will be placed in the `dist` directory.

 5. Install MDHelper using

        python3 -m pip install dist/mdhelper-<version>-py3-none-any.whl

    where `<version>` is the version of MDHelper you are installing.

 6. To test that MDHelper has been installed correctly, run

        python -c "import mdhelper"

### Option 4: Portable package

 1. Change to the directory where you want to store a copy of MDHelper using

        cd <directory>

    where `<directory>` is the relative path to the desired directory. 
    If you are already in the correct location, skip this step.

 2. Create a local copy of the MDHelper repository on your machine using

        git clone https://github.com/bbye98/mdhelper.git

 3. Enter the root directory of MDHelper using

        cd mdhelper

 4. Install the dependencies using

        python3 -m pip install -r requirements.txt

    or

        conda install --file requirements.txt

 5. Once done, you can use MDHelper by first adding the path to the 
    `src` directory in your Python scripts. To test that MDHelper has 
    been installed correctly, run

        python3 -c "import sys; sys.path.insert(0, '/path/to/mdhelper/src'); import mdhelper"

### Postrequisites

To use the image of method charges 
(`mdhelper.openmm.system.image_charges()`) in your simulations, you must
compile and install [`constvplugin`](https://github.com/scychon/openmm_constV).