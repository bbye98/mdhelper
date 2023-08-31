# MDHelper

[![continuous-integration](https://github.com/bbye98/mdhelper/actions/workflows/ci.yml/badge.svg)](https://github.com/bbye98/mdhelper/actions/workflows/ci.yml)

MDHelper is a batteries-included toolkit of analysis modules and helper
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
section of the OpenMM User Guide for more information.

If you use Conda, it is recommended that you use the conda-forge 
channel to install dependencies. To make conda-forge the default 
channel, use

    conda config --add channels conda-forge

### Virtual environment

It is recommended, but not necessary, that you create a virtual 
environment to prevent dependency conflicts.

#### pip: `venv` or `virtualenv`

If you use pip as your Python package manager, you can create a virtual 
environment using either the built-in `venv` or the (better) `virtualenv`
packages. With `venv`, run

    python -m venv <venv_path>

to initialize the new virtual environment, where `<venv_path>` is the 
path to the directory to be created, and one of the following commands 
to activate the environment, depending on your operating system (OS) and 
shell:

* POSIX: bash/zsh

      source <venv_path>/bin/activate

* POSIX/Windows: PowerShell

      <venv_path>\Scripts\Activate.ps1

* Windows: cmd.exe

      <venv_path>\Scripts\activate.bat

With `virtualenv`, you can create a virtual environment using

    virtualenv <venv_name>

where `<venv_name>` is the name of the new environment, and activate it 
using

* Linux or macOS:

      source <venv_name>/bin/activate

* Windows: 

      .\<venv_name>\Scripts\activate

#### Conda: `conda` or `mamba`

If you use Conda as your Python package manager, you can create and 
activate a virtual environment named `<venv_name>` using

    conda create --name <venv_name>
    conda activate <venv_name>

(For Mamba users, replace `conda` with `mamba`.)

### Option 1: Install using pip

 1. Install MDHelper and its dependencies through pip using 

        python -m pip install mdhelper

 2. To verify that MDHelper has been installed correctly, execute

        python -c "import mdhelper"

### Option 2: Install using Conda

 1. Install MDHelper and its dependencies through Conda using

        conda install -c bbye98 mdhelper

 2. To verify that MDHelper has been installed correctly, execute

        python -c "import mdhelper"

### Option 3: Install from source

 1. Change to the directory where you want to store a copy of MDHelper 
    using

        cd <install_path>

    where `<install_path>` is the path to the desired directory. If you
    are already in the correct location, skip this step.

 2. Create a local copy of the MDHelper repository on your machine using

        git clone https://github.com/bbye98/mdhelper.git

 3. Enter the root directory of MDHelper using

        cd mdhelper

 4. Install MDHelper and its dependencies through pip using

        python -m pip install -e .

 5. To verify that MDHelper has been installed correctly, execute

        python -c "import mdhelper"

### Option 4: Portable package

 1. Change to the directory where you want to store a copy of MDHelper 
    using

        cd <install_path>

    where `<install_path>` is the path to the desired directory. If you
    are already in the correct location, skip this step.

 2. Create a local copy of the MDHelper repository on your machine using

        git clone https://github.com/bbye98/mdhelper.git

 3. Enter the root directory of MDHelper using

        cd mdhelper

 4. Install the required dependencies using

        python -m pip install -r requirements_minimal.txt

    or

        conda install --file requirements_minimal.txt

    If you would like to install the optional dependencies as well,
    remove the `_minimal` prefix from the filenames above.

 5. Now, you can use MDHelper by first adding the path to the `src` 
    directory in your Python scripts. To verify that MDHelper has been 
    installed correctly, execute

        python -c "import sys; sys.path.insert(0, '<install_path>/mdhelper/src'); import mdhelper"

### Postrequisites

To use the image of method charges 
(`mdhelper.openmm.system.image_charges()`) in your simulations, you must
compile and install [`constvplugin`](https://github.com/scychon/openmm_constV).