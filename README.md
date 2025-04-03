# Berry Phase Analysis

This repository contains a Python script for analyzing Berry phases using various methods. The script calculates Berry connections, Berry phases, Berry curvature, and visualizes the potential components \( V_A \) and \( V_X \) for a given Hamiltonian.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)
- [Documentation](#documentation)
- [Notes](#notes)

## Installation

To run this script, you need Python 3.x and the following packages:
- NumPy
- Matplotlib
- SciPy

You can install the required packages using pip:

```bash
pip install numpy matplotlib scipy
```

## Usage
Clone the repository:

```bash
git clone https://github.com/kirzolaa/Arrowhead.git
```
Note: You need to have the Arrowhead/generalized package installed. You also can just copy-paste the generalized directory to the root folder as well, since the scripts are using the generalized folder directly now.

Run the script:

```bash
python new_bph.py
```

## Parameters

The script uses the following parameters:

- `omega`: Frequency
- `aVx`: Potential parameter
- `aVa`: Potential parameter
- `c_const`: Potential constant
- `x_shift`: Shift in x direction
- `d`: Radius of the circle
- `theta_min`: Minimum angle
- `theta_max`: Maximum angle
- `num_points`: Number of points to generate

## Output

The script generates the following output files in the specified output directory:

- `plots`: Directory containing the generated plots.
- `npy`: Directory containing the generated numpy files.
- `vectors`: Directory containing the generated vector files.

## Documentation

The documentation for the repository can be found in the `docs` directory. I have also written some important notes for the `new_bph.py` script in the docs directory.

## Notes

I think this implementation is correct both mathematically and physically. Further analysis is needed to confirm this.