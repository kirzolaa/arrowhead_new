# Berry Phase Analysis

This repository contains a Python script for analyzing Berry phases using the Wilson loop method. The script calculates Berry phases, Berry curvature, and visualizes the potential components \( V_A \) and \( V_X \) for a given Hamiltonian.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)

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

Run the script:

```bash
python berry_phase_analysis.py
```

## Parameters

The script uses the following parameters:

- `c`: Coupling constant
- `omega`: Frequency
- `a`: Potential parameter
- `b`: Potential parameter
- `c_const`: Potential constant
- `x_shift`: Shift in x direction
- `y_shift`: Shift in y direction
- `d`: Radius of the circle
- `theta_min`: Minimum angle
- `theta_max`: Maximum angle
- `num_points`: Number of points to generate

## Output

The script generates the following output files in the specified output directory:

- `Hamiltonians.npy`: Contains the Hamiltonians for each theta value.
- `Va_values.npy`: Contains the potential values for ( V_A ).
- `Vx_values.npy`: Contains the potential values for ( V_X ).
- `eigenvalues.npy`: Contains the eigenvalues of the Hamiltonians.
- `eigenvectors.npy`: Contains the eigenvectors of the Hamiltonians.
- `berry_phases.dat`: Contains the calculated Berry phases.
- `accumulated_phases.dat`: Contains the accumulated phases.
- `phase_log_berry_curvature.out`: Log of phase differences and accumulated phases.

