# Berry Phase Analysis

This repository contains a Python script for analyzing Berry phases using the Wilson loop method. The script calculates Berry phases, Berry curvature, and visualizes the potential components \( V_A \) and \( V_X \) for a given Hamiltonian.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)
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
- `eigenvector_diff.out`: Log of eigenvector differences.
- `summary.txt`: Summary of the analysis.

## Notes

The script uses natural units where ħ = 1.

The script imports the `create_perfect_orthogonal_vectors` function from the Arrowhead/generalized package. If the import fails, the script will fall back to a simple circle implementation.

The script is currently not working well, since the parameters are not well-chosen.
The d should be small, and the VA potential should be shifted by the x and y shift parameters well.
The omega should be a well-chosen value, which is a bit smaller than the y shift parameter.
I have been too wrong about the y_shift parameter.
The y_shift should be the c_const parameter, since it is the y axis shifting in the potential.