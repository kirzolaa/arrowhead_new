# Berry Phase Analysis with Tau Matrix Method

This repository contains a comprehensive implementation for analyzing Berry phases in quantum systems using various numerical methods, with a focus on the tau matrix method. The code calculates Berry connections, Berry phases, and Berry curvature for a 4×4 arrowhead Hamiltonian system, and visualizes the results through various plots and analyses.

## Table of Contents
- [Introduction](#introduction)
- [Methods Implemented](#methods-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Output](#output)
- [Documentation](#documentation)

## Introduction

The Berry phase is a geometric phase acquired by a quantum system when it undergoes adiabatic evolution along a closed path in parameter space. This repository provides tools to calculate and analyze Berry phases using multiple complementary methods, with particular focus on the tau matrix method which enables detailed analysis of the coupling between different eigenstates.

## Methods Implemented

The repository implements several methods for Berry phase calculation:

1. **Tau Matrix Method**: Calculates the full Berry connection matrix τ<sub>nm</sub>(θ) = ⟨ψ<sub>m</sub>(θ)|∂/∂θ|ψ<sub>n</sub>(θ)⟩ and integrates it to obtain the Berry phase matrix.

2. **Wilson Loop Method**: Computes Berry phases directly from the overlaps between consecutive eigenstates along the closed path.

3. **Berry Curvature Calculation**: Implements direct calculation of the Berry curvature as the curl of the Berry connection.

4. **Visualization Tools**: Provides comprehensive visualization of eigenvalues, eigenvectors, Berry connection, and Berry phase evolution.

## Installation

To run the scripts in this repository, you need Python 3.x and the following packages:
- NumPy
- Matplotlib
- SciPy

It is recommended to use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy matplotlib scipy
```

## Usage

Clone the repository:

```bash
git clone https://github.com/kirzolaa/arrowhead_new.git
cd arrowhead_new
```

The repository contains several scripts for different analyses:

1. For the main Berry phase calculation using the tau matrix method:
```bash
python -m bph.py or python -m gabor_bph.py
```

2. For the Hamiltonian implementation and basic analysis:
```bash
python -m new_bph.py
```

3. For the Wilson loop method implementation:
```bash
python -m deprecated_scriptz/check_tau.py
```

Note: The original Berry phase calculator is deprecated and is not recommended for use. It is only provided for reference. Use Gábor's repaired function since it seems to be working as intented. My function gives a smaller difference from pi than Gábor's function.

## Parameters

The scripts use the following physical parameters:

- `omega`: Frequency parameter in the Hamiltonian
- `aVx`: Coefficient for the potential V<sub>x</sub>(R) = aVx·R²
- `aVa`: Coefficient for the potential V<sub>a</sub>(R) = aVa·(R-x_shift)²+c
- `c_const`: Constant offset in the potential
- `x_shift`: Shift parameter in the potential V<sub>a</sub>
- `d`: Radius of the circle in parameter space
- `theta_min`, `theta_max`: Angular range for the closed path
- `num_points`: Number of discretization points (higher values give better accuracy)
- `R_0`: Origin point for the parameter space

## Output

The scripts generate the following outputs in the specified directories:

- `plots/`: Contains visualization plots including:
  - Eigenvalue evolution
  - Eigenvector components
  - Tau matrix elements
  - Berry phase accumulation
  - Potential components
  - Verification plots

- `npy/`: Contains NumPy array files with raw numerical data:
  - Eigenvectors and eigenvalues
  - Tau matrices
  - Berry phases
  - Parameter vectors

- `vectors/`: Contains verification results for the parameter space path

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- `tau_matrix_method.tex`: Detailed explanation of the tau matrix method implementation
- `misc/formulation.tex`: Mathematical formulation of the Hamiltonian
- `misc/formulation_2.tex`: Berry phase calculation details

Additional documentation for the generalized package can be found in the `generalized/docs/` directory.