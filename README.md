# Berry Phase and Energy Gap Analysis for Arrowhead Hamiltonian

This repository contains scripts for calculating Berry phases and analyzing energy gaps in quantum systems, with a focus on an arrowhead Hamiltonian structure.

## Overview

The Berry phase is a geometric phase acquired by a quantum state when it is transported around a closed loop in parameter space. In our case, we're studying how the eigenstates of a 4×4 arrowhead Hamiltonian evolve as we vary the parameter θ from 0 to 2π. Additionally, we analyze the energy gaps between eigenstates to understand the relationship between gap structure and topological properties.

## Key Files

### Berry Phase Calculation
1. **fixed_new.py**: Initial attempt at calculating Berry phases for the arrowhead Hamiltonian.
2. **explicit_berry_phase.py**: Modified Hamiltonian with explicit Berry phase terms.
3. **direct_berry_phase.py**: Simplified two-level system with a known analytical Berry phase.
4. **analytical_implementation.py**: Direct analytical implementation of Berry phase calculation.
5. **arrowhead_analytical.py**: Arrowhead Hamiltonian with analytical Berry phase calculation.
6. **final_arrowhead_berry.py**: Comprehensive script with both numerical and analytical approaches.
7. **improved_berry_phase.py**: Enhanced implementation with both analytical and numerical (Wilson loop) methods for Berry phase calculation, including detailed degeneracy analysis.

### Energy Gap Analysis
8. **energy_gap_analysis.py**: Module for calculating and analyzing energy gaps between eigenstates.
9. **run_optimal_gap_analysis.py**: Script to run energy gap analysis with optimal parameters.
10. **gap_topology_analysis.py**: Advanced analysis of the relationship between energy gaps and Berry phases.

## Methods for Berry Phase Calculation

We've implemented several methods for calculating Berry phases:

1. **Numerical Differentiation**: Calculating the Berry connection as A_n(θ) = -i⟨ψ_n(θ)|∂ψ_n(θ)/∂θ⟩ using finite differences.
2. **Overlap Method**: Approximating the Berry connection as A_n(θ_i) ≈ Im[log(⟨ψ_n(θ_i)|ψ_n(θ_{i+1})⟩)].
3. **Wilson Loop Method**: Calculating the Berry phase directly as γ_n = -Im[log(∏_i ⟨ψ_n(θ_i)|ψ_n(θ_{i+1})⟩)].
4. **Analytical Approximation**: Using analytical formulas based on the structure of the Hamiltonian.

## Results

### Berry Phases with Parabolic Potentials:
- State 0: 0.0
- State 1: -π (-3.1416)
- State 2: -π (-3.1416)
- State 3: 0.0

### Original Optimal Parameter Berry Phases:
- State 0: 0.0
- State 1: -π/2 (-1.5708)
- State 2: π/2 (1.5708)
- State 3: 0.0

## Energy Gap Analysis

We analyze the energy gaps between eigenstates as a function of the adiabatic parameter θ. This analysis provides insights into:

1. **Adiabaticity**: Small energy gaps can lead to diabatic transitions between states.
2. **Topological Transitions**: Gap closings often indicate topological phase transitions.
3. **Berry Phase Accumulation**: The rate of Berry phase accumulation is related to the energy gap structure.

### Key Gap Findings:
- Gap 0-1: Minimum of 0.000437 at θ = 0.3175
- Gap 1-2: Minimum of 0.000000 at θ = 0.0126 (near-degeneracy)
- Gap 2-3: Minimum of 274.291549 at θ = 0.0251

## Gap-Topology Analysis

We've developed tools to explore the relationship between energy gaps and topological properties:

1. **Parameter Scanning**: Systematically vary parameters to map out the gap and Berry phase landscape.
2. **Correlation Analysis**: Identify correlations between minimum gaps and Berry phase differences.
3. **Topological Transitions**: Locate parameter regions where gaps close and Berry phases change rapidly.

## Key Insights

1. **Phase Convention**: Proper phase convention is crucial for Berry phase calculations. We ensure that the first component of each eigenvector is real and positive.
2. **Explicit θ Dependence**: To obtain non-zero Berry phases, the Hamiltonian must have explicit θ dependence, typically in the form of complex exponentials e^(±iθ).
3. **Wilson Loop Method**: The Wilson loop method is more reliable for numerical Berry phase calculations than direct integration of the Berry connection.
4. **Analytical Approximation**: For simple Hamiltonians, analytical approximations can provide insight into the expected Berry phases.
5. **Gap-Phase Relationship**: The near-degeneracy between states 1 and 2 corresponds to the states with non-zero Berry phases, suggesting a topological origin for this degeneracy.

## Usage

### Berry Phase Calculation

```bash
python3 improved_berry_phase.py
```

### Energy Gap Analysis

```bash
python3 Arrowhead/run_optimal_gap_analysis.py
```

### Gap-Topology Analysis

```bash
python3 Arrowhead/run_gap_topology_analysis.py
```

### Output

Each script will generate:

1. **Plots**: Visualizations of eigenvalues, Berry connections, energy gaps, and correlations.
2. **Summary Reports**: Text files containing key findings and statistics.
3. **Raw Data**: Numpy files (.npz) containing the raw numerical results for further analysis.
