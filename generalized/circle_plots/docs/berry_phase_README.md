# Berry Phase Calculation

## Overview

This document provides a guide to the Berry phase calculation implemented in the Arrowhead project. The Berry phase is a geometric phase acquired by a quantum state when transported along a closed path in parameter space. It is a fundamental concept in quantum mechanics with important applications in topological physics.

## Implementation

The Berry phase calculation is implemented in `new_berry_phase.py`. The script:

1. Loads eigenvectors from `.npy` files generated for different values of the parameter θ
2. Computes the Berry phase for each eigenstate by calculating overlaps between consecutive eigenvectors
3. Normalizes and quantizes the Berry phases to identify their topological properties
4. Visualizes the results with various plots
5. Generates a detailed summary of the Berry phase analysis

## Running the Calculation

To run the Berry phase calculation:

```bash
# Activate the virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Run the Berry phase calculation
python new_berry_phase.py
```

By default, the script looks for eigenvector files in the `berry_phase_r0_000_theta_0_360_5/berry_phase_logs` directory. You can specify a different directory using command-line arguments.

## Key Features

- **Comprehensive Berry Phase Analysis**: Calculates raw Berry phases, normalized phases (mod 2π), and quantized phases (multiples of π)
- **Winding Number Calculation**: Identifies how many times the phase wraps around 2π during the parameter cycle
- **Overlap Analysis**: Checks the quality of eigenvector overlaps to identify potential numerical issues
- **Visualization**: Generates plots of overlap magnitudes, phase angles, cumulative Berry phases, and more
- **Detailed Output**: Produces a comprehensive text summary of the Berry phase results

## Results Interpretation

The Berry phase calculation shows that all eigenstates have a Berry phase of π (mod 2π), which is the expected behavior for a system where the parameter path encircles a degeneracy point.

The different winding numbers for each eigenstate (ranging from 3 to 9) reflect the different rates at which the phase accumulates along the path, but all correctly result in a final Berry phase of π (mod 2π).

## Output Files

The script generates the following output:

- **Berry Phase Plots**: Saved in the `berry_phase_plots` directory
- **Berry Phase Summary**: Saved as `berry_phase_results/berry_phase_summary.txt`

## Theoretical Background

The Berry phase γₙ for the n-th eigenstate |n(R)⟩ is defined as:

```
γₙ = i ∮ ⟨n(R)|∇ᵣ|n(R)⟩ · dR
```

In practice, we calculate the Berry phase using a discrete approximation:

```
γₙ ≈ -Im ln ∏ⱼ₌₁ᴺ ⟨n(Rⱼ)|n(Rⱼ₊₁)⟩
```

This formula computes the Berry phase from the overlaps between eigenstates at consecutive points along the path.

## References

1. Berry, M. V. (1984). Quantal phase factors accompanying adiabatic changes. Proceedings of the Royal Society of London. A. Mathematical and Physical Sciences, 392(1802), 45-57.
2. Xiao, D., Chang, M. C., & Niu, Q. (2010). Berry phase effects on electronic properties. Reviews of Modern Physics, 82(3), 1959.
3. Resta, R. (2000). Manifestations of Berry's phase in molecules and condensed matter. Journal of Physics: Condensed Matter, 12(9), R107.
4. Bohm, A., Mostafazadeh, A., Koizumi, H., Niu, Q., & Zwanziger, J. (2013). The geometric phase in quantum systems: foundations, mathematical concepts, and applications in molecular and condensed matter physics. Springer Science & Business Media.
