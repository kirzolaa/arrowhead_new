"""
Berry Phase Analysis Script

This script uses functions from improved_berry_phase.py to generate Va, Vx, and arrowhead matrices,
and calculates the Berry phase using the Wilson loop method.
"""

import numpy as np
from improved_berry_phase import (
    generate_arrowhead_matrix,
    calculate_numerical_berry_phase,
    calculate_wilson_loop_berry_phase
)

# Parameters for the arrowhead matrix
c = 1.0  # Coupling constant

# Generate the arrowhead matrix and Va, Vx
H, R_theta, Vx, Va = generate_arrowhead_matrix(c)

# Define theta values for the loop
theta_vals = np.linspace(0, 2 * np.pi, 1000)

# Calculate eigenvectors at each theta value
eigenvectors = np.array([np.linalg.eigh(H(theta))[1] for theta in theta_vals])

# Calculate the Berry phase using the numerical method
berry_phases_num, accumulated_phases_num = calculate_numerical_berry_phase(theta_vals, eigenvectors)

# Calculate the Berry phase using the Wilson loop method
berry_phases_wilson, accumulated_phases_wilson = calculate_wilson_loop_berry_phase(theta_vals, eigenvectors)

# Print the results
print("Numerical Berry Phases:", berry_phases_num)
print("Wilson Loop Berry Phases:", berry_phases_wilson)
