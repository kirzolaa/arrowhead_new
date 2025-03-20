"""
Berry Phase Analysis Script

This script uses functions from improved_berry_phase.py to generate Va, Vx, and arrowhead matrices,
and calculates the Berry phase using the Wilson loop method.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import datetime

# Define physical constants
hbar = 1.0  # Using natural units where ħ = 1 (common in quantum mechanics)

# Import the perfect orthogonal circle generation function from the Arrowhead/generalized package
import sys
import os
sys.path.append('/home/zoli/arrowhead/Arrowhead/generalized')
try:
    from vector_utils import create_perfect_orthogonal_vectors
except ImportError:
    print("Warning: Could not import create_perfect_orthogonal_vectors from Arrowhead/generalized package.")
    print("Falling back to simple circle implementation.")
    # Define a fallback function if the import fails
    def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([1, -1/2, -1/2])  # First basis vector
        basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
        
        # Normalize the basis vectors
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Create a point at distance d from the origin in the plane spanned by basis1 and basis2
        R = np.array(R_0) + d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
        
        return R

# Function to create R_theta vector that traces a perfect circle orthogonal to the x=y=z line
def R_theta(d, theta):
    """
    Create a vector that traces a perfect circle orthogonal to the x=y=z line using the
    create_perfect_orthogonal_vectors function from the Arrowhead/generalized package.
    
    Parameters:
    d (float): The radius of the circle
    theta (float): The angle parameter
    
    Returns:
    numpy.ndarray: A 3D vector orthogonal to the x=y=z line
    """
    # Origin vector
    R_0 = np.array([0, 0, 0])
    
    # Generate the perfect orthogonal vector
    return create_perfect_orthogonal_vectors(R_0, d, theta)

# Define the potential functions V_x and V_a based on R_theta
def V_x(R_theta, a, b, c):
    # Calculate individual V_x components for each R_theta component
    Vx0 = a * R_theta[0]**2 + b * R_theta[0] + c
    Vx1 = a * R_theta[1]**2 + b * R_theta[1] + c
    Vx2 = a * R_theta[2]**2 + b * R_theta[2] + c
    return [Vx0, Vx1, Vx2]

def V_a(R_theta, a, b, c, x_shift, y_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va0 = a * (R_theta[0] - x_shift)**2 + b * (R_theta[0] - y_shift) + c
    Va1 = a * (R_theta[1] - x_shift)**2 + b * (R_theta[1] - y_shift) + c
    Va2 = a * (R_theta[2] - x_shift)**2 + b * (R_theta[2] - y_shift) + c
    return [Va0, Va1, Va2]

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, a, b, c_const)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, a, b, c_const, x_shift, y_shift)  # [Va0, Va1, Va2]
    
    # Create a 4x4 Hamiltonian with an arrowhead structure
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    H[1, 1] = Va[0] + Vx[1] + Vx[2]
    H[2, 2] = Vx[0] + Va[1] + Vx[2]
    H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    
    # Coupling between states 0 and 1 without theta dependence
    H[0, 1] = c 
    H[1, 0] = c 
    
    # Coupling between states 0 and 2 without theta dependence
    H[0, 2] = c 
    H[2, 0] = c 
    
    # Coupling between states 0 and 3 (constant)
    H[0, 3] = H[3, 0] = c
    
    return H, R_theta_val, Vx, Va

# Parameters for the arrowhead matrix
c = 1.0  # Coupling constant
omega = 1.0  # Frequency
a = 1.0  # Potential parameter
b = 0.5  # Potential parameter
c_const = 1.0  # Potential constant
x_shift = 0.2  # Shift in x direction
y_shift = 0.2  # Shift in y direction
d = 1.0  # Radius of the circle
theta_min = 0
theta_max = 2 * np.pi

# Generate the arrowhead matrix and Va, Vx
theta_vals = np.linspace(theta_min, theta_max, 1000, endpoint=True)

# Calculate eigenvectors at each theta value, explicitly including endpoint
eigenvectors = []
for i, theta in enumerate(theta_vals):
    # Diagonalize Hamiltonian
    evals, evecs = np.linalg.eigh(hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)[0])
    
    # Ensure phase continuity with previous eigenvector
    if i > 0:
        for j in range(evecs.shape[1]):
            if np.abs(np.vdot(evecs[:,j], eigenvectors[-1][:,j])) < 0:
                evecs[:,j] *= -1
    
    eigenvectors.append(evecs)

eigenvectors = np.array(eigenvectors)

# Calculate the Berry phase using the new wilson loop method
def calculate_wilson_loop_berry_phase_new(theta_vals, eigenvectors):
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]

    berry_phases = np.zeros(n_states)
    accumulated_phases = np.zeros((n_states, n_points))

    for state in range(n_states):
        phase_acc = 0.0  # Track phase accumulation
        
        for i in range(n_points - 1):
            current_eigenvector = eigenvectors[i, :, state]
            next_eigenvector = eigenvectors[i + 1, :, state]

            # Overlap calculation
            overlap = np.vdot(current_eigenvector, next_eigenvector)
            overlap /= np.abs(overlap)  # Normalize to unit phase factor
            phase_diff = np.angle(overlap)

            # Unwrap the phase difference and accumulate (wrap between -pi to pi)
            phase_acc += phase_diff
            phase_acc = np.arctan2(np.sin(phase_acc), np.cos(phase_acc))  # Wrap phase to -pi to pi
            
            # Store the accumulated phase
            accumulated_phases[state, i+1] = phase_acc

        # Final Berry phase is the last accumulated phase value
        berry_phases[state] = accumulated_phases[state, -1]
        
        # Optionally: add closure correction for full loop
        final_overlap = np.vdot(eigenvectors[-1, :, state], eigenvectors[0, :, state])
        final_overlap /= np.abs(final_overlap)
        berry_phase_correction = np.angle(final_overlap)
        berry_phases[state] += berry_phase_correction  # Adjust for closure

    # Unwrap accumulated phases across states (if needed)
    accumulated_phases = np.unwrap(accumulated_phases, axis=1)

    return berry_phases, accumulated_phases

# Calculate and plot eigenstate overlaps
overlaps = np.zeros((eigenvectors.shape[2], len(theta_vals)))

plt.figure(figsize=(12, 6))

for state in range(eigenvectors.shape[2]):
    for i in range(len(theta_vals)):
        current_eigenvector = eigenvectors[i, :, state]
        next_eigenvector = eigenvectors[(i + 1) % len(theta_vals), :, state]
        # Include endpoint by using the first eigenvector for the last point
        if i == len(theta_vals) - 1:
            next_eigenvector = eigenvectors[0, :, state]
        overlaps[state, i] = np.abs(np.vdot(current_eigenvector, next_eigenvector))
    
    plt.plot(theta_vals, overlaps[state], label=f'State {state}')


plt.xlabel('Theta')
plt.ylabel('Eigenstate Overlap')
plt.title('Eigenstate Overlaps vs Theta')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#plots
berry_phases, accumulated_phases = calculate_wilson_loop_berry_phase_new(theta_vals, eigenvectors)

# Create output directory and detailed report
output_dir = f'output_berry_phase_results_thetamin_{theta_min}_thetamax_{theta_max}'
os.makedirs(output_dir, exist_ok=True)

# Write detailed text report
with open(f'{output_dir}/summary.txt', 'w') as f:
    f.write(f'Berry Phase Analysis Report\n')
    f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write(f'Parameters:\n')
    f.write(f'  c = {c}\n')
    f.write(f'  omega = {omega}\n')
    f.write(f'  a = {a}\n')
    f.write(f'  b = {b}\n')
    f.write(f'  c_const = {c_const}\n')
    f.write(f'  x_shift = {x_shift}\n')
    f.write(f'  y_shift = {y_shift}\n')
    f.write(f'  d = {d}\n\n')
    f.write(f'Calculation Parameters:\n')
    f.write(f'  Theta range: [{theta_min}, {theta_max}]\n')
    f.write(f'  Number of points: {len(theta_vals)}\n')
    f.write(f'  Number of states: {len(berry_phases)}\n\n')
    f.write(f'Results:\n')
    for state, phase in enumerate(berry_phases):
        f.write(f'  State {state}: Berry phase = {phase:.6f}\n')

# Write berry_phases to a .out file
with open(f'{output_dir}/berry_phases.out', 'w') as f:
    f.write('Berry Phase Accumulation Data\n')
    f.write('===========================\n\n')
    f.write('Final Berry Phases:\n')
    for state, phase in enumerate(berry_phases):
        f.write(f'State {state}: {phase:.8f}\n')
    f.write('\nAccumulated Berry Phases vs Theta:\n')
    f.write('Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
    for i, theta in enumerate(theta_vals):
        theta_deg = np.degrees(theta)
        f.write(f'{theta_deg:.2f}\t')
        for state in range(len(berry_phases)):
            f.write(f'{accumulated_phases[state][i]:.8f}\t')
        f.write('\n')

# Write berry_phases to a .dat file
np.savetxt(f'{output_dir}/berry_phases.dat', berry_phases, header='Berry phases for each state')

# Write accumulated_phases to a .dat file with theta values
with open(f'{output_dir}/accumulated_phases.dat', 'w') as f:
    f.write('# Theta (radians)\tState 0\tState 1\tState 2\tState 3\n')
    for i, theta in enumerate(theta_vals):
        f.write(f'{theta:.8f}\t')
        np.savetxt(f, accumulated_phases[:, i].reshape(1, -1), fmt='%.8f', delimiter='\t')

# Write theta values to a .dat file
np.savetxt(f'{output_dir}/theta_values.dat', theta_vals, header='Theta values used in calculation')

# Write eigenstate overlaps to file
with open(f'{output_dir}/eigenstate_overlaps.out', 'w') as f:
    f.write('Eigenstate Overlaps vs Theta\n')
    f.write('Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
    for i, theta in enumerate(theta_vals):
        theta_deg = np.degrees(theta)
        f.write(f'{theta_deg:.2f}\t')
        for state in range(eigenvectors.shape[2]):
            f.write(f'{overlaps[state, i]:.8f}\t')
        f.write('\n')

# Write eigenstates to file
with open(f'{output_dir}/eigenstates.out', 'w') as f:
    f.write('Eigenstates vs Theta\n')
    for i, theta in enumerate(theta_vals):
        theta_deg = np.degrees(theta)
        f.write(f'Theta = {theta_deg:.2f} degrees\n')
        for state in range(eigenvectors.shape[2]):
            f.write(f'State {state}:\n')
            np.savetxt(f, eigenvectors[i, :, state].reshape(1, -1), fmt='%.8f')
        f.write('\n')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for state in range(len(berry_phases)):
    plt.plot(theta_vals, accumulated_phases[state], label=f'State {state}')
plt.xlabel('Theta')
plt.ylabel('Berry Phase')
plt.title('Berry Phase vs Theta')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(theta_vals, accumulated_phases.T)
plt.xlabel('Theta')
plt.ylabel('Accumulated Phase')
plt.title('Accumulated Phase vs Theta')

plt.tight_layout()
plt.show()