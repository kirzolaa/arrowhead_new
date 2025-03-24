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
from scipy.constants import hbar

# Import the perfect orthogonal circle generation function from the Arrowhead/generalized package
import sys
import os
sys.path.append('/home/zoli/arrowhead/Arrowhead/generalized')
try:
    from vector_utils import create_perfect_orthogonal_vectors
    print("Successfully imported create_perfect_orthogonal_vectors from Arrowhead/generalized package.")
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
def V_x(R_theta, a):
    # Calculate individual V_x components for each R_theta component
    Vx = [a * (R_theta[i] ** 2) for i in range(len(R_theta))]
    return Vx

def V_a(R_theta, a, b, c, x_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va = [a * ((R_theta[i] - x_shift) ** 2) + b * (R_theta[i] - x_shift) + c for i in range(len(R_theta))]
    return Va

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, a)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, a, b, c_const, x_shift)  # [Va0, Va1, Va2]
    
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
c = 0.2  # Coupling constant
omega = 0.1  # Frequency
a = 1.0  # Potential parameter
b = 1.0  # Potential parameter
c_const = 1.0  # Potential constant, shifts the 2d parabola on the y axis
x_shift = 1.0  # Shift in x direction
y_shift = 0.0  # Shift in y direction --> turns out that this is not a y axis shift like I wanted it!!!!!
d = 0.001  # Radius of the circle
theta_min = 0
theta_max = 2 * np.pi
num_points = 50
# Generate the arrowhead matrix and Va, Vx
theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

# Calculate eigenvectors at each theta value, explicitly including endpoint
eigenvectors = []
for i, theta in enumerate(theta_vals):
    # Diagonalize Hamiltonian
    evals, evecs = np.linalg.eigh(hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)[0])
    eigenvectors.append(evecs)

eigenvectors = np.array(eigenvectors)
output_dir = f'output_berry_phase_results_thetamin_{theta_min:.2f}_thetamax_{theta_max:.2f}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
# Improved phase wrapping with higher consistency for Berry phase accumulation
def calculate_wilson_loop_berry_phase_new(theta_vals, eigenvectors):
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]

    berry_phases = np.zeros(n_states)
    accumulated_phases = np.zeros((n_states, n_points))

    # Open a file to log phase_diff and phase_acc
    with open(f'{output_dir}/phase_log.out', "w") as log_file:
        log_file.write("Theta Phase_Diff Phase_Accumulated\n")  # Header

        for state in range(n_states):
            phase_acc = 0.0
            previous_phase = 0.0

            for i in range(n_points - 1):
                current_eigenvector = eigenvectors[i, :, state]
                next_eigenvector = eigenvectors[i + 1, :, state]

                overlap = np.conj(current_eigenvector).T @ next_eigenvector
                #overlap /= np.abs(overlap)
                phase_diff = np.angle(overlap)
                phase_diff = phase_diff - 2 * np.pi * np.round((phase_diff - previous_phase) / (2 * np.pi))

                phase_acc += phase_diff
                accumulated_phases[state, i + 1] = phase_acc
                previous_phase = phase_diff
                
                # Log the values
                log_file.write(f"{theta_vals[i]:.6f} {phase_diff:.6f} {phase_acc:.6f}\n")

            # Closure correction for full loop
            final_overlap = np.conj(eigenvectors[-1, :, state]).T @ eigenvectors[0, :, state]
            #final_overlap /= np.abs(final_overlap)
            berry_phase_correction = np.angle(final_overlap)
            berry_phases[state] = phase_acc + berry_phase_correction

    return berry_phases, accumulated_phases

    #log the norm of the difference between consecutive eigenvectors into a new log file in output directory
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/eigenvector_diff.out', "a") as log_file:
    log_file.write('#State Theta Norm_Diff\n')
    for i in range(1, len(theta_vals)):
        for j in range(eigenvectors.shape[2]):
            log_file.write(f"State {j}, Theta {theta_vals[i]:.2f}: {np.linalg.norm(eigenvectors[i, j] - eigenvectors[i-1, j]):.6f}\n")

def calculate_berry_curvature(eigenvectors, theta_vals, output_dir):
    """
    Calculate the Berry curvature (i.e., the rate of change of the Berry phase)
    by computing the overlap of consecutive eigenvectors.

    Parameters:
    eigenvectors (ndarray): Array of eigenvectors at different theta values.
    theta_vals (ndarray): Array of theta values at which eigenvectors are calculated.
    output_dir (str): Directory where output will be saved.

    Returns:
    ndarray: Berry curvature array.
    """
    curvature = np.zeros((len(theta_vals) - 1, eigenvectors.shape[2]))  # For each state
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the curvature
    with open(f'{output_dir}/curvature.out', "w") as log_file:
        log_file.write("#Theta Curvature\n")  # Header
        
        # Compute the Berry curvature for each state
        for i in range(1, len(theta_vals)):
            for j in range(eigenvectors.shape[2]):
                # Corrected Berry connection calculation
                A_n_i = np.imag(np.conj(eigenvectors[i, :, j]).T @ eigenvectors[i + 1, :, j])

                # Approximate the Berry curvature (simplified)
                # Use a finite difference approximation for the derivative
                dtheta = theta_vals[i+1] - theta_vals[i]
                curvature[i-1, j] = (A_n_i) / dtheta  # Finite difference approximation

            # Log the curvature for each state
            log_file.write(f"{theta_vals[i]:.6f} " + " ".join([f"{curvature[i-1, j]:.6f}" for j in range(eigenvectors.shape[2])]) + "\n")

    return curvature


def calculate_berry_phase_with_berry_curvature(theta_vals, eigenvectors, output_dir):
    berry_curvature = calculate_berry_curvature(eigenvectors, theta_vals, output_dir)

    berry_phases = np.zeros(eigenvectors.shape[2])
    accumulated_phases = np.zeros((eigenvectors.shape[2], len(theta_vals)))

    # Open a file to log accumulated phase values
    with open(f'{output_dir}/phase_log_berry_curvature.out', "w") as log_file:
        log_file.write("#Theta Accumulated_Phase\n")  # Header

        # Compute the Berry phase for each state
        for j in range(eigenvectors.shape[2]):
            berry_phase = np.zeros(len(theta_vals), dtype=complex)
            berry_phase[0] = 0  # Phase at the first point is set to 0
            
            # Compute the Berry phase by accumulating Berry curvature
            for i in range(1, len(theta_vals)):
                # Use trapezoidal rule to integrate Berry curvature
                berry_phase[i] = berry_phase[i-1] + np.trapezoid(berry_curvature[:i, j], theta_vals[:i])
                berry_phase[i] = (np.angle(berry_phase[i]) + np.pi) % (2 * np.pi) - np.pi  # Wrap phase between -π and π

            berry_phases[j] = berry_phase[-1].real  # Final Berry phase for each state
            accumulated_phases[j] = berry_phase.real  # Accumulated Berry phase for each point

        # Log the accumulated phase values for each state (average across all states)
        for i in range(len(theta_vals)):
            log_file.write(f"{theta_vals[i]:.15f} {np.mean(accumulated_phases[:, i]):.15f}\n")

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
plt.savefig(f'{output_dir}/eigenstate_overlaps.png')

def inspect_eigenvectors(eigenvectors, theta_vals, states):
    for state in states:
        print(f"State {state}:")
        for i, theta in enumerate(theta_vals):
            print(f"Theta = {theta:.2f}, Eigenvector: {eigenvectors[i, :, state]}")

#inspect_eigenvectors(eigenvectors, theta_vals, range(eigenvectors.shape[2]))


#plots
berry_phases, accumulated_phases = calculate_berry_phase_with_berry_curvature(theta_vals, eigenvectors, output_dir)

# Create output directory and detailed report
os.makedirs(output_dir, exist_ok=True)

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
    f.write('# Eigenstate Overlaps vs Theta\n')
    f.write('# Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
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
plt.savefig(f'{output_dir}/accumulated_phases.png')

#calculate and save the Hamiltonians, Va and Vx into .npy files
# Assuming you have defined the Hamiltonian function and potential functions
Hamiltonians = []
Va_values = []
Vx_values = []

for theta in theta_vals:
    H, R_theta_val, Vx, Va = hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)
    Hamiltonians.append(H)
    Va_values.append(Va)
    Vx_values.append(Vx)

# Convert lists to numpy arrays
Hamiltonians = np.array(Hamiltonians)
Va_values = np.array(Va_values)
Vx_values = np.array(Vx_values)

# Save the Hamiltonians, Va and Vx into .npy files
np.save(f'{output_dir}/Hamiltonians.npy', Hamiltonians)
np.save(f'{output_dir}/Va_values.npy', Va_values)
np.save(f'{output_dir}/Vx_values.npy', Vx_values)

#plot Va potential components
plt.figure(figsize=(12, 6))
Va_values = np.load(f'{output_dir}/Va_values.npy')
for i in range(3):
    plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i}]')
plt.xlabel('Theta (θ)')
plt.ylabel('Va Components')
plt.title('Va Components vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/Va_components.png')

#plot Vx potential components
plt.figure(figsize=(12, 6))
Vx_values = np.load(f'{output_dir}/Vx_values.npy')
for i in range(3):
    plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i}]')
plt.xlabel('Theta (θ)')
plt.ylabel('Vx Components')
plt.title('Vx Components vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/Vx_components.png')

# Initialize lists to store eigenvalues and eigenvectors
eigenvalues_list = []
eigenvectors_list = []

for theta in theta_vals:
    H, R_theta_val, Vx, Va = hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)
    
    # Calculate eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(H)  # Use np.linalg.eigh for Hermitian matrices
    eigenvalues_list.append(evals)
    eigenvectors_list.append(evecs)

# Convert lists to numpy arrays
eigenvalues_array = np.array(eigenvalues_list)
eigenvectors_array = np.array(eigenvectors_list)

# Save the eigenvalues and eigenvectors into .npy files
np.save(f'{output_dir}/eigenvalues.npy', eigenvalues_array)
np.save(f'{output_dir}/eigenvectors.npy', eigenvectors_array)

# Plot the eigenvalues
plt.figure(figsize=(12, 6))

# Transpose eigenvalues_array for correct plotting
for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
    plt.plot(theta_vals, eigenvalues_array[:, i], label=f'Eigenvalue {i+1}')

plt.xlabel('Theta (θ)')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs Theta')
plt.grid(True)
plt.legend()  # Add legend to identify each eigenvalue
plt.tight_layout()
plt.savefig(f'{output_dir}/eigenvalues.png')

## Check for degeneracy
tolerance = 1e-3  # Define a tolerance level for degeneracy
small_difference_threshold = 0.1  # Define a threshold for small differences
degeneracy_list = []
near_degeneracy_list = []
difference_list = []
report = []

for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
    for j in range(i + 1, eigenvalues_array.shape[1]):
        difference = np.abs(eigenvalues_array[:, i] - eigenvalues_array[:, j])
        difference_list.append(difference)
        if np.all(difference < tolerance):
            degeneracy_list.append((i, j))  # Store the indices of degenerate states
        elif np.all(difference < small_difference_threshold):
            near_degeneracy_list.append((i, j))  # Store the indices of near-degenerate states

# Log the degeneracy results
with open(f'{output_dir}/degeneracy_check.out', 'w') as log_file:
    log_file.write("# Degeneracy Check\n")
    log_file.write("======================================\n\n")
    log_file.write(f"Parameters:\n")
    log_file.write(f"Tolerance level: {tolerance}\n")
    log_file.write(f"Small difference threshold: {small_difference_threshold}\n\n")
    log_file.write("======================================\n\n")
    if not degeneracy_list and not near_degeneracy_list:
        log_file.write("No degeneracies or near degeneracies found.\n")
    if not degeneracy_list and near_degeneracy_list:
        log_file.write("No degenerate eigenstates found.\n")
        for state1, state2 in near_degeneracy_list:
            log_file.write(f"Eigenstates {state1} and {state2} are near degenerate.\n")
    if not near_degeneracy_list and degeneracy_list:
        log_file.write("No near-degenerate eigenstates found.\n")
        for state1, state2 in degeneracy_list:
            log_file.write(f"Eigenstates {state1} and {state2} are degenerate.\n")
    log_file.write('\n')
    # Log the differences between each eigenstate
    log_file.write('======================================\nDifferences between eigenstates:\n')
    for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
        for j in range(i + 1, eigenvalues_array.shape[1]):
            difference = np.abs(eigenvalues_array[:, i] - eigenvalues_array[:, j])
            log_file.write(f"Eigenstates {i} and {j}: {difference}\n")  # Convert to string
    
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
    f.write(f'\nDegenerate Eigenstates:\n')
    if degeneracy_list:
        for state1, state2 in degeneracy_list:
            f.write(f'  Eigenstates {state1} and {state2} are degenerate.\n')
    else:
        f.write('  No degeneracies found.\n')
    f.write('Detailed degeneracy check logged in degeneracy_check.out\n')
    
    f.write('\n')
    f.write('Berry curvature logged in phase_log_berry_curvature.out\n')
    f.write('\n')
    f.write('Eigenvalue plot saved as eigenvalues.png\n')

    f.write('Eigenvector differences logged in eigenvector_diff.out\n')
    f.write('\nEigenvalues and Eigenvectors:\n')
    f.write('To load the eigenvalues and eigenvectors, use:\n')
    f.write('eigenvalues = np.load(f"{output_dir}/eigenvalues.npy")\n')
    f.write('eigenvectors = np.load(f"{output_dir}/eigenvectors.npy")\n')
    
    f.write('\nFor Hamiltonians, Va, Vx:\n')
    f.write('Use np.load() to load the data as a numpy array. Example usage:\n')
    f.write('H = np.load(f"{output_dir}/Hamiltonians.npy")\n')
    f.write('Va = np.load(f"{output_dir}/Va_values.npy")\n')
    f.write('Vx = np.load(f"{output_dir}/Vx_values.npy")\n')
    
    if report:
        f.write('\nNear Degenerate Eigenstates:\n')
        for line in report:
            f.write(line + '\n')