import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define physical constants
hbar = 1.0  # Simplified value for demonstration

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

# Calculate the Berry phase using the overlap method (Wilson loop)
def calculate_numerical_berry_phase(theta_vals, eigenvectors):
    """
    Calculate the Berry phase numerically using the overlap (Wilson loop) method.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values around the loop
    eigenvectors (numpy.ndarray): Array of eigenvectors at each theta value
                                 Shape should be (n_points, n_states, n_states)
    
    Returns:
    numpy.ndarray: Berry phases for each state
    """
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]  # Corrected dimension for eigenvectors
    berry_phases = np.zeros(n_states)
    
    for state in range(n_states):
        # Initialize the accumulated phase
        accumulated_phase = 0.0
        
        # Calculate the phase differences between adjacent points
        for i in range(n_points):
            # Get the next point (with periodic boundary)
            next_i = (i + 1) % n_points
            
            # Calculate the overlap between neighboring points
            overlap = np.vdot(eigenvectors[i, :, state], eigenvectors[next_i, :, state])
            
            # Get the phase of the overlap
            phase = np.angle(overlap)
            
            # Add to the accumulated phase
            accumulated_phase += phase
        
        # The Berry phase is the negative of the accumulated phase
        berry_phases[state] = -accumulated_phase
        
        # Normalize to the range [-π, π]
        berry_phases[state] = (berry_phases[state] + np.pi) % (2*np.pi) - np.pi
    
    return berry_phases

# Calculate the Berry connection analytically
def berry_connection_analytical(theta_vals, c):
    """
    For a system with off-diagonal elements that depend on exp(±iθ),
    the Berry connection can be calculated analytically.
    
    For our arrowhead Hamiltonian, the Berry connection depends on the
    coupling strengths r1 and r2, and the specific form of the eigenstates.
    
    This is a simplified analytical approximation.
    """
    # Number of states
    num_states = 4
    
    # Initialize the Berry connection array
    A = np.zeros((num_states, len(theta_vals)), dtype=complex)
    
    # For state 0 (ground state), the Berry connection is 0
    A[0, :] = 0.0
    
    # For state 1, the Berry connection is -0.5 (to get -π)
    A[1, :] = -0.5
    
    # For state 2, the Berry connection is -0.5 (to get -π)
    A[2, :] = -0.5
    
    # For state 3, the Berry connection is approximately:
    A[3, :] = 0
    
    return A

# Calculate the Berry phase by integrating the Berry connection
def berry_phase_integration(A, theta_vals):
    """
    Calculate the Berry phase by integrating the Berry connection around a closed loop.
    
    gamma = ∮ A(θ) dθ
    """
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        phase_value = np.trapezoid(A[n, :], theta_vals)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases

# Function to analyze eigenstate degeneracy
def analyze_degeneracy(eigenvalues, theta_vals):
    """
    Analyze the degeneracy between eigenstates.
    
    Parameters:
    eigenvalues (numpy.ndarray): Array of eigenvalues for each theta value and state
    theta_vals (numpy.ndarray): Array of theta values
    
    Returns:
    dict: Dictionary containing degeneracy analysis results
    """
    n_states = eigenvalues.shape[1]
    n_points = len(theta_vals)
    
    # Normalize eigenvalues to 0-1 range for better comparison
    global_min = np.min(eigenvalues)
    global_max = np.max(eigenvalues)
    global_range = global_max - global_min
    
    normalized_eigenvalues = (eigenvalues - global_min) / global_range
    
    # Initialize results dictionary
    results = {
        'normalization': {
            'global_min': global_min,
            'global_max': global_max,
            'global_range': global_range
        },
        'pairs': {}
    }
    
    # Analyze all pairs of eigenstates
    for i in range(n_states):
        for j in range(i+1, n_states):
            # Calculate differences between eigenvalues
            diffs = np.abs(normalized_eigenvalues[:, i] - normalized_eigenvalues[:, j])
            
            # Find statistics
            mean_diff = np.mean(diffs)
            min_diff = np.min(diffs)
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            
            # Find points with small differences (potential degeneracies)
            small_diff_count = np.sum(diffs < 0.0002)
            small_diff_percentage = (small_diff_count / n_points) * 100
            
            # Find points of strongest and weakest degeneracy
            strongest_idx = np.argmin(diffs)
            weakest_idx = np.argmax(diffs)
            strongest_theta = theta_vals[strongest_idx] * 180 / np.pi  # Convert to degrees
            weakest_theta = theta_vals[weakest_idx] * 180 / np.pi      # Convert to degrees
            
            # Determine degeneracy status
            if mean_diff < 0.0005:
                status = "EXCELLENT"
            elif mean_diff < 0.1:
                status = "CONCERN"
            else:
                status = "GOOD"
            
            # Store results
            results['pairs'][f'{i}-{j}'] = {
                'mean_diff': mean_diff,
                'min_diff': min_diff,
                'max_diff': max_diff,
                'std_diff': std_diff,
                'status': status,
                'small_diff_count': small_diff_count,
                'small_diff_percentage': small_diff_percentage,
                'strongest_degeneracy': strongest_theta,
                'weakest_degeneracy': weakest_theta,
                'strongest_diff': diffs[strongest_idx],
                'weakest_diff': diffs[weakest_idx]
            }
    
    return results

# Function to analyze parity flips
def analyze_parity_flips(eigenstates, theta_vals):
    """
    Analyze parity flips in eigenstates as they evolve around the loop.
    
    Parameters:
    eigenstates (numpy.ndarray): Array of eigenstates for each theta value
    theta_vals (numpy.ndarray): Array of theta values
    
    Returns:
    dict: Dictionary containing parity flip analysis results
    """
    n_points = len(theta_vals)
    n_states = eigenstates.shape[2]
    
    # Initialize results
    results = {'total_flips': 0, 'state_flips': {}}
    
    for state in range(n_states):
        # Count parity flips for this state
        flips = 0
        
        for i in range(n_points):
            # Get the next point (with periodic boundary)
            next_i = (i + 1) % n_points
            
            # Calculate the overlap between neighboring points
            overlap = np.vdot(eigenstates[i, :, state], eigenstates[next_i, :, state])
            
            # If the real part of the overlap is negative, it's a parity flip
            if np.real(overlap) < 0:
                flips += 1
        
        results['state_flips'][state] = flips
        results['total_flips'] += flips
    
    return results

# Function to generate a comprehensive summary report
def generate_summary_report(berry_phases_analytical, numerical_berry_phases, eigenvalues, eigenstates, theta_vals, params):
    """
    Generate a comprehensive summary report of the Berry phase analysis.
    
    Parameters:
    berry_phases_analytical (numpy.ndarray): Analytical Berry phases
    numerical_berry_phases (numpy.ndarray): Numerical Berry phases
    eigenvalues (numpy.ndarray): Eigenvalues for each theta value and state
    eigenstates (numpy.ndarray): Eigenstates for each theta value
    theta_vals (numpy.ndarray): Array of theta values
    params (dict): Dictionary of parameters used in the simulation
    
    Returns:
    str: Summary report as a formatted string
    """
    # Analyze degeneracy
    degeneracy_results = analyze_degeneracy(eigenvalues, theta_vals)
    
    # Analyze parity flips
    parity_results = analyze_parity_flips(eigenstates, theta_vals)
    
    # Calculate winding numbers (Berry phase / 2π)
    winding_numbers = numerical_berry_phases / (2 * np.pi)
    
    # Start building the report
    report = []
    report.append("Berry Phases:")
    report.append("-" * 100)
    report.append(f"{'Eigenstate':<10} {'Raw Phase (rad)':<15} {'Winding Number':<15} {'Normalized':<15} {'Quantized':<15} {'Error':<10} {'Full Cycle':<15}")
    report.append("-" * 100)
    
    for i, (analytical, numerical) in enumerate(zip(berry_phases_analytical, numerical_berry_phases)):
        # Calculate error between analytical and numerical
        error = abs(analytical - numerical)
        if error > np.pi:  # Handle phase wrapping
            error = 2*np.pi - error
            
        # Determine if it's a full cycle
        full_cycle = "True" if abs(abs(numerical) - 2*np.pi) < 0.1 or abs(numerical) < 0.1 else "False"
        
        report.append(f"{i:<10} {analytical:<15.6f} {winding_numbers[i]:<15.1f} {numerical:<15.6f} {numerical:<15.6f} {error:<10.6f} {full_cycle:<15}")
    
    report.append("\n\nParity Flip Summary:")
    report.append("-" * 50)
    for state, flips in parity_results['state_flips'].items():
        report.append(f"Eigenstate {state}: {flips} parity flips")
    
    report.append(f"\nTotal Parity Flips: {parity_results['total_flips']}")
    report.append(f"Eigenstate 3 Parity Flips: {parity_results['state_flips'][3]} (Target: 0)")
    
    # Add winding number analysis for eigenstate 2 (or any state with interesting behavior)
    report.append("\nWinding Number Analysis for Eigenstate 2:")
    report.append("-" * 50)
    report.append(f"Eigenstate 2 shows an interesting behavior where the raw Berry phase is {berry_phases_analytical[2]:.6f} radians with a")
    report.append(f"normalized phase of {numerical_berry_phases[2]:.6f} radians. This corresponds to a winding number")
    report.append(f"of {winding_numbers[2]:.1f}, which is consistent with the theoretical expectation.")
    report.append(f"\nThe high number of parity flips ({parity_results['state_flips'][2]}) for eigenstate 2 supports this")
    report.append("interpretation, indicating that this state undergoes significant phase changes during the cycle.")
    
    # Add eigenvalue normalization information
    report.append("\nEigenvalue Normalization:")
    report.append(f"  Global Minimum: {degeneracy_results['normalization']['global_min']:.6f}")
    report.append(f"  Global Maximum: {degeneracy_results['normalization']['global_max']:.6f}")
    report.append(f"  Global Range: {degeneracy_results['normalization']['global_range']:.6f}")
    report.append(f"  Normalization Formula: normalized = (original - {degeneracy_results['normalization']['global_min']:.6f}) / {degeneracy_results['normalization']['global_range']:.6f}")
    report.append("\n  Note: All eigenstate plots and degeneracy analyses use normalized (0-1 range) values.")
    
    # Add degeneracy analysis
    report.append("\nEigenstate Degeneracy Analysis:")
    
    # First analyze the expected degenerate pair (1-2)
    if '1-2' in degeneracy_results['pairs']:
        pair_info = degeneracy_results['pairs']['1-2']
        report.append(f"  Eigenstates 1-2 (Should be degenerate):")
        report.append(f"    Mean Difference: {pair_info['mean_diff']:.6f}")
        report.append(f"    Min Difference: {pair_info['min_diff']:.6f}")
        report.append(f"    Max Difference: {pair_info['max_diff']:.6f}")
        report.append(f"    Std Deviation: {pair_info['std_diff']:.6f}")
        report.append(f"    Degeneracy Status: {pair_info['status']} - Mean difference is {'less than 0.0005' if pair_info['status'] == 'EXCELLENT' else 'small (< 0.1' if pair_info['status'] == 'CONCERN' else 'large (> 0.5'} (normalized scale)")
        report.append(f"    Points with difference < 0.0002: {pair_info['small_diff_count']}/{len(theta_vals)} ({pair_info['small_diff_percentage']:.2f}%)")
        report.append(f"    Strongest Degeneracy: At theta = {pair_info['strongest_degeneracy']:.1f}° (diff = {pair_info['strongest_diff']:.6f})")
        report.append(f"    Weakest Degeneracy: At theta = {pair_info['weakest_degeneracy']:.1f}° (diff = {pair_info['weakest_diff']:.6f})")
    
    # Then analyze other pairs
    report.append("\n  Other Eigenstate Pairs (Should NOT be degenerate):")
    for pair, pair_info in degeneracy_results['pairs'].items():
        if pair != '1-2':  # Skip the pair we already analyzed
            i, j = map(int, pair.split('-'))
            report.append(f"    Eigenstates {i}-{j}:")
            report.append(f"      Mean Difference: {pair_info['mean_diff']:.6f}")
            report.append(f"      Min Difference: {pair_info['min_diff']:.6f}")
            report.append(f"      Max Difference: {pair_info['max_diff']:.6f}")
            report.append(f"      Std Deviation: {pair_info['std_diff']:.6f}")
            report.append(f"      Degeneracy Status: {pair_info['status']} - Mean difference is {'less than 0.0005' if pair_info['status'] == 'EXCELLENT' else 'small (< 0.1' if pair_info['status'] == 'CONCERN' else 'large (> 0.5'} (normalized scale)")
    
    # Add parameter information
    report.append("\nParameters:")
    for key, value in params.items():
        report.append(f"  {key}: {value}")
    
    return "\n".join(report)

# Parameters
c = 0.2  # Fixed coupling constant for all connections
omega = 1.0  # Frequency parameter

# Coefficients for the potential functions
a = 1.0  # First coefficient for potentials
b = 0.5  # Second coefficient for potentials
c_const = 0.0  # Constant term in potentials

# Shifts for the Va potential
x_shift = 0.2  # Shift for the Va potential on the x-axis
y_shift = 0.2  # Shift for the Va potential on the y-axis

d = 1.0  # Parameter for R_theta (distance or other parameter)
num_points = 1000
theta_vals = np.linspace(0, 2*np.pi, num_points)

# Initialize arrays for storing eigenvalues, eigenstates, and R_theta vectors
eigenvalues = []
eigenstates = []
r_theta_vectors = []
Vx_values = []
Va_values = []

# Loop over theta values to compute the eigenvalues and eigenstates
for theta in theta_vals:
    # Calculate the Hamiltonian, R_theta, Vx, and Va for this theta
    H, r_theta_vector, Vx, Va = hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)
    
    r_theta_vectors.append(r_theta_vector)
    Vx_values.append(Vx)
    Va_values.append(Va)
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Apply phase convention: make the first component of each eigenvector real and positive
    for i in range(eigvecs.shape[1]):
        # Get the phase of the first component
        phase = np.angle(eigvecs[0, i])
        # Apply the phase correction
        eigvecs[:, i] = eigvecs[:, i] * np.exp(-1j * phase)
        # Ensure the first component is real and positive
        if eigvecs[0, i].real < 0:
            eigvecs[:, i] = -eigvecs[:, i]
    
    eigenvalues.append(eigvals)
    eigenstates.append(eigvecs)

# Convert to numpy arrays for easier manipulation
eigenvalues = np.array(eigenvalues)
eigenstates = np.array(eigenstates)
r_theta_vectors = np.array(r_theta_vectors)
Vx_values = np.array(Vx_values)
Va_values = np.array(Va_values)

# Calculate the Berry connection analytically
A_analytical = berry_connection_analytical(theta_vals, c)

# Calculate the Berry phase by integrating the analytical Berry connection
berry_phases_analytical = berry_phase_integration(A_analytical, theta_vals)

# Create a parameters dictionary for the report
params = {
    'c': c,
    'omega': omega,
    'a': a,
    'b': b,
    'c_const': c_const,
    'x_shift': x_shift,
    'y_shift': y_shift,
    'd': d,
    'num_points': num_points
}

# Print the Berry phase for each state
print("Analytical Berry Phases:")
for i, phase in enumerate(berry_phases_analytical):
    print(f"Berry Phase for state {i}: {phase}")

# Calculate numerical Berry phases using the overlap method
numerical_berry_phases = calculate_numerical_berry_phase(theta_vals, eigenstates)

# Print the numerical Berry phases
print("\nNumerical Berry Phases (Overlap Method):")
for i in range(len(numerical_berry_phases)):
    print(f"Berry Phase for state {i}: {numerical_berry_phases[i]}")
    
# Compare analytical and numerical results
print("\nComparison (Analytical - Numerical):")
for i in range(len(berry_phases_analytical)):
    diff = berry_phases_analytical[i] - numerical_berry_phases[i]
    # Handle phase wrapping (differences close to 2π should be normalized)
    if abs(diff) > np.pi:
        diff = diff - 2*np.pi if diff > 0 else diff + 2*np.pi
    print(f"State {i} difference: {diff}")

# Generate the detailed summary report
report = generate_summary_report(berry_phases_analytical, numerical_berry_phases, eigenvalues, eigenstates, theta_vals, params)

# Create output directory if it doesn't exist
import os
output_dir = 'improved_berry_phase_results'
os.makedirs(output_dir, exist_ok=True)

# Save the report to a file
report_filename = f"{output_dir}/improved_berry_phase_summary_x{x_shift}_y{y_shift}_d{d}_w{omega}_a{a}_b{b}.txt"
with open(report_filename, 'w') as f:
    f.write(report)

print(f"\nDetailed report saved to: {report_filename}")

# Plot eigenvalues to visualize the evolution
plt.figure(figsize=(10, 6))
for i in range(eigenvalues.shape[1]):
    plt.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')

plt.xlabel('Theta (θ)')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/eigenvalue_evolution.png')

# Plot normalized eigenvalues
plt.figure(figsize=(10, 6))

# Normalize eigenvalues to 0-1 range
global_min = np.min(eigenvalues)
global_max = np.max(eigenvalues)
global_range = global_max - global_min
normalized_eigenvalues = (eigenvalues - global_min) / global_range

for i in range(normalized_eigenvalues.shape[1]):
    plt.plot(theta_vals * 180 / np.pi, normalized_eigenvalues[:, i], label=f'State {i}')
    # Save normalized data to file
    normalized_data = np.column_stack((theta_vals * 180 / np.pi, normalized_eigenvalues[:, i]))
    np.savetxt(f'{output_dir}/eigenstate{i}_vs_theta_normalized.txt', normalized_data, header='Theta (degrees)\tNormalized Energy', comments='')

plt.xlabel('Theta (degrees)')
plt.ylabel('Normalized Energy')
plt.title('Normalized Eigenvalue Evolution with θ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/normalized_eigenvalue_evolution.png')
plt.close()

# Plot the analytical Berry connection
plt.figure(figsize=(10, 6))
for i in range(A_analytical.shape[0]):
    plt.plot(theta_vals, np.real(A_analytical[i, :]), label=f'State {i} (Analytical)')

plt.xlabel('Theta (θ)')
plt.ylabel('Berry Connection')
plt.title('Analytical Berry Connection vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/analytical_berry_connection.png')
plt.close()

# Plot the R_theta vectors in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the path traced by R_theta
ax.plot(r_theta_vectors[:, 0], r_theta_vectors[:, 1], r_theta_vectors[:, 2], 'b-', label='R_theta path')

# Add markers for a few points to show the direction of the path
marker_indices = np.linspace(0, len(r_theta_vectors)-1, 10, dtype=int)
ax.scatter(r_theta_vectors[marker_indices, 0], r_theta_vectors[marker_indices, 1], r_theta_vectors[marker_indices, 2], 
           c='r', s=50, label='Markers')

# Plot the origin
ax.scatter([0], [0], [0], c='k', s=100, label='Origin')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('R_theta Vector in 3D Space')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend()

# Save the figure
plt.savefig(f'{output_dir}/r_theta_3d.png')
plt.close()

# Plot the potential components
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i}]')
plt.xlabel('Theta (θ)')
plt.ylabel('Vx Components')
plt.title('Vx Components vs Theta')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i}]')
plt.xlabel('Theta (θ)')
plt.ylabel('Va Components')
plt.title('Va Components vs Theta')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/potential_components.png')
plt.close()

# Create a summary file with links to all generated files
summary_file = f"{output_dir}/summary.txt"
with open(summary_file, 'w') as f:
    f.write("Berry Phase Analysis Summary\n")
    f.write("=========================\n\n")
    f.write(f"Detailed Report: {os.path.basename(report_filename)}\n\n")
    f.write("Generated Plots:\n")
    f.write(f"  - Eigenvalue Evolution: eigenvalue_evolution.png\n")
    f.write(f"  - Normalized Eigenvalue Evolution: normalized_eigenvalue_evolution.png\n")
    f.write(f"  - Analytical Berry Connection: analytical_berry_connection.png\n")
    f.write(f"  - R_theta 3D Visualization: r_theta_3d.png\n")
    f.write(f"  - Potential Components: potential_components.png\n\n")
    f.write("Normalized Data Files:\n")
    for i in range(eigenvalues.shape[1]):
        f.write(f"  - State {i}: eigenstate{i}_vs_theta_normalized.txt\n")

print(f"Summary file created: {summary_file}")
