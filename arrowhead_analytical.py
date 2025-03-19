import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define physical constants
hbar = 1.0  # Simplified value for demonstration

# Function to create R_theta vector orthogonal to x=y=z plane
def R_theta(d, theta):
    # Define the direction of the line x=y=z (the axis of rotation)
    axis = np.array([1, 1, 1])
    
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    
    # Rotation matrix for rotating around the axis [1, 1, 1] by angle theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    
    # Rotation matrix to rotate around axis [1, 1, 1]
    rotation_matrix = np.array([
        [cos_theta + axis[0]**2 * one_minus_cos, axis[0] * axis[1] * one_minus_cos - axis[2] * sin_theta, axis[0] * axis[2] * one_minus_cos + axis[1] * sin_theta],
        [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_theta, cos_theta + axis[1]**2 * one_minus_cos, axis[1] * axis[2] * one_minus_cos - axis[0] * sin_theta],
        [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_theta, axis[2] * axis[1] * one_minus_cos + axis[0] * sin_theta, cos_theta + axis[2]**2 * one_minus_cos]
    ])
    
    # The original vector pointing along the x, y, and z axes
    original_vector = np.array([d, 0, 0])  # We start from (d, 0, 0) along the x-axis
    
    # Apply the rotation to the original vector to get the point on the circle
    R_theta_vector = np.dot(rotation_matrix, original_vector)
    
    return R_theta_vector

# Define the potential functions V_x and V_a based on R_theta
def V_x(R_theta, a, b, c):
    # Calculate individual V_x components for each R_theta component
    Vx0 = a * R_theta[0] + b * R_theta[0] + c
    Vx1 = a * R_theta[1] + b * R_theta[1] + c
    Vx2 = a * R_theta[2] + b * R_theta[2] + c
    return [Vx0, Vx1, Vx2]

def V_a(R_theta, a, b, c, x_shift, y_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va0 = a * (R_theta[0] - x_shift) + b * (R_theta[0] - y_shift) + c
    Va1 = a * (R_theta[1] - x_shift) + b * (R_theta[1] - y_shift) + c
    Va2 = a * (R_theta[2] - x_shift) + b * (R_theta[2] - y_shift) + c
    return [Va0, Va1, Va2]

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, r1, r2, r3, omega, a, b, c, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, a, b, c)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, a, b, c, x_shift, y_shift)  # [Va0, Va1, Va2]
    
    # Create a 4x4 Hamiltonian with an arrowhead structure
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    H[1, 1] = Va[0] + Vx[1] + Vx[2]
    H[2, 2] = Vx[0] + Va[1] + Vx[2]
    H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    
    # Coupling between states 0 and 1 with theta dependence
    H[0, 1] = r1 * np.exp(1j * theta)
    H[1, 0] = r1 * np.exp(-1j * theta)
    
    # Coupling between states 0 and 2 with theta dependence
    H[0, 2] = r2 * np.exp(-1j * theta)
    H[2, 0] = r2 * np.exp(1j * theta)
    
    # Coupling between states 0 and 3 (constant)
    H[0, 3] = H[3, 0] = r3
    
    return H

# Calculate the Berry connection analytically
def berry_connection_analytical(theta_vals, r1, r2):
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
    
    # For state 0 (ground state), the Berry connection is approximately:
    A[0, :] = 0.5 * (r1**2 - r2**2) / (r1**2 + r2**2)
    
    # For state 1, the Berry connection is approximately:
    A[1, :] = -0.5 * r1**2 / (r1**2 + r2**2)
    
    # For state 2, the Berry connection is approximately:
    A[2, :] = 0.5 * r2**2 / (r1**2 + r2**2)
    
    # For state 3, the Berry connection is approximately:
    A[3, :] = 0
    
    return A

# Calculate the Berry phase by integrating the Berry connection
def berry_phase(A, theta_vals):
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

# Parameters
r1 = 0.5  # Coupling between states 0 and 1
r2 = 0.7  # Coupling between states 0 and 2
r3 = 0.3  # Coupling between states 0 and 3
omega = 1.0  # Frequency parameter

# Coefficients for the potential functions
a = 1.0  # First coefficient for potentials
b = 0.5  # Second coefficient for potentials
c = 0.0  # Constant term in potentials

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

# Loop over theta values to compute the eigenvalues and eigenstates
for theta in theta_vals:
    # Calculate R_theta vector for this theta
    r_theta_vector = R_theta(d, theta)
    r_theta_vectors.append(r_theta_vector)
    
    # Calculate the Hamiltonian for this theta
    H = hamiltonian(theta, r1, r2, r3, omega, a, b, c, x_shift, y_shift, d)
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    
    eigenvalues.append(eigvals)
    eigenstates.append(eigvecs)

# Convert to numpy arrays for easier manipulation
eigenvalues = np.array(eigenvalues)
eigenstates = np.array(eigenstates)
r_theta_vectors = np.array(r_theta_vectors)

# Calculate the Berry connection analytically
A_analytical = berry_connection_analytical(theta_vals, r1, r2)

# Calculate the Berry phase by integrating the analytical Berry connection
berry_phases = berry_phase(A_analytical, theta_vals)

# Print the Berry phase for each state
for i, phase in enumerate(berry_phases):
    print(f"Berry Phase for state {i}: {phase}")

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

# Plot the analytical Berry connection
plt.figure(figsize=(10, 6))
for i in range(A_analytical.shape[0]):
    plt.plot(theta_vals, np.real(A_analytical[i, :]), label=f'State {i}')

plt.xlabel('Theta (θ)')
plt.ylabel('Berry Connection')
plt.title('Analytical Berry Connection vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()

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

# Show the plots
plt.show()
