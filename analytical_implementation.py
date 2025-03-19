import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a simple two-level system with a known Berry phase
def hamiltonian(theta, phi):
    """
    Create a 2x2 Hamiltonian for a spin-1/2 particle in a magnetic field.
    
    H = B·σ = B_x σ_x + B_y σ_y + B_z σ_z
    
    where B = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    and σ_x, σ_y, σ_z are the Pauli matrices.
    """
    # Magnetic field components in spherical coordinates
    B_x = np.sin(theta) * np.cos(phi)
    B_y = np.sin(theta) * np.sin(phi)
    B_z = np.cos(theta)
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Hamiltonian
    H = B_x * sigma_x + B_y * sigma_y + B_z * sigma_z
    
    return H, np.array([B_x, B_y, B_z])

# Calculate the Berry connection analytically
def berry_connection_analytical(theta, phi_vals):
    """
    Calculate the Berry connection for a two-level system analytically.
    
    For a spin-1/2 system with the ground state |ψ⟩, the Berry connection is:
    A_φ = -i⟨ψ|∂ψ/∂φ⟩
    
    For a magnetic field tracing a circle at fixed theta, the Berry connection is:
    A_φ = (1/2) * (1 - cos(theta)) for the ground state
    A_φ = -(1/2) * (1 - cos(theta)) for the excited state
    """
    # Berry connection for the ground state
    A_ground = 0.5 * (1 - np.cos(theta)) * np.ones_like(phi_vals)
    
    # Berry connection for the excited state
    A_excited = -0.5 * (1 - np.cos(theta)) * np.ones_like(phi_vals)
    
    return np.array([A_ground, A_excited])

# Calculate the Berry phase analytically
def berry_phase_analytical(theta):
    """
    Calculate the Berry phase analytically for a spin-1/2 system.
    
    For a magnetic field tracing a circle at fixed theta, the Berry phase is:
    gamma = -π * (1 - cos(theta)) for the ground state
    gamma = π * (1 - cos(theta)) for the excited state
    """
    # Berry phase for the ground state
    gamma_ground = -np.pi * (1 - np.cos(theta))
    
    # Berry phase for the excited state
    gamma_excited = np.pi * (1 - np.cos(theta))
    
    return np.array([gamma_ground, gamma_excited])

# Parameters
theta = np.pi/2  # Fixed polar angle (90 degrees)
num_points = 1000
phi_vals = np.linspace(0, 2*np.pi, num_points)

# Initialize arrays for storing eigenvalues, eigenstates, and magnetic field vectors
eigenvalues = []
eigenstates = []
B_vectors = []

# Loop over phi values to compute the eigenvalues and eigenstates
for phi in phi_vals:
    # Calculate the Hamiltonian and magnetic field for this phi
    H, B = hamiltonian(theta, phi)
    B_vectors.append(B)
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    
    eigenvalues.append(eigvals)
    eigenstates.append(eigvecs)

# Convert to numpy arrays for easier manipulation
eigenvalues = np.array(eigenvalues)
eigenstates = np.array(eigenstates)
B_vectors = np.array(B_vectors)

# Calculate the Berry connection analytically
A_analytical = berry_connection_analytical(theta, phi_vals)

# Calculate the Berry phase analytically
berry_phases_analytical = berry_phase_analytical(theta)

# Calculate the Berry phase by integrating the analytical Berry connection
berry_phases_integrated = np.zeros(2)
for n in range(2):
    # Numerical integration of the analytical Berry connection
    berry_phases_integrated[n] = np.trapezoid(A_analytical[n, :], phi_vals)

# Print the Berry phase for each state
for i in range(2):
    print(f"Analytical Berry Phase for state {i}: {berry_phases_analytical[i]}")
    print(f"Integrated Berry Phase for state {i}: {berry_phases_integrated[i]}")
    print(f"Difference: {berry_phases_integrated[i] - berry_phases_analytical[i]}")
    print()

# Plot eigenvalues to visualize the evolution
plt.figure(figsize=(10, 6))
for i in range(eigenvalues.shape[1]):
    plt.plot(phi_vals, eigenvalues[:, i], label=f'State {i}')

plt.xlabel('Phi (φ)')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs Phi')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot the analytical Berry connection
plt.figure(figsize=(10, 6))
for i in range(A_analytical.shape[0]):
    plt.plot(phi_vals, A_analytical[i, :], label=f'State {i}')

plt.xlabel('Phi (φ)')
plt.ylabel('Berry Connection')
plt.title('Analytical Berry Connection vs Phi')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot the magnetic field vectors in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the path traced by the magnetic field
ax.plot(B_vectors[:, 0], B_vectors[:, 1], B_vectors[:, 2], 'b-', label='B field path')

# Add markers for a few points to show the direction of the path
marker_indices = np.linspace(0, len(B_vectors)-1, 10, dtype=int)
ax.scatter(B_vectors[marker_indices, 0], B_vectors[marker_indices, 1], B_vectors[marker_indices, 2], 
           c='r', s=50, label='Markers')

# Plot the origin
ax.scatter([0], [0], [0], c='k', s=100, label='Origin')

# Set labels and title
ax.set_xlabel('Bx')
ax.set_ylabel('By')
ax.set_zlabel('Bz')
ax.set_title('Magnetic Field Vector in 3D Space')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend()

# Show the plots
plt.show()
