import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define physical constants
hbar = 1.0  # Simplified value for demonstration

# Define the Hamiltonian for a two-level system with a rotating magnetic field
def hamiltonian(theta, phi, B0):
    # B0 is the magnitude of the magnetic field
    # theta and phi are the spherical angles defining the direction of the field
    
    # Magnetic field components in spherical coordinates
    Bx = B0 * np.sin(theta) * np.cos(phi)
    By = B0 * np.sin(theta) * np.sin(phi)
    Bz = B0 * np.cos(theta)
    
    # Pauli matrices dotted with the B field
    H = np.array([
        [Bz, Bx - 1j*By],
        [Bx + 1j*By, -Bz]
    ])
    
    return H

# Calculate the Berry connection for a two-level system
def berry_connection(eigenstates, phi):
    # eigenstates shape is (num_phi_points, num_dimensions, num_states)
    num_phi_points, num_dimensions, num_states = eigenstates.shape
    A = np.zeros((num_states, num_phi_points), dtype=complex)
    
    for n in range(num_states):
        for p in range(num_phi_points):
            # Extract the eigenstate at phi point p for state n
            psi_n = eigenstates[p, :, n]
            
            # Calculate the next phi point (with periodic boundary)
            p_next = (p + 1) % num_phi_points
            psi_n_next = eigenstates[p_next, :, n]
            
            # Ensure proper phase relationship between adjacent eigenstates
            overlap = np.vdot(psi_n, psi_n_next)
            phase = np.angle(overlap)
            
            # Adjust the phase of the next state to ensure smooth evolution
            if abs(overlap) > 1e-10:  # Only adjust if there's significant overlap
                psi_n_next = psi_n_next * np.exp(-1j * phase)
            
            # Simple finite difference for derivative
            d_phi = phi[p_next] - phi[p] if p_next > p else phi[p_next] + 2*np.pi - phi[p]
            # Avoid division by zero
            if abs(d_phi) < 1e-10:
                d_psi_n_dphi = np.zeros_like(psi_n)
            else:
                d_psi_n_dphi = (psi_n_next - psi_n) / d_phi
            
            # Berry connection: A = -i⟨ψ|∂ψ/∂φ⟩
            A[n, p] = -1j * np.vdot(psi_n, d_psi_n_dphi)
    
    return A

# Calculate the Berry phase
def berry_phase(A, phi):
    # A has shape (num_states, num_phi_points)
    # We integrate over phi for each state
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        # The Berry phase is the integral of the Berry connection around a closed loop
        phase_value = np.trapezoid(A[n, :], phi)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases

# Parameters
B0 = 1.0  # Magnitude of the magnetic field
theta = np.pi/3  # Fixed polar angle (from z-axis)
phi_vals = np.linspace(0, 2*np.pi, 1000)  # Azimuthal angle (around z-axis)

# Initialize arrays for storing eigenvalues and eigenstates
eigenvalues = []
eigenstates = []
magnetic_field_vectors = []

# Loop over phi values to compute the eigenvalues and eigenstates
for phi in phi_vals:
    # Calculate the Hamiltonian
    H = hamiltonian(theta, phi, B0)
    
    # Calculate the magnetic field vector
    Bx = B0 * np.sin(theta) * np.cos(phi)
    By = B0 * np.sin(theta) * np.sin(phi)
    Bz = B0 * np.cos(theta)
    magnetic_field_vectors.append([Bx, By, Bz])
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Apply phase convention: make the first component of each eigenvector real and positive
    eigvecs = eigvecs.astype(complex)
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
magnetic_field_vectors = np.array(magnetic_field_vectors)

# Compute Berry connection for all states
A = berry_connection(eigenstates, phi_vals)

# Compute the Berry phase for all states
berry_phases = berry_phase(A, phi_vals)

# Print the Berry phase for each state
for i, phase in enumerate(berry_phases):
    print(f"Berry Phase for state {i}: {phase}")
    # For a two-level system with a rotating magnetic field, the Berry phase should be ±(1-cos(theta))*π
    expected_phase = (1 - np.cos(theta)) * np.pi
    if i == 0:
        expected_phase = -expected_phase
    print(f"Expected Berry Phase for state {i}: {expected_phase}")
    print(f"Difference: {phase - expected_phase}")
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

# Plot the magnetic field vector in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the path traced by the magnetic field
ax.plot(magnetic_field_vectors[:, 0], magnetic_field_vectors[:, 1], magnetic_field_vectors[:, 2], 'b-', label='B field path')

# Add markers for a few points to show the direction of the path
marker_indices = np.linspace(0, len(magnetic_field_vectors)-1, 10, dtype=int)
ax.scatter(magnetic_field_vectors[marker_indices, 0], magnetic_field_vectors[marker_indices, 1], magnetic_field_vectors[marker_indices, 2], 
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

# Show the plot
plt.tight_layout()
plt.show()
