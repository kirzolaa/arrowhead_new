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

# Calculate the Berry connection directly
def berry_connection_direct(eigenvectors, phi):
    """
    Calculate the Berry connection for a two-level system directly.
    
    For a spin-1/2 system with the ground state |ψ⟩, the Berry connection is:
    A_φ = -i⟨ψ|∂ψ/∂φ⟩
    
    For a magnetic field tracing a circle at fixed theta, the Berry connection is:
    A_φ = (1/2) * (1 - cos(theta))
    """
    num_phi_points, num_dimensions, num_states = eigenvectors.shape
    A = np.zeros((num_states, num_phi_points), dtype=complex)
    
    for n in range(num_states):
        for p in range(num_phi_points):
            # Extract the eigenstate at phi point p for state n
            psi = eigenvectors[p, :, n]
            
            # Calculate the next phi point (with periodic boundary)
            p_next = (p + 1) % num_phi_points
            psi_next = eigenvectors[p_next, :, n]
            
            # Ensure proper phase relationship between adjacent eigenstates
            overlap = np.vdot(psi, psi_next)
            phase = np.angle(overlap)
            
            # Adjust the phase of the next state to ensure smooth evolution
            if abs(overlap) > 1e-10:  # Only adjust if there's significant overlap
                psi_next = psi_next * np.exp(-1j * phase)
            
            # Simple finite difference for derivative
            d_phi = phi[p_next] - phi[p] if p_next > p else phi[p_next] + 2*np.pi - phi[p]
            
            # Avoid division by zero
            if abs(d_phi) < 1e-10:
                d_psi_dphi = np.zeros_like(psi)
            else:
                d_psi_dphi = (psi_next - psi) / d_phi
            
            # Berry connection: A = -i⟨ψ|∂ψ/∂φ⟩
            A[n, p] = -1j * np.vdot(psi, d_psi_dphi)
    
    return A

# Calculate the Berry phase by integrating the Berry connection
def berry_phase(A, phi):
    """
    Calculate the Berry phase by integrating the Berry connection around a closed loop.
    
    gamma = ∮ A(φ) dφ
    """
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        phase_value = np.trapezoid(A[n, :], phi)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases

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
B_vectors = np.array(B_vectors)

# Compute Berry connection for all states
A = berry_connection_direct(eigenstates, phi_vals)

# Compute the Berry phase for all states
berry_phases = berry_phase(A, phi_vals)

# Calculate the analytical Berry phase
# For a spin-1/2 system, the Berry phase for the ground state is:
# gamma = -π * (1 - cos(theta))
analytical_phases = np.array([-np.pi * (1 - np.cos(theta)), np.pi * (1 - np.cos(theta))])

# Print the Berry phase for each state
for i, phase in enumerate(berry_phases):
    print(f"Numerical Berry Phase for state {i}: {phase}")
    print(f"Analytical Berry Phase for state {i}: {analytical_phases[i]}")
    print(f"Difference: {phase - analytical_phases[i]}")
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

# Plot the Berry connection (real and imaginary parts)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for i in range(A.shape[0]):
    plt.plot(phi_vals, np.real(A[i, :]), label=f'State {i} (Real)')
plt.xlabel('Phi (φ)')
plt.ylabel('Berry Connection (Real Part)')
plt.title('Real Part of Berry Connection vs Phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
for i in range(A.shape[0]):
    plt.plot(phi_vals, np.imag(A[i, :]), label=f'State {i} (Imag)')
plt.xlabel('Phi (φ)')
plt.ylabel('Berry Connection (Imaginary Part)')
plt.title('Imaginary Part of Berry Connection vs Phi')
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
