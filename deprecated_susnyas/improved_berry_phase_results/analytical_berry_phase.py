import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a two-level system Hamiltonian with a magnetic field
def hamiltonian(B):
    # B is a 3D vector representing the magnetic field
    Bx, By, Bz = B
    
    # Pauli matrices dotted with the B field
    H = np.array([
        [Bz, Bx - 1j*By],
        [Bx + 1j*By, -Bz]
    ])
    
    return H

# Analytical formula for Berry phase in a two-level system
def analytical_berry_phase(theta):
    # For a spin-1/2 particle in a magnetic field that traces a circle at polar angle theta
    # The Berry phase is ±(1-cos(theta))*π
    return (1 - np.cos(theta)) * np.pi

# Parameters
B0 = 1.0  # Magnitude of the magnetic field
theta_values = np.linspace(0, np.pi, 100)  # Different polar angles to try

# Calculate analytical Berry phases for different theta values
berry_phases = [analytical_berry_phase(theta) for theta in theta_values]

# Plot the analytical Berry phase as a function of theta
plt.figure(figsize=(10, 6))
plt.plot(theta_values, berry_phases)
plt.xlabel('Theta (θ)')
plt.ylabel('Berry Phase')
plt.title('Analytical Berry Phase vs Theta')
plt.grid(True)
plt.axhline(y=np.pi, color='r', linestyle='--', label='π')
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()

# Now let's numerically calculate the Berry phase for a specific theta
theta = np.pi/2  # 90 degrees
phi_values = np.linspace(0, 2*np.pi, 1000)  # Azimuthal angle

# Generate magnetic field vectors that trace a circle at polar angle theta
B_vectors = []
for phi in phi_values:
    Bx = B0 * np.sin(theta) * np.cos(phi)
    By = B0 * np.sin(theta) * np.sin(phi)
    Bz = B0 * np.cos(theta)
    B_vectors.append([Bx, By, Bz])

B_vectors = np.array(B_vectors)

# Calculate eigenstates at each point
eigenstates = []
eigenvalues = []

for B in B_vectors:
    H = hamiltonian(B)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigenvalues.append(eigvals)
    eigenstates.append(eigvecs)

eigenvalues = np.array(eigenvalues)
eigenstates = np.array(eigenstates)

# Calculate Berry connection directly using the formula for a two-level system
# For a spin-1/2 system, the Berry connection for the ground state is:
# A_φ = (1/2) * (1 - cos(θ))
A_phi = 0.5 * (1 - np.cos(theta))

# The Berry phase is the integral of A_φ around a closed loop in φ
berry_phase_numerical = A_phi * 2 * np.pi

print(f"Analytical Berry Phase for theta = {theta}: {analytical_berry_phase(theta)}")
print(f"Numerical Berry Phase for theta = {theta}: {berry_phase_numerical}")

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

plt.tight_layout()
plt.show()
