import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def generate_arrowhead_hamiltonian(theta):
    """Generates an arrowhead Hamiltonian with complex off-diagonal elements."""
    a = np.cos(theta)
    b = np.sin(theta) + 1j * np.cos(theta)  # Complex element
    c = np.cos(theta)
    d = np.sin(theta)
    e = 1
    f = 2
    g = 3
    
    return np.array([
        [a, b, c, d],
        [b, e, 0, 0],
        [c, 0, f, 0],
        [d, 0, 0, g]
    ])

def calculate_berry_phase(Hamiltonians, theta_vals, eigenstate_index):
    """Calculates the Berry phase with phase fixing."""
    
    berry_phase = 0.0
    
    # Calculate eigenvectors for the first Hamiltonian
    eigenvalues_i, eigenvectors_i = eigh(Hamiltonians[0])
    ground_state_i = eigenvectors_i[:, eigenstate_index]
    
    for i in range(len(theta_vals) - 1):
        H_ip1 = Hamiltonians[i + 1]
        
        # Calculate eigenvectors
        eigenvalues_ip1, eigenvectors_ip1 = eigh(H_ip1)
        ground_state_ip1 = eigenvectors_ip1[:, eigenstate_index]
        
        # Phase fixing: Ensure continuity
        overlap = np.dot(ground_state_i.conj(), ground_state_ip1)
        ground_state_ip1 = ground_state_ip1 * np.exp(-1j * np.angle(overlap))
        
        # Calculate the Berry connection
        berry_connection = np.angle(np.dot(ground_state_i.conj(), ground_state_ip1))
        
        # Accumulate the Berry phase
        berry_phase += berry_connection
        
        # Update the current eigenvector
        ground_state_i = ground_state_ip1
        
    return berry_phase

# Test Script
num_hamiltonians = 1000
theta_vals = np.linspace(0, 2 * np.pi, num_hamiltonians)

Hamiltonians = [generate_arrowhead_hamiltonian(theta) for theta in theta_vals]

# Calculate and print Berry phases
for eigenstate_index in range(4):
    berry_phase = calculate_berry_phase(Hamiltonians, theta_vals, eigenstate_index)
    print(f"Berry phase (state {eigenstate_index}): {berry_phase}")

# Visualize Eigenvalues
eigenvalues_over_theta = []
for H in Hamiltonians:
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_over_theta.append(eigenvalues)

eigenvalues_over_theta = np.array(eigenvalues_over_theta)

for i in range(4):
    plt.plot(theta_vals, eigenvalues_over_theta[:, i], label=f"Eigenvalue {i}")

plt.xlabel("Theta")
plt.ylabel("Eigenvalue")
plt.legend()
plt.show()