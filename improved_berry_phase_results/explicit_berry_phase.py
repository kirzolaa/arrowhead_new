import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define physical constants
hbar = 1.0  # Simplified value for demonstration

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta):
    """
    Create a 4x4 Hamiltonian with an arrowhead structure that explicitly 
    includes terms that will produce a non-zero Berry phase.
    
    This Hamiltonian has the form:
    H = [  E0,  g1*e^(i*theta),  g2*e^(-i*theta),  g3  ]
        [ g1*e^(-i*theta),  E1,    0,    0  ]
        [ g2*e^(i*theta),    0,    E2,    0  ]
        [  g3,    0,    0,    E3  ]
    
    where E0, E1, E2, E3 are the diagonal energies and g1, g2, g3 are coupling constants.
    """
    # Diagonal energies
    E0 = 2.0
    E1 = 1.0
    E2 = 1.0
    E3 = 1.0
    
    # Coupling constants
    g1 = 0.5
    g2 = 0.7
    g3 = 0.3
    
    # Create the Hamiltonian matrix
    H = np.zeros((4, 4), dtype=complex)
    
    # Diagonal elements
    H[0, 0] = E0
    H[1, 1] = E1
    H[2, 2] = E2
    H[3, 3] = E3
    
    # Off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    H[0, 1] = g1 * np.exp(1j * theta)
    H[1, 0] = g1 * np.exp(-1j * theta)
    
    H[0, 2] = g2 * np.exp(-1j * theta)
    H[2, 0] = g2 * np.exp(1j * theta)
    
    H[0, 3] = g3
    H[3, 0] = g3
    
    return H

# Calculate the Berry connection
def berry_connection(eigenstates, theta):
    """
    Calculate the Berry connection for each eigenstate.
    
    A_n(theta) = -i <psi_n(theta)| d/dtheta |psi_n(theta)>
    """
    num_theta_points, num_dimensions, num_states = eigenstates.shape
    A = np.zeros((num_states, num_theta_points), dtype=complex)
    
    for n in range(num_states):
        for t in range(num_theta_points):
            # Extract the eigenstate at theta point t for state n
            psi_n = eigenstates[t, :, n]
            
            # Calculate the next theta point (with periodic boundary)
            t_next = (t + 1) % num_theta_points
            psi_n_next = eigenstates[t_next, :, n]
            
            # Ensure proper phase relationship between adjacent eigenstates
            overlap = np.vdot(psi_n, psi_n_next)
            phase = np.angle(overlap)
            
            # Adjust the phase of the next state to ensure smooth evolution
            if abs(overlap) > 1e-10:  # Only adjust if there's significant overlap
                psi_n_next = psi_n_next * np.exp(-1j * phase)
            
            # Simple finite difference for derivative
            d_theta = theta[t_next] - theta[t] if t_next > t else theta[t_next] + 2*np.pi - theta[t]
            
            # Avoid division by zero
            if abs(d_theta) < 1e-10:
                d_psi_n_dtheta = np.zeros_like(psi_n)
            else:
                d_psi_n_dtheta = (psi_n_next - psi_n) / d_theta
            
            # Berry connection: A = -i⟨ψ|∂ψ/∂θ⟩
            A[n, t] = -1j * np.vdot(psi_n, d_psi_n_dtheta)
    
    return A

# Calculate the Berry phase
def berry_phase(A, theta):
    """
    Calculate the Berry phase by integrating the Berry connection around a closed loop.
    
    gamma_n = ∮ A_n(theta) dtheta
    """
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        phase_value = np.trapezoid(A[n, :], theta)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases

# Parameters
num_points = 1000
theta_vals = np.linspace(0, 2*np.pi, num_points)

# Initialize arrays for storing eigenvalues and eigenstates
eigenvalues = []
eigenstates = []

# Loop over theta values to compute the eigenvalues and eigenstates
for theta in theta_vals:
    # Calculate the Hamiltonian for this theta
    H = hamiltonian(theta)
    
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

# Compute Berry connection for all states
A = berry_connection(eigenstates, theta_vals)

# Compute the Berry phase for all states
berry_phases = berry_phase(A, theta_vals)

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

# Plot the Berry connection (real and imaginary parts)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for i in range(A.shape[0]):
    plt.plot(theta_vals, np.real(A[i, :]), label=f'State {i} (Real)')
plt.xlabel('Theta (θ)')
plt.ylabel('Berry Connection (Real Part)')
plt.title('Real Part of Berry Connection vs Theta')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
for i in range(A.shape[0]):
    plt.plot(theta_vals, np.imag(A[i, :]), label=f'State {i} (Imag)')
plt.xlabel('Theta (θ)')
plt.ylabel('Berry Connection (Imaginary Part)')
plt.title('Imaginary Part of Berry Connection vs Theta')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Show the plots
plt.show()
