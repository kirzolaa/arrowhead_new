import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define physical constants
hbar = 1.0545718e-34  # Planck's constant divided by 2π in J·s (joule-seconds)

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
    # v_{x0}(R[0]): potential based on R_theta[0]
    Vx0 = a * R_theta[0] + b * R_theta[0] + c
    # v_{x1}(R[1]): potential based on R_theta[1]
    Vx1 = a * R_theta[1] + b * R_theta[1] + c
    # v_{x2}(R[2]): potential based on R_theta[2]
    Vx2 = a * R_theta[2] + b * R_theta[2] + c
    # Return individual components as a list
    return [Vx0, Vx1, Vx2]

def V_a(R_theta, a, b, c, x_shift, y_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    # v_{a0}(R[0]): potential based on R_theta[0]
    Va0 = a * (R_theta[0] - x_shift) + b * (R_theta[0] - y_shift) + c
    # v_{a1}(R[1]): potential based on R_theta[1]
    Va1 = a * (R_theta[1] - x_shift) + b * (R_theta[1] - y_shift) + c
    # v_{a2}(R[2]): potential based on R_theta[2]
    Va2 = a * (R_theta[2] - x_shift) + b * (R_theta[2] - y_shift) + c
    # Return individual components as a list
    return [Va0, Va1, Va2]

# Define the Hamiltonian matrix D(θ)
def hamiltonian(theta, r1, r2, r3, omega, a, b, c, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, a, b, c)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, a, b, c, x_shift, y_shift)  # [Va0, Va1, Va2]
    
    # Create a Hamiltonian with explicit theta dependence to ensure non-zero Berry phase
    # This is inspired by the spin-1/2 system in a magnetic field
    
    # Define a 4x4 Hamiltonian matrix
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    H[1, 1] = Va[0] + Vx[1] + Vx[2]
    H[2, 2] = Vx[0] + Va[1] + Vx[2]
    H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    
    # Coupling between states 0 and 1 with theta dependence
    H[0, 1] = H[1, 0] = r1
    
    # Coupling between states 0 and 2 with theta dependence
    H[0, 2] = H[2, 0] = r2
    
    # Coupling between states 0 and 3 (constant)
    H[0, 3] = H[3, 0] = r3
    
    # Other off-diagonal elements are zero (arrowhead structure)
    
    return H
    
    # Hamiltonian matrix
    D = np.array([[D00, D01, D02, D03],
                  [D10, D11, D12, D13],
                  [D20, D21, D22, D23],
                  [D30, D31, D32, D33]])
    
    return D

# Compute the Berry connection
def berry_connection(eigenstates, theta):
    # eigenstates shape is (num_theta_points, num_dimensions, num_states)
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
            # Calculate the overlap between current and next state
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
    # A has shape (num_states, num_theta_points)
    # We integrate over theta for each state
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        # The Berry phase is the integral of the Berry connection around a closed loop
        # We need to ensure the integration is over a full 2π cycle
        phase_value = np.trapezoid(A[n, :], theta)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases

# Parameters
# Increase coupling parameters to make Berry phase more pronounced
r1 = 0.5  # Coupling between states 0 and 1
r2 = 0.7  # Coupling between states 0 and 2
r3 = 0.9  # Coupling between states 0 and 3
omega = 1.0  # Frequency parameter

# Coefficients for the potential functions with more variation
a = 1.5  # First coefficient for potentials
b = 0.8  # Second coefficient for potentials
c = 0.3  # Constant term in potentials

# Shifts for the Va potential to create asymmetry
x_shift = 0.5  # Shift for the Va potential on the x-axis
y_shift = 0.7  # Shift for the Va potential on the y-axis

d = 1.0  # Parameter for R_theta (distance or other parameter)
theta_vals = np.linspace(0, 2 * np.pi, 500)  # 500 points around the circle for better numerical accuracy

# Initialize arrays for storing eigenvalues, eigenstates, R_theta vectors, and potentials
eigenvalues = []
eigenstates = []
r_theta_vectors = []  # Store the R_theta vectors for plotting

# Arrays for individual potential components
vx0_values = []  # Store Vx0 values
vx1_values = []  # Store Vx1 values
vx2_values = []  # Store Vx2 values
va0_values = []  # Store Va0 values
va1_values = []  # Store Va1 values
va2_values = []  # Store Va2 values

# Loop over theta values to compute the eigenvalues and eigenstates
for theta in theta_vals:
    # Calculate R_theta vector for this theta
    r_theta_vector = R_theta(d, theta)
    r_theta_vectors.append(r_theta_vector)
    
    # Calculate potentials (each returns a list of 3 components)
    vx = V_x(r_theta_vector, a, b, c)  # [Vx0, Vx1, Vx2]
    va = V_a(r_theta_vector, a, b, c, x_shift, y_shift)  # [Va0, Va1, Va2]
    
    # Store individual potential components
    vx0_values.append(vx[0])
    vx1_values.append(vx[1])
    vx2_values.append(vx[2])
    va0_values.append(va[0])
    va1_values.append(va[1])
    va2_values.append(va[2])
    
    # Calculate Hamiltonian
    D = hamiltonian(theta, r1, r2, r3, omega, a, b, c, x_shift, y_shift, d)
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = la.eigh(D)
    
    # Apply phase convention: make the first component of each eigenvector real and positive
    # Convert eigvecs to complex type to avoid casting warnings
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

# Plot the individual potential components
plt.figure(figsize=(12, 8))

# Plot Vx components
plt.subplot(2, 1, 1)
plt.plot(theta_vals, vx0_values, 'r-', label='V_x0')
plt.plot(theta_vals, vx1_values, 'g-', label='V_x1')
plt.plot(theta_vals, vx2_values, 'b-', label='V_x2')
plt.plot(theta_vals, np.array(vx0_values) + np.array(vx1_values) + np.array(vx2_values), 'k--', label='V_x (sum)')
plt.xlabel('Theta (θ)')
plt.ylabel('Potential Value')
plt.title('V_x Components vs Theta')
plt.grid(True)
plt.legend()

# Plot Va components
plt.subplot(2, 1, 2)
plt.plot(theta_vals, va0_values, 'r-', label='V_a0')
plt.plot(theta_vals, va1_values, 'g-', label='V_a1')
plt.plot(theta_vals, va2_values, 'b-', label='V_a2')
plt.plot(theta_vals, np.array(va0_values) + np.array(va1_values) + np.array(va2_values), 'k--', label='V_a (sum)')
plt.xlabel('Theta (θ)')
plt.ylabel('Potential Value')
plt.title('V_a Components vs Theta')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Convert r_theta_vectors to a numpy array for easier manipulation
r_theta_vectors = np.array(r_theta_vectors)

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

# Plot the axis of rotation (the line x=y=z)
axis_line = np.array([[-1, -1, -1], [1, 1, 1]]) * d
ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], 'g--', label='Rotation Axis (x=y=z)')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('R_theta Vectors in 3D Space')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
