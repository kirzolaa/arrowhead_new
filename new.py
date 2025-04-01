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
import multiprocessing

# Import the perfect orthogonal circle generation function from the Arrowhead/generalized package
import sys
import os
sys.path.append('/home/zoli/arrowhead_new/completely_new/arrowhead_new/generalized')
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle
from main import *
print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")

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
def V_x(R_theta, aVx):
    # Calculate individual V_x components for each R_theta component
    Vx = [aVx * (R_theta[i] ** 2) for i in range(len(R_theta))]
    return Vx

def V_a(R_theta, aVa, c, x_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va = [aVa * ((R_theta[i] - x_shift) ** 2) + c for i in range(len(R_theta))]
    return Va
"""
# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, aVx)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, aVa, c_const, x_shift)  # [Va0, Va1, Va2]
    
    # Create a 4x4 Hamiltonian with an arrowhead structure
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    #H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    #H[1, 1] = Va[0] + Vx[1] + Vx[2]
    #H[2, 2] = Vx[0] + Va[1] + Vx[2]
    #H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    #or we can do it like this:
    #H[0, 0] = hbar * omega + [sum of all V
    sumVx = sum(Vx)
    H[0, 0] = hbar * omega + sumVx
    for i in range(1, len(H)):
        H[i, i] = H[0, 0] + Va[i-1] - Vx[i-1]
        

    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    for i in range(1, len(H)):
        H[i, 0] = H[0, i] = c
    eigvals, eigvecs = np.linalg.eigh(H)
    sorted_indices = np.argsort(eigvals)
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    #print(H.shape)
    return H, R_theta_val, Vx, Va, sorted_eigvals, sorted_eigvecs

"""

def position_matrix(R_theta_val):
    """
    Conceptual example: Creates a position matrix based on R_theta_val.
    This is not physically accurate for most systems.
    """
    x, y, z = R_theta_val
    return np.array([
        [0, x, 0, 0],
        [x, 0, y, 0],
        [0, y, 0, z],
        [0, 0, z, 0]
    ], dtype=complex)

def transitional_dipole_moment(eigvec_i, eigvec_f, position_operator):
    return np.vdot(eigvec_f, np.dot(position_operator, eigvec_i))

def hamiltonian(theta, omega, aVx, aVa, c_const, x_shift, d):
    R_theta_val = R_theta(d, theta)
    Vx = V_x(R_theta_val, aVx)
    Va = V_a(R_theta_val, aVa, c_const, x_shift)

    H = np.zeros((4, 4), dtype=complex)
    sumVx = sum(Vx)
    H[0, 0] = hbar * omega + sumVx
    for i in range(1, len(H)):
        H[i, i] = H[0, 0] + Va[i-1] - Vx[i-1]

    eigvals, eigvecs = np.linalg.eigh(H)
    sorted_indices = np.argsort(eigvals)
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]

    pos_mat = position_matrix(R_theta_val) # create the position matrix.

    c10 = transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 1], pos_mat)
    c20 = transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 2], pos_mat)
    c30 = transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 3], pos_mat)

    H[0, 1] = H[1, 0] = c10
    H[0, 2] = H[2, 0] = c20
    H[0, 3] = H[3, 0] = c30

    return H, R_theta_val, Vx, Va, sorted_eigvals, sorted_eigvecs

"""
# Function to compute the Berry connection A(R_theta)
def berry_connection(R_vals, eigvecs):
    num_states = eigvecs.shape[1]
    num_points = R_vals.shape[0]

    A_R = np.zeros((num_points - 1, num_states), dtype=complex)

    for j in range(1, num_points - 1): # Notice the change in range
        dR = R_vals[j + 1] - R_vals[j - 1]
        for state_index in range(num_states):
            v0 = eigvecs[j - 1, :, state_index]
            v2 = eigvecs[j + 1, :, state_index]
            dv_dR = (v2 - v0) / (2 * np.linalg.norm(dR))
            A_R[j, state_index] = np.vdot(eigvecs[j, :, state_index], 1j * dv_dR)
    # now we must do the first and last values using forward, and backwards differences.
    dRforward = R_vals[1] - R_vals[0]
    dRbackwards = R_vals[-1] - R_vals[-2]

    for state_index in range(num_states):
        v1 = eigvecs[0, :, state_index]
        v2 = eigvecs[1, :, state_index]
        dv_dR = (v2 - v1) / np.linalg.norm(dRforward)
        A_R[0, state_index] = np.vdot(v1, 1j * dv_dR)

        v1 = eigvecs[-2, :, state_index]
        v2 = eigvecs[-1, :, state_index]
        dv_dR = (v2 - v1) / np.linalg.norm(dRbackwards)
        A_R[-1, state_index] = np.vdot(eigvecs[-2, :, state_index], 1j * dv_dR)

    return A_R
"""

def berry_connection(eigvecs_all):
    """
    Computes the Berry connection A(theta) for each eigenstate.

    Parameters:
    - eigvecs_all: Array of eigenvectors at each theta (shape: (num_points, 4, 4)).

    Returns:
    - A_theta: Berry connection values for each eigenstate (shape: (num_points-1, 4)).
    """
    num_states = eigvecs_all.shape[2]  # Number of eigenstates (should be 4 or 2 for testing)
    num_points = eigvecs_all.shape[0]
    
    A_theta = np.zeros((num_points - 1, num_states), dtype=complex)
    
    # Compute numerical derivative with respect to theta
    for j in range(num_points - 1):
        for state_index in range(num_states):
            v1 = eigvecs_all[j, :, state_index]  # Eigenvector at point j
            
            # Use numpy's gradient function to compute ∂v/∂theta
            dv_dtheta = np.gradient(eigvecs_all[:, :, state_index], axis=0)[j]
            
            # Compute A(theta) = ⟨v | i ∇_theta | v⟩
            A_theta[j, state_index] = np.vdot(v1, 1j * dv_dtheta)
    print(A_theta[j, state_index])
    
    return A_theta

# Function to compute the Berry phase γ by integrating A(R_theta)
def berry_phase(A_R):
    """
    Computes the Berry phase by integrating A(R_theta) over the full cycle.

    Parameters:
    - A_R: Berry connection values for each eigenstate (shape: (num_points-1, 4)).

    Returns:
    - Berry phase γ for each eigenstate (shape: (4,)).
    """
    # Integrate A(R_theta) over the full cycle using summation
    gamma = np.sum(np.real(A_R), axis=0)
    return gamma % (2 * np.pi)  # Wrap to [0, 2π]

def test_berry_hamiltonian(theta_vals):
    H_thetas = [np.array([
        [np.cos(theta), np.sin(theta)],
        [np.sin(theta), -np.cos(theta)]
    ]) for theta in theta_vals] #this has a pi berry phase
    return H_thetas
    
# Parameters for the arrowhead matrix
omega = 0.1  # Frequency
#let a be an aVx and an aVa parameter
aVx = 1.0
aVa = 2.0
b = 1.0  # Potential parameter
c_const = 25.0  # Potential constant, shifts the 2d parabola on the y axis
x_shift = 25.0  # Shift in x direction
d = 1.0  # Radius of the circle, use the unit circle
theta_min = 0
theta_max = 2 * np.pi
num_points = 5000
R_0 = (0, 0, 0)
# Generate the arrowhead matrix and Va, Vx
theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
# Compute R_theta values for all theta values
#R_vals = np.array([R_theta(d, theta) for theta in theta_vals])



# Initialize an empty list to store the eigenvectors
eigvecs_all = []
eigvals_all = []
R_vals = []
H_theta = []

# Compute eigenvectors for all theta values
for theta in theta_vals:
    H, R_theta_val, Vx, Va, sorted_eigvals, sorted_eigvecs = hamiltonian(theta, omega, aVx, aVa, c_const, x_shift, d)
    eigvecs_all.append(sorted_eigvecs)
    R_vals.append(R_theta_val)
    H_theta.append(H)
    eigvals_all.append(sorted_eigvals)

# Convert the lists to numpy arrays
eigvecs_all = np.array(eigvecs_all)
R_vals = np.array(R_vals)
H_theta = np.array(H_theta)
eigvals_all = np.array(eigvals_all)

#plot the R_vals, Vx, Va, eigvals_all, eigvecs_all, H*v for each eigenstate
import matplotlib.pyplot as plt

plot_dir = 'plots'
npy_dir = 'npy'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)
#append the plots dir with real, imag and abs dirs
os.makedirs(os.path.join(plot_dir, 'real'), exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'imag'), exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'abs'), exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'total_sum'), exist_ok=True)


#plot the H*v aka Hamiltonian times eigenvectors
plt.figure(figsize=(12, 6))
for state in range(eigvecs_all.shape[2]):
    # Calculate H*v for each theta value
    Hv_results = np.zeros((len(theta_vals), eigvecs_all.shape[1]), dtype=complex)
    #get eigenvaluesof each H_theta, it is not theta vals
    #calculate H_thetas array by calculating H_theta, it should be a (num_points, 4, 4) array, like (theta_value, 4, 4)
    H_thetas = np.array([hamiltonian(theta, omega, aVx, aVa, c_const, x_shift, d)[0] for theta in theta_vals])
    print(H_thetas.shape)
    # Get all the eigenvalues
    eigenvalues = np.array([np.linalg.eigvalsh(H) for H in H_thetas])
    
    # Get all eigenvalues and eigenvectors separately
    eigenvals_eigvecs = [np.linalg.eigh(H) for H in H_thetas]
    eigenvalues_full = np.array([ev[0] for ev in eigenvals_eigvecs])
    # Extract the eigenvalues and eigenvectors
    eigenvalues = np.array([ev[0] for ev in eigenvals_eigvecs])
    eigenstates = np.array([ev[1] for ev in eigenvals_eigvecs])
    
    # For reference, H*v = λ*v
    #calculate H*v
    #use H_thetas and eigenstates
    for i, theta in enumerate(theta_vals):
        Hv_results[i] = H_thetas[i] @ eigenstates[i, :, state]
    #calculate λ*v
    lambda_v = eigenvalues[:, state][:, np.newaxis] * eigenstates[:, :, state]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for j in range(4):  # Just plot components for one state at a time
        #plot the magnitude of H*v
        axs[j].plot(theta_vals, np.abs(Hv_results[:, j]), 'ro', label='|H*v|')
        #plot the magnitude of λ*v
        axs[j].plot(theta_vals, np.abs(lambda_v[:, j]), 'bo', label='|λ*v|')
        
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
        
    plt.tight_layout()
    plt.suptitle(f'H*v for State {state}')
    plt.subplots_adjust(top=0.92)
        
    plt.savefig(f'{plot_dir}/abs/H_times_v_state_{state}.png')
    plt.close()

    #save the Hv_results
    np.save(f'{npy_dir}/H_times_v_state_{state}', Hv_results)
    #save the lambda_v
    np.save(f'{npy_dir}/lambda_v_state_{state}', lambda_v)

    #plot the real part of H*v and lambda_v
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for j in range(4):  # Just plot components for one state at a time
        #plot the real of H*v
        axs[j].plot(theta_vals, np.real(Hv_results[:, j]), 'ro', label='Re(H*v)')
        #plot the real of lambda_v
        axs[j].plot(theta_vals, np.real(lambda_v[:, j]), 'bo', label='Re(λ*v)')
        
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
        
    plt.tight_layout()
    plt.suptitle(f'H*v for State {state}')
    plt.subplots_adjust(top=0.92)
        
    plt.savefig(f'{plot_dir}/real/H_times_v_state_{state}_real.png')
    plt.close()

    #plot the imaginary part of H*v and lambda_v
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for j in range(4):  # Just plot components for one state at a time
        #plot the imaginary of H*v
        axs[j].plot(theta_vals, np.imag(Hv_results[:, j]), 'ro', label='Im(H*v)')
        #plot the imaginary of lambda_v
        axs[j].plot(theta_vals, np.imag(lambda_v[:, j]), 'bo', label='Im(λ*v)')
        
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
        
    plt.tight_layout()
    plt.suptitle(f'H*v for State {state}')
    plt.subplots_adjust(top=0.92)
        
    plt.savefig(f'{plot_dir}/imag/H_times_v_state_{state}_imag.png')
    plt.close()

    #sum up Hv ACROSS ALL THE 4 COMPONENTS
    Hv_sum = np.zeros((4, len(theta_vals)), dtype=complex)
    for j in range(4):
        Hv_sum[j] = Hv_results[:, j]
    
    #plot each component of Hv_sum in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for j in range(4):
        axs[j].plot(theta_vals, np.abs(Hv_sum[j]), 'ro', label=f'Component {j}')
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
    plt.tight_layout()
    plt.suptitle(f'H*v Sum Components for State {state}')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/abs/H_times_v_sum_components_state_{state}.png')
    #print(f"Saved {plot_dir}/abs/H_times_v_sum_components_state_{state}.png")
    plt.close()

    #plot each component of Hv_sum in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for j in range(4):
        axs[j].plot(theta_vals, np.real(Hv_sum[j]), 'ro', label=f'Component {j}')
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
    plt.tight_layout()
    plt.suptitle(f'H*v Sum Components for State {state}')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/real/H_times_v_sum_components_state_{state}.png')
    #print(f"Saved {plot_dir}/real/H_times_v_sum_components_state_{state}.png")
    plt.close()

    #plot each component of Hv_sum in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for j in range(4):
        axs[j].plot(theta_vals, np.imag(Hv_sum[j]), 'ro', label=f'Component {j}')
        axs[j].set_title(f'Component {j}')
        axs[j].set_xlabel('Theta')
        axs[j].set_ylabel('Value')
        axs[j].grid(True)
        axs[j].legend()
    plt.tight_layout()
    plt.suptitle(f'H*v Sum Components for State {state}')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/imag/H_times_v_sum_components_state_{state}.png')
    #print(f"Saved {plot_dir}/imag/H_times_v_sum_components_state_{state}.png")
    plt.close()

        # Calculate the sum of H*v across all components and theta values
    S_total = np.sum(Hv_results, axis=(1, 0))

    # Calculate the sum of lambda*v across all components and theta values
    lambda_total = np.sum(lambda_v, axis=(1, 0))

    # Print the total sums
    print(f"State {state}: Sum(H*v) = {S_total}, Sum(lambda*v) = {lambda_total}")
    with open(f'{plot_dir}/total_sum/total_sum_state_{state}.txt', 'a') as f:
        f.write(f"State {state}\n====================\nSum(H*v) = {S_total}\nSum(lambda*v) = {lambda_total}\n")

    # Compute the Berry connection
    A_R_vals = berry_connection(eigvecs_all)

    # Compute the Berry phase
    berry_phases_corrected = berry_phase(A_R_vals)

# Output the computed Berry phases
print(np.array2string(berry_phases_corrected, formatter={'float_kind':lambda x: np.format_float_scientific(x, precision=10)}))

H_thetas = test_berry_hamiltonian(theta_vals)

# Compute the eigenvalues and eigenvectors for each H_theta
eigvecs_all = np.array([np.linalg.eigh(H)[1] for H in H_thetas])

# Compute the Berry connection using H_thetas eigenvectors
A_R_vals = berry_connection(eigvecs_all)

# Compute the Berry phase
berry_phases_corrected = berry_phase(A_R_vals)
print("Berry phases from test Hamiltonian:", berry_phases_corrected)
