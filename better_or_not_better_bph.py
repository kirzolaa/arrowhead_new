import numpy as np
import matplotlib.pyplot as plt
import os
from new_bph import Hamiltonian
#import an rk4 integrator
from scipy.integrate import odeint
from os.path import join
from generalized.vector_utils import multiprocessing_create_perfect_orthogonal_circle
from perfect_orthogonal_circle import verify_circle_properties, visualize_perfect_orthogonal_circle
from scipy.constants import hbar
import multiprocessing as mp
    
def visualize_vectorz(R_0, d, num_points, theta_min, theta_max, save_dir):
    #use the perfect_orthogonal_circle.py script to visualize the R_theta vectors
    
    #visualize the R_theta vectors
    points = multiprocessing_create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max) #we already have a method for this
    #points = create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max)
    print(points.shape)
    visualize_perfect_orthogonal_circle(points, save_dir)
    verify_circle_properties(d, num_points, points, save_dir)

def plot_matrix_elements(tau, gamma, theta_vals, output_dir):
    """
    Plot the evolution of specific matrix elements (01, 12, 13) for both tau and gamma matrices.
    
    Parameters:
    - tau: 3D array of shape (M, M, N) containing tau values over theta
    - gamma: 3D array of shape (M, M, N) containing gamma values over theta
    - theta_vals: 1D array of theta values
    - output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # Elements to plot
    elements = [(0, 1), (1, 2), (1, 3)]
    
    # Plot real and imaginary parts of tau
    plt.subplot(2, 1, 1)
    for i, j in elements:
        plt.plot(theta_vals, np.real(tau[i, j, :]), 
                label=f'Re(τ_{i+1}{j+1})', linestyle='-')
        plt.plot(theta_vals, np.imag(tau[i, j, :]), 
                label=f'Im(τ_{i+1}{j+1})', linestyle='--')
    plt.xlabel('θ')
    plt.ylabel('τ')
    plt.title('Evolution of τ matrix elements')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/tau_matrix_elements.png')
    
    
    plt.subplot(2, 1, 2)
    for i, j in elements:
        plt.plot(theta_vals, np.real(gamma[i, j, :]), 
                label=f'Re(γ_{i+1}{j+1})', linestyle='-')
        plt.plot(theta_vals, np.imag(gamma[i, j, :]), 
                label=f'Im(γ_{i+1}{j+1})', linestyle='--')
    plt.xlabel('θ')
    plt.ylabel('γ')
    plt.title('Evolution of γ matrix elements')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/gamma_matrix_elements.png')
    plt.close()
    
    # Create separate plots for each element
    for i, j in elements:
        plt.figure(figsize=(10, 6))
        plt.plot(theta_vals, np.real(tau[i, j, :]), label=f'Re(τ_{i+1}{j+1})')
        plt.plot(theta_vals, np.imag(tau[i, j, :]), label=f'Im(τ_{i+1}{j+1})')
        plt.plot(theta_vals, np.real(gamma[i, j, :]), '--', label=f'Re(γ_{i+1}{j+1})')
        plt.plot(theta_vals, np.imag(gamma[i, j, :]), '--', label=f'Im(γ_{i+1}{j+1})')
        plt.xlabel('θ')
        plt.ylabel('Value')
        plt.title(f'Evolution of τ and γ_{i+1}{j+1} elements')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/element_{i+1}{j+1}_evolution.png')
        plt.close()

def format_matrix(matrix, title=None, output_dir=None):
    """Format a matrix with box drawing characters"""
    n, m = matrix.shape
    max_len = max(len(f"{x:.4f}") for row in matrix for x in row)
    
    # Calculate width based on max number length and matrix dimensions
    width = (max_len + 3) * m + 1
    
    lines = []
    if title:
        lines.append(f"    |{title:^{width-2}}|")
    
    # Top border
    lines.append("    |‾" + "‾" * (width-2) + "‾|")
    
    # Matrix rows
    for i in range(n):
        row = "    |  "
        for j in range(m):
            if i == j:
                # Diagonal elements (γ_nn)
                row += f"{matrix[i,j]:.4f}"
            else:
                # Off-diagonal elements (γ_nm)
                row += f"{matrix[i,j]:.4f}"
            if j < m - 1:
                row += "  "
            else:
                row += "    |"
        lines.append(row)
    
    # Bottom border
    lines.append("    |_" + "_" * (width-2) + "_|")
    
    return "\n".join(lines)

class Eigenvalues:
    """
    Eigenvalues class for a quantum system with a potentials:

    V(x) = aVx * x^2, and
    Va(x) = aVa * (x - x_shift)^2 + c

    Creates a 4x4 matrix with an arrowhead structure, where
    
    |‾                                                  ‾|
    |    hbar*omega + Σ Vx(i)  tdm01   tdm02    tdm03    |
    |    tdm01                 V_e(i)  0        0        |
    |    tdm02                 0       V_e(i+1) 0        |
    |    tdm03                 0       0        V_e(i+2) |
    |_                                                  _|
    
    where Σ Vx(i) represents the sum of Vx values from i=0 to N, and
    V_e(i) represents the potential Σ Vx(i) + Va(i) - Vx(i) at angle i.
    
    """
    def __init__(self, H_thetas, output_dir, theta_vals):
        """
        Initialize the Eigenvalues with parameters.
        
        Parameters:
        H_thetas (numpy.ndarray): Array of Hamiltonian matrices for each angle theta
        output_dir (str): Directory to save the plots
        theta_vals (numpy.ndarray): Array of angle values
        """
        self.H_thetas = H_thetas
        self.output_dir = output_dir
        self.theta_vals = theta_vals
        self.eigenvalues = self.compute()
        
    def compute(self):
        """
        Compute the eigenvalues for each angle theta.
        
        Returns:
        numpy.ndarray: Array of eigenvalues for each angle theta
        """
        self.eigenvalues = np.array([np.linalg.eigh(H)[0] for H in self.H_thetas])
        return self.eigenvalues

    def plot(self):
        """
        Plot the eigenvalues for each angle theta.
        """
        # Plot the eigenvalues
        plt.figure(figsize=(10, 6))
        plt.plot(self.theta_vals, self.eigenvalues[:,0], 'r-', label='Eigenvalue 1')
        plt.plot(self.theta_vals, self.eigenvalues[:,1], 'b-', label='Eigenvalue 2')
        plt.plot(self.theta_vals, self.eigenvalues[:,2], 'g-', label='Eigenvalue 3')
        plt.plot(self.theta_vals, self.eigenvalues[:,3], 'c-', label='Eigenvalue 4')
        plt.xlabel('Theta')
        plt.ylabel('Eigenvalue')
        plt.title(f'Eigenvalues vs Theta')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/eigenvalues.png')
        plt.close()

class Eigenvectors:
    """
    Eigenvectors class for a quantum system with a potentials:

    V(x) = aVx * x^2, and
    Va(x) = aVa * (x - x_shift)^2 + c

    Creates a 4x4 matrix with an arrowhead structure, where
    
    |‾                                                  ‾|
    |    hbar*omega + Σ Vx(i)  tdm01   tdm02    tdm03    |
    |    tdm01                 V_e(i)  0        0        |
    |    tdm02                 0       V_e(i+1) 0        |
    |    tdm03                 0       0        V_e(i+2) |
    |_                                                  _|
    
    where Σ Vx(i) represents the sum of Vx values from i=0 to N, and
    V_e(i) represents the potential Σ Vx(i) + Va(i) - Vx(i) at angle i.
    
    """
    def __init__(self, H_thetas, output_dir, theta_vals, printout=0):
        """
        Initialize the Eigenvectors with parameters.
        
        Parameters:
        H_thetas (numpy.ndarray): Array of Hamiltonian matrices for each angle theta
        output_dir (str): Directory to save the plots
        theta_vals (numpy.ndarray): Array of angle values
        printout (int): Printout level (default: 0)
        """
        self.H_thetas = H_thetas
        self.output_dir = output_dir
        self.theta_vals = theta_vals
        self.printout = printout

    def fix_sign(self, eigvecs):
        """
        Fix the sign of the eigenvectors to ensure positive real part.
        
        Parameters:
        eigvecs (numpy.ndarray): Array of eigenvectors for each angle theta
        
        Returns:
        numpy.ndarray: Array of eigenvectors with fixed sign
        """
        # Ensure positive real part of eigenvectors
        with open(f'{self.output_dir}/eigvecs_sign_flips_{self.printout}.out', "a") as log_file:
            for i in range(eigvecs.shape[0]): #for every theta
                for j in range(eigvecs.shape[2]): #for every eigvec
                    s = 0.0
                    for k in range(eigvecs.shape[1]): #for every component
                        s += np.real(eigvecs[i, k, j]) * np.real(eigvecs[i-1, k, j]) #dot product of current and previous eigvec
                    if s < 0:
                        log_file.write(f"Flipping sign of state {j+1} at theta {i} (s={s})\n")
                        log_file.write(f"Pervious eigvec: {eigvecs[i-1, :, j]}\n")
                        log_file.write(f"Current eigvec: {eigvecs[i, :, j]}\n")
                        eigvecs[i, :, j] *= -1
        return eigvecs

    def compute(self):
        """
        Compute the eigenvectors for each angle theta.
        
        Returns:
        numpy.ndarray: Array of eigenvectors for each angle theta
        """
        eigenvectors = np.array([np.linalg.eigh(H)[1] for H in self.H_thetas])
        eigenvectors = self.fix_sign(eigenvectors)
        return eigenvectors

    def plot(self, eigvecs):# Plot eigenvector components (4 subplots in a 2x2 grid for each eigenstate)
        """
        Plot the eigenvector components for each eigenstate.
        
        Parameters:
        eigvecs (numpy.ndarray): Array of eigenvectors for each angle theta
        """
        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvector Components - All eigenvectors', fontsize=16)  # Overall title
        for state in range(eigvecs.shape[2]):
            #nest a for loop for vec_comp and use it like: :, state, vect_comp
            for vect_comp in range(4):
                plt.subplot(2, 2, vect_comp + 1)  # Top left subplot
                plt.plot(self.theta_vals, np.real(eigvecs[:, state, vect_comp]), label=f'Re(State {state+1})')
                #plt.plot(theta_vals, np.imag(eigenvectors[:, state, vect_comp]), label=f'Im(Comp {vect_comp})')
                #plt.plot(theta_vals, np.abs(eigenvectors[:, state, vect_comp]), label=f'Abs(Comp {vect_comp})')
                plt.xlabel('Theta')
                plt.ylabel(f'Component {vect_comp}')
                plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for overall title
        plt.savefig(f'{self.output_dir}/eigenvector_components_for_eigvec_2x2.png')
        plt.close()
        
def compute_berry_phase_og(eigvectors_all, theta_vals):
    """
    Note: the first and the last tau, gamma matrices are seemingly wrong!
    
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - theta_vals: ndarray of shape (N,), parameter values along the path

    Returns:
    - tau: ndarray of shape (M, M, N), Berry connection for each eigenstate in radians
    - gamma: ndarray of shape (M, M, N), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    
    tau = np.zeros((M, M, N), dtype=np.float64)
    gamma = np.zeros((M, M, N), dtype=np.float64)

    for n in range(M):
        for m in range(M):
            for i in range(N): 
                # Handle boundary conditions for the finite difference
                # Inside compute_berry_phase

                if i == 0:
                    psi_prev = eigvectors_all[N - 1, :, n]  # Vector at theta_max (which is theta_0 if path is closed)
                                                            # OR eigvectors_all[N-2,:,n] if using N-1 points to define the distinct loop points (0 to N-2) and N-1 is same as 0
                                                            # Let's assume N points, theta_vals[N-1] is distinct from theta_vals[0] but psi(theta_vals[N-1]) is "before" psi(theta_vals[0])
                    psi_next = eigvectors_all[1, :, n]
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                elif i == N - 1:
                    psi_prev = eigvectors_all[N - 2, :, n]
                    psi_next = eigvectors_all[0, :, n] # Vector at theta_0 (which is theta_N-1 + step if path is closed)
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                else:
                    psi_prev = eigvectors_all[i - 1, :, n]
                    psi_next = eigvectors_all[i + 1, :, n]
                    delta_theta_for_grad = theta_vals[i + 1] - theta_vals[i - 1]

                psi_curr = eigvectors_all[i, :, m]
                # Normalize for safety (elvileg 1-gyel osztunk itt, mivel a vektorok eigh-val számolva)
                psi_prev = psi_prev / np.linalg.norm(psi_prev)
                psi_next = psi_next / np.linalg.norm(psi_next)
                psi_curr = psi_curr / np.linalg.norm(psi_curr)

                # Finite difference approximation of ∇_theta |ψ>
                delta_psi = psi_next - psi_prev
                grad_psi = delta_psi / (delta_theta_for_grad) # Corrected delta_theta

                # τ = ⟨ψ_i | ∇_theta | ψ_{i-1}⟩  (Corrected index for tau)
                tau_val = np.vdot(psi_curr, grad_psi)
                tau[n, m, i] = tau_val
                # · d_theta to integrate. 
                if i == 0:
                   gamma[n, m, i] = 0.0
                else:
                    delta_theta_integrate = theta_vals[i] - theta_vals[i-1]
                    # Add the area of the segment from theta_vals[i-1] to theta_vals[i]
                    # Option 1: Using tau at the end of the interval (simplest Riemann sum)
                    gamma[n, m, i] = gamma[n, m, i-1] + tau[n, m, i] * delta_theta_integrate

                    # Option 2: Using trapezoidal rule (generally more accurate)
                    #gamma[n, m, i] = gamma[n, m, i-1] + (tau[n, m, i] + tau[n, m, i-1]) / 2.0 * delta_theta_integrate
                    #gamma[n, m, i] = gamma[n, m, i-1] + (tau[n, m, i] + tau[n, m, i-1]) / 2.0 * delta_theta_integrate
    return tau, gamma

def save_and__visalize_va_and_vx(npy_dir, Hamiltonians, Va_values, Vx_values, theta_vals, plot_dir):
    # Save the Hamiltonians, Va and Vx into .npy files
    np.save(f'{npy_dir}/Hamiltonians.npy', Hamiltonians)
    np.save(f'{npy_dir}/Va_values.npy', Va_values)
    np.save(f'{npy_dir}/Vx_values.npy', Vx_values)

    
    theta_values = np.linspace(0, 2*np.pi, len(Va_values))
    
    #plot Va potential components
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i+1}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Va Components')
    plt.title('Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Va_components.png')
    print("Va plots saved to figures directory.")
    plt.close()

    #plot Vx potential components
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i+1}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Vx Components')
    plt.title('Vx Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Vx_components.png')
    print("Vx plots saved to figures directory.")
    plt.close()

def compute_berry_phase_maybe_gud(eigvectors_all_, theta_vals, continuity_check=False):
    """
    Compute Berry connection (τ) and Berry phase (γ) matrices along a closed path.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors at each θ (N points, M states)
    - theta_vals: ndarray of shape (N,), θ values in path
    - continuity_check: bool, if True, print overlap continuity for debugging

    Returns:
    - tau: ndarray (M, M, N), complex Berry connection ⟨ψ_m | dψ_n/dθ⟩
    - gamma: ndarray (M, M, N), Berry phase integral over θ
    """
    # Pad one more point to close the loop
    eigvectors_all_ = np.concatenate([eigvectors_all_, eigvectors_all_[:1]], axis=0)
    theta_vals_ = np.append(theta_vals, theta_vals[-1])
    #START WITH THE SECOND THETA_VALUE
    eigvectors_all_ = eigvectors_all_[1:]
    theta_vals_ = theta_vals_[1:]
    
    N, M, _ = eigvectors_all_.shape
    tau = np.zeros((M, M, N), dtype=np.complex128)
    gamma = np.zeros((M, M, N), dtype=np.float64)

    # Normalize all eigenvectors (for safety)
    eigvectors_all_ = eigvectors_all_ / np.linalg.norm(eigvectors_all_, axis=1, keepdims=True)

    delta_theta = theta_vals_[1] - theta_vals_[0]  # assume uniform spacing

    for i in range(N):
        for n in range(M):
            for m in range(M):
                # Get previous and next for central difference
                psi_prev = eigvectors_all_[i - 1, :, n] if i > 0 else eigvectors_all_[N - 1, :, n]
                psi_next = eigvectors_all_[(i + 1) % N, :, n]
                psi_curr = eigvectors_all_[i, :, m]

                # Central difference derivative
                grad_psi = (psi_next - psi_prev) / (2 * delta_theta)

                # Berry connection τ_{nm} = i⟨ψ_m | dψ_n/dθ⟩
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                tau[n, m, i] = tau_val

                # Accumulate γ_{nm}
                if i == 0:
                    gamma[n, m, i] = 0.0
                else:
                    gamma[n, m, i] = gamma[n, m, i - 1] + np.imag(tau[n, m, i]) * delta_theta

    if continuity_check:
        print("Eigenvector continuity (should be ~1):")
        for i in range(1, N):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all_[i - 1, :, n], eigvectors_all_[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

    return tau, gamma

def compute_berry_phase_BAD(eigvectors_all_, theta_vals_, theta_max, continuity_check=False):
    """
    Compute Berry connection (τ) and Berry phase (γ) matrices along a closed path (ring-like).
    """
    # Pad one more point to close the loop
    eigvectors_all_ = np.concatenate([eigvectors_all_, eigvectors_all_[:1]], axis=0)
    theta_vals_ = np.append(theta_vals_, theta_vals[-1])
    N = eigvectors_all_.shape[0]  # now N = original + 1
    M = eigvectors_all_.shape[1]

    # Now initialize tau and gamma with updated shape
    tau = np.zeros((M, M, N), dtype=np.complex128)
    gamma = np.zeros((M, M, N), dtype=np.float64)

    delta_theta = theta_vals_[1] - theta_vals_[0]
    for i in range(N+1):
        i_prev = (i - 1) % N
        i_next = (i + 1) % N

        for n in range(M):
            for m in range(M):
                psi_prev = eigvectors_all_[i_prev, :, n]
                psi_next = eigvectors_all_[i_next, :, n]
                psi_curr = eigvectors_all_[i, :, m]

                grad_psi = (psi_next - psi_prev) / (2 * delta_theta)
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                tau[n, m, i] = tau_val

                if i > 0:
                    gamma[n, m, i] = gamma[n, m, i - 1] + np.imag(tau_val) * delta_theta

    if continuity_check:
        print("Eigenvector continuity (should be ~1):")
        for i in range(1, N):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all_[i - 1, :, n], eigvectors_all_[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

    return tau, gamma


def compute_berry_phase(eigvectors_all, theta_vals, continuity_check=False):
    """
    Compute Berry connection (τ) and Berry phase (γ) matrices using central difference,
    with ring continuity and stabilized diagonals.
    """

    N_orig, M, _ = eigvectors_all.shape

    # Extend θ and eigvecs for periodic boundary
    eigvectors_all = np.concatenate([eigvectors_all, eigvectors_all[:1]], axis=0)
    theta_vals = np.append(theta_vals, theta_vals[0] + 2 * np.pi)
    N = N_orig + 1
    delta_theta = theta_vals[1] - theta_vals[0]

    tau = np.zeros((M, M, N), dtype=np.complex128)
    gamma = np.zeros((M, M, N), dtype=np.float64)
    
    # Create arrays to track the magnitude of tau and gamma for diagnostics
    tau_imag_magnitudes = np.zeros((M, M, N_orig))
    tau_real_magnitudes = np.zeros((M, M, N_orig))
    gamma_increments = np.zeros((M, M, N_orig))

    # Optional: normalize eigenvectors for safety
    eigvectors_all = eigvectors_all / np.linalg.norm(eigvectors_all, axis=1, keepdims=True)

    # Track the largest tau values and their locations
    largest_tau_imag = 0
    largest_tau_imag_loc = (0, 0, 0)
    largest_gamma_increment = 0
    largest_gamma_increment_loc = (0, 0, 0)

    for i in range(N):
        i_prev = (i - 1)
        i_next = (i + 1) % N

        for n in range(M):
            for m in range(M):
                if i == 0:
                    i_prev = N - 1
                else:
                    i_prev = i - 1
                if i == N - 1: # at the end of the loop
                    i_next = 0
                else:
                    i_next = i + 1  # Fixed: this should just be i+1, not len(eigvectors_all)-1
                
                psi_prev = eigvectors_all[i_prev, :, n] # (THETA, COMPONENT, STATE)
                psi_next = eigvectors_all[i_next, :, n]
                psi_curr = eigvectors_all[i, :, m]

                grad_psi = (psi_next - psi_prev) / delta_theta
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                
                # Check for unusually large imaginary values at the end of the loop
                # This prevents the sudden jump in gamma at 2π
                if i == N-1 and np.abs(np.imag(tau_val)) > 10.0:
                    # Use the average of previous values instead
                    if i > 0:
                        prev_tau = tau[n, m, i-1]
                        tau_val = prev_tau
                        print(f"Limiting large tau value at (n={n+1}, m={m+1}, theta_idx={i})")
                
                tau[n, m, i] = tau_val
                
                # Store magnitudes for diagnostics
                if i < N_orig:
                    tau_imag_magnitudes[n, m, i] = np.abs(np.imag(tau_val))
                    tau_real_magnitudes[n, m, i] = np.abs(np.real(tau_val))
                    
                    # Track largest values
                    if np.abs(np.imag(tau_val)) > largest_tau_imag:
                        largest_tau_imag = np.abs(np.imag(tau_val))
                        largest_tau_imag_loc = (n, m, i)

                # Accumulate γ (imaginary part of τ)
                if i > 0:
                    # Use trapezoidal rule for more accurate integration
                    gamma_increment = 0.5 * np.imag(tau_val + tau[n, m, i - 1]) * delta_theta
                    gamma[n, m, i] = gamma[n, m, i - 1] + gamma_increment
                    
                    # Store increment for diagnostics
                    if i < N_orig:
                        gamma_increments[n, m, i-1] = gamma_increment
                        
                        # Track largest increment
                        if np.abs(gamma_increment) > largest_gamma_increment:
                            largest_gamma_increment = np.abs(gamma_increment)
                            largest_gamma_increment_loc = (n, m, i-1)

    # Remove padded τ/γ at final point to keep shape = original
    tau = tau[:, :, :N_orig]
    gamma = gamma[:, :, :N_orig]
    
    # Print diagnostic information about large tau and gamma values
    print(f"\nDiagnostic Information:")
    print(f"Largest imaginary tau value: {largest_tau_imag:.6f} at (n={largest_tau_imag_loc[0]+1}, m={largest_tau_imag_loc[1]+1}, theta_idx={largest_tau_imag_loc[2]})")
    print(f"Largest gamma increment: {largest_gamma_increment:.6f} at (n={largest_gamma_increment_loc[0]+1}, m={largest_gamma_increment_loc[1]+1}, theta_idx={largest_gamma_increment_loc[2]})")
    
    # Find the matrix elements with the largest contribution to the trace
    tau_imag_sum = np.sum(np.abs(np.imag(tau)), axis=2)
    gamma_final_abs = np.abs(gamma[:,:,-1])
    
    # Get the indices of the top 5 contributors
    tau_flat_indices = np.argsort(tau_imag_sum.flatten())[-5:]
    gamma_flat_indices = np.argsort(gamma_final_abs.flatten())[-5:]
    
    print("\nTop 5 contributors to tau (imaginary part sum):")
    for idx in tau_flat_indices[::-1]:
        n, m = np.unravel_index(idx, tau_imag_sum.shape)
        print(f"  Tau[{n+1},{m+1}]: Sum of abs(imag) = {tau_imag_sum[n,m]:.6f}, Final value = {np.imag(tau[n,m,-1]):.6f}")
    
    print("\nTop 5 contributors to gamma (final values):")
    for idx in gamma_flat_indices[::-1]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        print(f"  Gamma[{n+1},{m+1}]: Final value = {gamma[n,m,-1]:.6f}")
    
    # Create diagnostic plots directory
    diag_dir = os.path.join('berry_phase_corrected_script', 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    # Plot the evolution of the largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        plt.plot(theta_vals[:-1], gamma[n,m,:], label=f'Gamma[{n+1},{m+1}]')
    plt.title('Evolution of Largest Gamma Contributors')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Gamma Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/largest_gamma_evolution.png')
    plt.close()
    
    # Plot the evolution of tau imaginary part for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in tau_flat_indices[-3:]:
        n, m = np.unravel_index(idx, tau_imag_sum.shape)
        plt.plot(theta_vals[:-1], np.imag(tau[n,m,:]), label=f'Imag(Tau[{n+1},{m+1}])')
    plt.title('Evolution of Largest Tau Imaginary Parts')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Imaginary Part of Tau')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/largest_tau_imag_evolution.png')
    plt.close()
    
    # Plot the gamma increments for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        # Calculate increments
        increments = np.diff(gamma[n,m,:])
        plt.plot(theta_vals[1:-1], increments, label=f'Gamma[{n+1},{m+1}] increments')
    plt.title('Gamma Increments for Largest Contributors')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Increment Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_increments.png')
    plt.close()
    
    # Plot a heatmap of the final gamma matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(gamma[:,:,-1], cmap='viridis')
    plt.colorbar(im, label='Gamma Value')
    plt.title('Final Gamma Matrix Heatmap')
    plt.xlabel('m')
    plt.ylabel('n')
    # Add text annotations
    for i in range(M):
        for j in range(M):
            text = plt.text(j, i, f'{gamma[i,j,-1]:.1f}',
                          ha="center", va="center", color="w" if abs(gamma[i,j,-1]) > 1000 else "k")
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_final_heatmap.png')
    plt.close()
    
    if continuity_check:
        print("\nEigenvector continuity (should be ~1):")
        for i in range(1, N_orig):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all[i - 1, :, n], eigvectors_all[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

    # Report total Berry phase per eigenstate
    gamma_closed_loop = gamma[:, :, -1]
    berry_phase_per_state = np.angle(np.exp(1j * gamma_closed_loop.diagonal()))
    print("\nBerry phase per state:", berry_phase_per_state)

    return tau, gamma


def main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0, extended=False):
    #space theta_vals uniformly
    theta_vals = theta_range = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range)
    H_thetas = hamiltonian.H_thetas()
    R_thetas = hamiltonian.R_thetas()
    
    
    #create a directory for the output
    output_dir = os.path.join(os.path.dirname(__file__), 'berry_phase_corrected_script')
    os.makedirs(output_dir, exist_ok=True)
    
    #create a directory for the plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    #create a directory for vectors
    vector_dir = os.path.join(output_dir, 'vectors')
    os.makedirs(vector_dir, exist_ok=True)
    
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)

    visualize_vectorz(R_0, d, num_points, theta_min, theta_max, vector_dir)
    
    eigenvectors = Eigenvectors(H_thetas, plot_dir, theta_vals)
    eigvecs = eigenvectors.compute()
    eigenvectors.plot(eigvecs)
    eigenvalues = Eigenvalues(H_thetas, plot_dir, theta_vals)
    eigenvalues.plot()
    
    with mp.Pool(processes=(mp.cpu_count()-1)) as pool:
        results = pool.apply_async(compute_berry_phase, (eigvecs, theta_vals))
        tau, gamma = results.get()
    #print("Tau:", tau)
    print("Gamma[:,:,-1]:\n", gamma[:,:,-1]) #print the last gamma matrix
    #create a report on the gamma matrix
    with open(f'{output_dir}/gamma_report.txt', "w") as f:
        f.write("Gamma matrix report:\n===========================================\n")
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                f.write(f"Gamma[{i+1},{j+1}]: {gamma[i,j,-1]}\n")
                f.write(f"Tau[{i+1},{j+1}]: {np.imag(tau[i,j,-1])}\n")
            f.write("\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(gamma[:,:,-1], "The last Berry Phase Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(np.imag(tau[:,:,-1]), "The last Tau Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(gamma[:,:,-2], "The one before last Berry Phase Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(np.imag(tau[:,:,-2]), "The one before last Tau Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")

    #print the gamma matrix
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            print(f"Gamma[{i+1},{j+1}]: {gamma[i,j,-1]}")
            print(f"Tau[{i+1},{j+1}]: {np.imag(tau[i,j,-1])}")
    

    #save the tau and gamma matrices
    np.save(os.path.join(npy_dir, 'tau.npy'), tau)
    np.save(os.path.join(npy_dir, 'gamma.npy'), gamma)

    #save the eigvecs
    np.save(os.path.join(npy_dir, 'eigvecs.npy'), eigvecs)

    #save the theta_vals
    np.save(os.path.join(npy_dir, 'theta_vals.npy'), theta_vals)


    #plot the gamma and tau matrices
    # Use the extended theta_vals or truncate tau and gamma to match original theta_vals
    if extended:
        # Option 1: Use extended theta_vals
        extended_theta_vals = np.append(theta_vals, theta_max)
        plot_matrix_elements(tau, gamma, extended_theta_vals, plot_dir)
    else:
        # Option 2: Truncate tau and gamma to match original theta_vals
        extended_theta_vals = theta_vals
        plot_matrix_elements(tau, gamma, extended_theta_vals, plot_dir)
    # Convert lists to numpy arrays
    Hamiltonians = np.array(H_thetas)
    Va_values = np.array(hamiltonian.Va_theta_vals(R_thetas))
    Vx_values = np.array(hamiltonian.Vx_theta_vals(R_thetas))

    #plot the substract of Vx-Va
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals, (Vx_values[:, i] + omega * hbar) - Va_values[:, i], label=f'Vx[{i+1}] - Va[{i+1}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Vx - Va Components')
    plt.title('Vx - Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Vx_minus_Va_components.png')
    print("Vx - Va plots saved to figures directory.")
    plt.close()

    save_and__visalize_va_and_vx(npy_dir, Hamiltonians, Va_values, Vx_values, theta_vals, plot_dir)
    
    #write the dataset to a file
    with open(f'{output_dir}/dataset.arg', 'w') as f:
        f.write(f'd = {d}\n')
        f.write(f'aVx = {aVx}\n')
        f.write(f'aVa = {aVa}\n')
        f.write(f'c_const = {c_const}\n')
        f.write(f'x_shift = {x_shift}\n')
        f.write(f'theta_min = {theta_min}\n')
        f.write(f'theta_max = {theta_max}\n')
        f.write(f'omega = {omega}\n')
        f.write(f'num_points = {num_points}\n')
        f.write(f'R_0 = {R_0}\n')

    # Get the number of states (M) from the gamma matrix shape
    M = gamma.shape[0]
    
    # Plot the trace of gamma and individual diagonal elements
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 subplot layout
    plt.subplot(2, 2, 1)
    # Plot the raw trace
    raw_trace = np.trace(gamma, axis1=0, axis2=1)
    # Make sure theta_vals and raw_trace have the same shape
    plt.plot(theta_vals[:len(raw_trace)], raw_trace, 'b-', linewidth=2, label='Raw Trace')
    plt.title("Raw Trace of γ Matrix")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Trace Value')
    plt.grid(True)
    plt.legend()
    
    # Plot the individual diagonal elements
    plt.subplot(2, 2, 2)
    for i in range(M):
        plt.plot(theta_vals[:len(gamma[i,i,:])], gamma[i, i, :], label=f'γ[{i+1},{i+1}]')
    plt.title("Diagonal Elements of γ Matrix")
    plt.xlabel('Theta (θ)')
    plt.ylabel('γ Value')
    plt.grid(True)
    plt.legend()
    
    # Plot the Berry phases for each state
    plt.subplot(2, 2, 3)
    berry_phases = np.zeros((M, gamma.shape[2]))
    for i in range(gamma.shape[2]):
        for j in range(M):
            berry_phases[j, i] = np.angle(np.exp(1j * gamma[j, j, i]))
    
    for i in range(M):
        plt.plot(theta_vals[:gamma.shape[2]], berry_phases[i, :], label=f'State {i+1}')
    plt.title("Berry Phase per State")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Berry Phase')
    plt.grid(True)
    plt.legend()
    
    # Plot the sum of Berry phases
    plt.subplot(2, 2, 4)
    sum_berry_phases = np.sum(berry_phases, axis=0)
    plt.plot(theta_vals[:gamma.shape[2]], sum_berry_phases, 'r-', linewidth=2)
    plt.title("Sum of Berry Phases")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Sum Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/gamma_analysis.png')
    plt.close()
    
    # Print some debug information about the trace
    print(f"Trace of gamma shape: {raw_trace.shape}")
    print(f"Extended theta_vals shape: {extended_theta_vals.shape}")
    print(f"First few trace values: {raw_trace[:5]}")
    print(f"Last few trace values: {raw_trace[-5:]}")
    print(f"Trace at 0, π/2, π, 3π/2, 2π: {raw_trace[0]}, {raw_trace[len(raw_trace)//4]}, {raw_trace[len(raw_trace)//2]}, {raw_trace[3*len(raw_trace)//4]}, {raw_trace[-1]}")
    
    # Calculate Berry phase from trace
    berry_phase_from_trace = raw_trace[-1] - raw_trace[0]
    print(f"Berry phase calculated from trace: {berry_phase_from_trace}")
    
    # Also create a plot showing just the trace with annotations
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals[:len(raw_trace)], raw_trace, 'b-', linewidth=2)
    
    # Add markers and annotations at key points
    key_indices = [0, len(raw_trace)//4, len(raw_trace)//2, 3*len(raw_trace)//4, -2]
    
    plt.scatter([theta_vals[i] for i in key_indices], 
                [raw_trace[i] for i in key_indices], 
                color='red', s=80, zorder=5)
    
    # Add annotations
    for i, idx in enumerate(key_indices):
        plt.annotate(f'{raw_trace[idx]:.6f}', 
                    (theta_vals[idx], raw_trace[idx]), 
                    textcoords="offset points", 
                    xytext=(0, 10 if i % 2 == 0 else -20), 
                    ha='center')
    
    # Also add a point at the very end to show the jump
    plt.scatter([theta_vals[-2]], [raw_trace[-1]], color='green', s=100, zorder=5)
    plt.annotate(f'{raw_trace[-1]:.6f}', 
                (theta_vals[-2], raw_trace[-1]), 
                textcoords="offset points", 
                xytext=(30, 0), 
                ha='left', 
                arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.title("Trace of γ Matrix vs θ")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Trace Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/trace_gamma_vs_theta_annotated.png')
    plt.close()
    
    # Let's simplify the plotting code to just focus on what's important
    # Create a plot showing the trace of gamma
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals[:len(raw_trace)], raw_trace, linewidth=2, label='Trace of γ')
    plt.title("Tr(γ) vs θ")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Tr_gamma_vs_theta_full.png')
    plt.close()
    
    # Add annotations to the trace plot
    key_indices = [0, len(raw_trace)//4, len(raw_trace)//2, 3*len(raw_trace)//4, -1]
    plt.scatter([theta_vals[i] for i in key_indices], 
                [raw_trace[i] for i in key_indices], 
                color='red', s=80, zorder=5)
    
    # Add annotations
    for i, idx in enumerate(key_indices):
        plt.annotate(f'{raw_trace[idx]:.2f}', 
                     (theta_vals[idx], raw_trace[idx]), 
                     textcoords="offset points", 
                     xytext=(0, 10 if i % 2 == 0 else -20), 
                     ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Tr_gamma_vs_theta.png')
    
    # Also plot just the beginning and end to see if there's a jump
    plt.figure(figsize=(12, 6))
    # Plot first 5% and last 5% of the data
    cutoff = int(len(raw_trace) * 0.05)
    
    plt.subplot(1, 2, 1)
    plt.plot(theta_vals[:cutoff], raw_trace[:cutoff], linewidth=2)
    plt.title("Start of Tr(γ) vs θ")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(theta_vals[-cutoff-1:-1], raw_trace[-cutoff:], linewidth=2)
    plt.title("End of Tr(γ) vs θ")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Tr_gamma_vs_theta_endpoints.png')
    plt.close()

    with open(f'{output_dir}/overlap_report.txt', 'w') as f:
        f.write("Overlap report:\n")
        for i in range(1, len(eigvecs)):
            for n in range(eigvecs.shape[2]):
                overlap = np.vdot(eigvecs[i - 1, :, n], eigvecs[i, :, n])
                f.write(f"Overlap (state {n+1}, θ={i}): {overlap:.6f}\n")
    
    
    plt.figure(figsize=(12, 6))
    # Use extended_theta_vals to match dimensions with tau
    plt.plot(extended_theta_vals, np.imag(tau[1,2,:]), label="Im(τ₁₂)")
    plt.scatter(extended_theta_vals[0], np.imag(tau[1,2,0]), color='green', label="First Point")
    plt.scatter(extended_theta_vals[-1], np.imag(tau[1,2,-1]), color='red', label="Last Point")
    plt.legend()
    plt.title("Tau continuity check (should be smooth)")
    plt.savefig(f'{plot_dir}/tau_continuity_check.png')
    plt.close()


    return tau, gamma, eigvecs, theta_vals

if __name__ == '__main__':
    dataset = 2

    if dataset == 1:
        d = 0.001 #radius of the circle
        aVx = 1.0
        aVa = 5.0
        c_const = 0.1  # Potential constant, shifts the 2d parabola on the y axis
        x_shift = 0.1  # Shift in x direction
        theta_min = 0
        theta_max = 2 * np.pi
        omega = 0.1
        num_points = 50000
        R_0 = (0, 0, 0)
    
    elif dataset == 2:
        #let a be an aVx and an aVa parameter
        d = 0.02  # Radius of the circle, use unit circle for bigger radius, még egy CI???
        aVx = 1.0
        aVa = 3.0
        c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
        x_shift = 0.01  # Shift in x direction
        theta_min = 0
        theta_max = 2 * np.pi
        omega = 0.1
        num_points = 5000
        R_0 = (0, 0, 0)

    
    
    #run the main function
    main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0, extended=False)
    