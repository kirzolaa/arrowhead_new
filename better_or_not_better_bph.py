import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a display
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
import time
import psutil
import json
import platform
import datetime
import subprocess
import humanize
import copy
try:
    import torch
except ImportError:
    torch = None
    print("torch not found")
    
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

def format_matrix(isgamma=False, matrix=None, title=None, output_dir=None):
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
    if isgamma:
        for i in range(n):
            row = "    |  "
            for j in range(m):
                if i == j:
                    # Diagonal elements (γ_nn)
                    row += f"{matrix[i,j]:.4f}π"
                else:
                    # Off-diagonal elements (γ_nm)
                    row += f"{matrix[i,j]:.4f}π"
                if j < m - 1:
                    row += "  "
                else:
                    row += "    |"
            lines.append(row)
    else:
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
        
def compute_berry_phase_og(eigvectors_all, theta_vals, continuity_check=False, OUT_DIR=None):
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

    # Create arrays to track the magnitude of tau and gamma for diagnostics
    tau_imag_magnitudes = np.zeros((M, M, N))
    tau_real_magnitudes = np.zeros((M, M, N))
    gamma_increments = np.zeros((M, M, N))

    # Optional: normalize eigenvectors for safety
    eigvectors_all = eigvectors_all / np.linalg.norm(eigvectors_all, axis=1, keepdims=True)

    # Track the largest tau values and their locations
    largest_tau_imag = 0
    largest_tau_imag_loc = (0, 0, 0)
    largest_gamma_increment = 0
    largest_gamma_increment_loc = (0, 0, 0)

    for n in range(M):
        for m in range(M):
            for i in range(N): 
                # Handle boundary conditions for the finite difference
                # Inside compute_berry_phase

                im1 = i - 1
                ip1 = i + 1

                if i == 0: # at the start of the loop, 0 and 2pi has mutual geometry, so we add one to i_prev
                    im1=-2
                if i == N - 1: # at the end of the loop, 2pi and 0 has mutual geometry, so we add one to i_next
                    ip1 = 1
                    
                psi_prev = eigvectors_all[im1, :, n] # (THETA, COMPONENT, STATE)
                psi_next = eigvectors_all[ip1, :, n]

                delta_theta_for_grad = theta_vals[2] - theta_vals[0]
                
                if ip1 - im1 != 2:
                    if np.vdot(psi_prev, psi_next).real < 0:
                        print("Negative overlap between previous and next eigenvector at theta index", im1, "and", ip1)
                        if i == 0:
                            psi_prev = -1 * copy.deepcopy(psi_prev)
                        else:
                            psi_next = -1 * copy.deepcopy(psi_next)
                
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

                
                # Store magnitudes for diagnostics
                if i < N:
                    tau_imag_magnitudes[n, m, i] = np.abs(np.imag(tau_val))
                    tau_real_magnitudes[n, m, i] = np.abs(np.real(tau_val))
                    
                    # Track largest values
                    if np.abs(np.imag(tau_val)) > largest_tau_imag:
                        largest_tau_imag = np.abs(np.imag(tau_val))
                        largest_tau_imag_loc = (n, m, i)

                # Accumulate γ (imaginary part of τ)
                if i > 0:
                    # Use trapezoidal rule for more accurate integration
                    gamma_increment = 0.5 * np.real(tau_val + tau[n, m, i - 1]) * delta_theta_for_grad
                    gamma[n, m, i] = gamma[n, m, i - 1] + gamma_increment
                    
                    # Store increment for diagnostics
                    if i < N:
                        gamma_increments[n, m, i-1] = gamma_increment
                        
                        # Track largest increment
                        if np.abs(gamma_increment) > largest_gamma_increment:
                            largest_gamma_increment = np.abs(gamma_increment)
                            largest_gamma_increment_loc = (n, m, i-1)

    

    # Print diagnostic information about large tau and gamma values
    print(f"\nDiagnostic Information:")
    print(f"Largest imaginary tau value: {largest_tau_imag:.6f} at (n={largest_tau_imag_loc[0]+1}, m={largest_tau_imag_loc[1]+1}, theta_idx={largest_tau_imag_loc[2]})")
    print(f"Largest gamma increment: {largest_gamma_increment:.6f} at (n={largest_gamma_increment_loc[0]+1}, m={largest_gamma_increment_loc[1]+1}, theta_idx={largest_gamma_increment_loc[2]})")
    
    # Find the matrix elements with the largest contribution to the trace
    tau_imag_sum = np.sum(np.abs(tau), axis=2)
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
    diag_dir = os.path.join(OUT_DIR, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    # Plot the evolution of the largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        plt.plot(theta_vals[1:]/np.pi, gamma[n,m,1:], label=f'Gamma[{n+1},{m+1}]')
    plt.title('Evolution of Largest Gamma Contributors')
    plt.xlabel('Theta (θ/π)')
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
        plt.plot(theta_vals[1:]/np.pi, np.imag(tau[n,m,1:]), label=f'Imag(Tau[{n+1},{m+1}])')
    plt.title('Evolution of Largest Tau Imaginary Parts')
    plt.xlabel('Theta (θ/π)')
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
        increments = np.diff(gamma[n,m,1:])
        plt.plot(theta_vals[1:-1]/np.pi, increments, label=f'Gamma[{n+1},{m+1}] increments')
    plt.title('Gamma Increments for Largest Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Increment Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_increments.png')
    plt.close()
    
    # Plot a heatmap of the final gamma matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(gamma[:,:,-1], cmap='coolwarm')
    plt.colorbar(im, label='Gamma Value')
    plt.title('Final Gamma Matrix Heatmap')
    plt.xlabel('m')
    plt.ylabel('n')
    # DISPLAY THE INDEXES ON THE PLOT
    plt.xticks(range(M), range(1, M+1))
    plt.yticks(range(M), range(1, M+1))
    # Add text annotations
    for i in range(M):
        for j in range(M):
            text = plt.text(j, i, f'{gamma[i,j,-1]:.3f}',
                          ha="center", va="center", color="w" if abs(gamma[i,j,-1]) > 1000 else "k")
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_final_heatmap.png')
    plt.close()
    
    final_plotz_from_bph_func_folder = os.path.join(OUT_DIR, 'final_plots_from_bph_func')
    os.makedirs(final_plotz_from_bph_func_folder, exist_ok=True)
    # /np.pi and plot these diagnostics plots
    # Plot the evolution of the largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        plt.plot(theta_vals[1:]/np.pi, gamma[n,m,1:], label=f'Gamma[{n+1},{m+1}]')
    plt.title('Evolution of Largest Gamma Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Gamma Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/largest_gamma_evolution.png')
    plt.close()
    
    # Plot the evolution of tau imaginary part for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in tau_flat_indices[-3:]:
        n, m = np.unravel_index(idx, tau_imag_sum.shape)
        plt.plot(theta_vals[1:]/np.pi, np.imag(tau[n,m,1:]), label=f'Imag(Tau[{n+1},{m+1}])')
    plt.title('Evolution of Largest Tau Imaginary Parts')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Imaginary Part of Tau')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/largest_tau_imag_evolution.png')
    plt.close()
    
    # Plot the gamma increments for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        # Calculate increments
        increments = np.diff(gamma[n,m,1:])
        plt.plot(theta_vals[1:-1]/np.pi, increments, label=f'Gamma[{n+1},{m+1}] increments')
    plt.title('Gamma Increments for Largest Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Increment Value')
    plt.grid(True)
    plt.legend(fontsize=10,loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/gamma_increments.png')
    plt.close()
    
    # Plot a heatmap of the final gamma matrix
    plt.figure(figsize=(12, 10))
    
    # Convert gamma values to multiples of pi
    gamma_pi = gamma[:,:,-1]/np.pi
    im = plt.imshow(gamma_pi, cmap='coolwarm')
    
    # Create colorbar with π units
    cbar = plt.colorbar(im, label='Gamma Value (×π)')
    cbar.set_ticks([-2, -1, -0.5, 0, 0.5, 1, 2])  # Common fractions of pi
    cbar.set_ticklabels(['-2π', '-π', '-π/2', '0', 'π/2', 'π', '2π'])
    
    plt.title('Final Gamma Matrix Heatmap')
    plt.xlabel('m')
    plt.ylabel('n')
    # DISPLAY THE INDEXES ON THE PLOT
    plt.xticks(range(M), range(1, M+1))
    plt.yticks(range(M), range(1, M+1))
    
    # Function to format values as fractions of pi
    def format_pi(val):
        if val == 0:
            return '0'
        
        # Common fractions
        fractions = {
            1.0: 'π',
            -1.0: '-π',
            0.5: 'π/2',
            -0.5: '-π/2',
            0.333: 'π/3',
            -0.333: '-π/3',
            0.666: '2π/3',
            -0.666: '-2π/3',
            0.25: 'π/4',
            -0.25: '-π/4',
            0.75: '3π/4',
            -0.75: '-3π/4',
            2.0: '2π',
            -2.0: '-2π'
        }
        
        # Check for common fractions first
        for frac, label in fractions.items():
            if abs(val - frac) < 0.001:  # Allow for small floating point errors
                return label.format(val) + 'π'
        
        # For other values, show as decimal * π
        return f'{val:.3f}'
    
    # Add text annotations with formatted values
    for i in range(M):
        for j in range(M):
            val = gamma_pi[i,j]
            text_label = format_pi(val)
            text = plt.text(j, i, text_label,
                          ha="center", va="center", 
                          color="w" if abs(val) > 0.5 else "k",
                          fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/gamma_final_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if continuity_check:
        print("\nEigenvector continuity (should be ~1):")
        for i in range(1, N_orig):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all[i - 1, :, n], eigvectors_all[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

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

def compute_berry_phase(eigvectors_all, theta_vals, continuity_check=False, OUT_DIR=None):
    """
    Compute Berry connection (τ) and Berry phase (γ) matrices using central difference,
    with ring continuity and stabilized diagonals.
    """

    N_orig, M, _ = eigvectors_all.shape
    output_dir = OUT_DIR
    # Extend θ and eigvecs for periodic boundary
    #eigvectors_all = eigvectors_all.copy() # np.concatenate([eigvectors_all, eigvectors_all[:1]], axis=0)
    #theta_vals = theta_vals.copy() #np.append(theta_vals, theta_vals[0] + 2 * np.pi)
    N = N_orig # + 1
    delta_theta = theta_vals[1] - theta_vals[0]

    tau = np.zeros((M, M, N), dtype=np.complex128)
    gamma = np.zeros((M, M, N), dtype=np.float64)
    
    # Create arrays to track the magnitude of tau and gamma for diagnostics
    tau_imag_magnitudes = np.zeros((M, M, N_orig))
    tau_real_magnitudes = np.zeros((M, M, N_orig))
    gamma_increments = np.zeros((M, M, N_orig))

    # Optional: normalize eigenvectors for safety
    #eigvectors_all = eigvectors_all / np.linalg.norm(eigvectors_all, axis=1, keepdims=True)

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

                im1 = i - 1
                ip1 = i + 1

                if i == 0: # at the start of the loop, 0 and 2pi has mutual geometry, so we add one to i_prev
                    im1=-2
                if i == N - 1: # at the end of the loop, 2pi and 0 has mutual geometry, so we add one to i_next
                    ip1 = 1
                
                
                
                psi_prev = eigvectors_all[im1, :, n] # (THETA, COMPONENT, STATE)
                psi_next = eigvectors_all[ip1, :, n]
                delta_theta_for_grad = theta_vals[2] - theta_vals[0]

                if ip1 - im1 != 2:
                    if np.vdot(psi_prev, psi_next).real < 0:
                        print("Negative overlap between previous and next eigenvector at theta index", im1, "and", ip1)
                        if i == 0:
                            psi_prev = -1 * copy.deepcopy(psi_prev)
                        else:
                            psi_next = -1 * copy.deepcopy(psi_next)

                psi_curr = eigvectors_all[i, :, m]

                grad_psi = (psi_next - psi_prev) / delta_theta_for_grad
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                
                # Zero out diagonal elements of tau
                # This is a common practice in Berry phase calculations
                #if n == m:
                #    tau_val = 0.0 AI MADE THIS CHANGE
                """
                # Check for unusually large imaginary values at the end of the loop
                # This prevents the sudden jump in gamma at 2π
                if i == N-1 and np.abs(np.imag(tau_val)) > 10.0:
                    # Use the average of previous values instead
                    if i > 0:
                        prev_tau = tau[n, m, i-1]
                        tau_val = prev_tau
                        print(f"Limiting large tau value at (n={n+1}, m={m+1}, theta_idx={i})")
                """
                tau[n, m, i] = tau_val
                
                # Store magnitudes for diagnostics
                if i < N:
                    tau_imag_magnitudes[n, m, i] = np.abs(np.imag(tau_val))
                    tau_real_magnitudes[n, m, i] = np.abs(np.real(tau_val))
                    
                    # Track largest values
                    if np.abs(np.imag(tau_val)) > largest_tau_imag:
                        largest_tau_imag = np.abs(np.imag(tau_val))
                        largest_tau_imag_loc = (n, m, i)

                # Accumulate γ (imaginary part of τ)
                if i >= 0:
                    # USE riemann sum
                    #gamma_increment = np.imag(tau[n, m, i]) * delta_theta
                    #gamma[n, m, i] = gamma[n, m, i-1] + gamma_increment
                    
                    # Use trapezoidal rule for more accurate integration
                    gamma_increment = 0.5 * (np.imag(tau_val) + np.imag(tau[n, m, i - 1])) * delta_theta_for_grad
                    gamma[n, m, i] = gamma[n, m, i - 1] + gamma_increment
                    
                    # Store increment for diagnostics
                    if i < N:
                        gamma_increments[n, m, i-1] = gamma_increment
                        
                        # Track largest increment
                        if np.abs(gamma_increment) > largest_gamma_increment:
                            largest_gamma_increment = np.abs(gamma_increment)
                            largest_gamma_increment_loc = (n, m, i-1)

    # Remove padded τ/γ at final point to keep shape = original
    #tau = tau[:, :, :N_orig]
    #gamma = gamma[:, :, :N_orig]
    
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
    diag_dir = os.path.join(OUT_DIR, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    # Plot the evolution of the largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        plt.plot(theta_vals[1:]/np.pi, gamma[n,m,1:], label=f'Gamma[{n+1},{m+1}]')
    plt.title('Evolution of Largest Gamma Contributors')
    plt.xlabel('Theta (θ/π)')
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
        plt.plot(theta_vals[1:]/np.pi, np.imag(tau[n,m,1:]), label=f'Imag(Tau[{n+1},{m+1}])')
    plt.title('Evolution of Largest Tau Imaginary Parts')
    plt.xlabel('Theta (θ/π)')
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
        increments = np.diff(gamma[n,m,1:])
        plt.plot(theta_vals[1:-1]/np.pi, increments, label=f'Gamma[{n+1},{m+1}] increments')
    plt.title('Gamma Increments for Largest Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Increment Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_increments.png')
    plt.close()
    
    # Plot a heatmap of the final gamma matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(gamma[:,:,-1], cmap='coolwarm')
    plt.colorbar(im, label='Gamma Value')
    plt.title('Final Gamma Matrix Heatmap')
    plt.xlabel('m')
    plt.ylabel('n')
    # DISPLAY THE INDEXES ON THE PLOT
    plt.xticks(range(M), range(1, M+1))
    plt.yticks(range(M), range(1, M+1))
    # Add text annotations
    for i in range(M):
        for j in range(M):
            text = plt.text(j, i, f'{gamma[i,j,-1]:.3f}',
                          ha="center", va="center", color="w" if abs(gamma[i,j,-1]) > 1000 else "k")
    plt.tight_layout()
    plt.savefig(f'{diag_dir}/gamma_final_heatmap.png')
    plt.close()
    
    final_plotz_from_bph_func_folder = os.path.join(OUT_DIR, 'final_plots_from_bph_func')
    os.makedirs(final_plotz_from_bph_func_folder, exist_ok=True)
    # /np.pi and plot these diagnostics plots
    # Plot the evolution of the largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        plt.plot(theta_vals[1:]/np.pi, gamma[n,m,1:], label=f'Gamma[{n+1},{m+1}]')
    plt.title('Evolution of Largest Gamma Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Gamma Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/largest_gamma_evolution.png')
    plt.close()
    
    # Plot the evolution of tau imaginary part for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in tau_flat_indices[-3:]:
        n, m = np.unravel_index(idx, tau_imag_sum.shape)
        plt.plot(theta_vals[1:]/np.pi, np.imag(tau[n,m,1:]), label=f'Imag(Tau[{n+1},{m+1}])')
    plt.title('Evolution of Largest Tau Imaginary Parts')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Imaginary Part of Tau')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/largest_tau_imag_evolution.png')
    plt.close()
    
    # Plot the gamma increments for largest contributors
    plt.figure(figsize=(12, 8))
    for idx in gamma_flat_indices[-3:]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        # Calculate increments
        increments = np.diff(gamma[n,m,1:])
        plt.plot(theta_vals[1:-1]/np.pi, increments, label=f'Gamma[{n+1},{m+1}] increments')
    plt.title('Gamma Increments for Largest Contributors')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Increment Value')
    plt.grid(True)
    plt.legend(fontsize=10,loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/gamma_increments.png')
    plt.close()
    
    # Plot a heatmap of the final gamma matrix
    plt.figure(figsize=(12, 10))
    
    # Convert gamma values to multiples of pi
    gamma_pi = gamma[:,:,-1]/np.pi
    im = plt.imshow(gamma_pi, cmap='coolwarm')
    
    # Create colorbar with π units
    cbar = plt.colorbar(im, label='Gamma Value (×π)')
    cbar.set_ticks([-2, -1, -0.5, 0, 0.5, 1, 2])  # Common fractions of pi
    cbar.set_ticklabels(['-2π', '-π', '-π/2', '0', 'π/2', 'π', '2π'])
    
    plt.title('Final Gamma Matrix Heatmap')
    plt.xlabel('m')
    plt.ylabel('n')
    # DISPLAY THE INDEXES ON THE PLOT
    plt.xticks(range(M), range(1, M+1))
    plt.yticks(range(M), range(1, M+1))
    
    # Function to format values as fractions of pi
    def format_pi(val):
        if val == 0:
            return '0'
        
        # Common fractions
        fractions = {
            1.0: 'π',
            -1.0: '-π',
            0.5: 'π/2',
            -0.5: '-π/2',
            0.333: 'π/3',
            -0.333: '-π/3',
            0.666: '2π/3',
            -0.666: '-2π/3',
            0.25: 'π/4',
            -0.25: '-π/4',
            0.75: '3π/4',
            -0.75: '-3π/4',
            2.0: '2π',
            -2.0: '-2π'
        }
        
        # Check for common fractions first
        for frac, label in fractions.items():
            if abs(val - frac) < 0.001:  # Allow for small floating point errors
                return label.format(val)
        
        # For other values, show as decimal * π
        return f'{val:.3f}'
    
    # Add text annotations with formatted values
    for i in range(M):
        for j in range(M):
            val = gamma_pi[i,j]
            text_label = format_pi(val)
            text = plt.text(j, i, text_label,
                          ha="center", va="center", 
                          color="w" if abs(val) > 0.5 else "k",
                          fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{final_plotz_from_bph_func_folder}/gamma_final_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if continuity_check:
        print("\nEigenvector continuity (should be ~1):")
        for i in range(1, N_orig):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all[i - 1, :, n], eigvectors_all[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

    return tau, gamma


def main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0, extended=False):
    start_time = time.time()
    
    #space theta_vals uniformly
    theta_vals = theta_range = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range)
    H_thetas = hamiltonian.H_thetas()
    R_thetas = hamiltonian.R_thetas()
    
    
    #create a directory for the output
    output_dir = os.path.join(os.path.dirname(__file__), f'berry_phase_AROUND_d_{d:.8f}')
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
        results = pool.apply_async(compute_berry_phase, (eigvecs, theta_vals, False, output_dir))
        tau, gamma = results.get()


    #print("Tau:", tau)
    print("Gamma[:,:,-1]:\n", gamma[:,:,-1]) #print the last gamma matrix
    #create a report on the gamma matrix
    with open(f'{output_dir}/gamma_report.txt', "w") as f:
        f.write("Gamma matrix report:\n===========================================\n")
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                f.write(f"Gamma[{i+1},{j+1}]: {gamma[i,j,-1]/np.pi}π\n")
                f.write(f"Tau[{i+1},{j+1}]: {np.imag(tau[i,j,-1])}\n")
            f.write("\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(isgamma=True,matrix=gamma[:,:,-1]/np.pi, title="The last Berry Phase Matrix", output_dir=output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(isgamma=False,matrix=np.imag(tau[:,:,-1]), title="The last Tau Matrix", output_dir=output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(isgamma=True,matrix=gamma[:,:,-2]/np.pi, title="The one before last Berry Phase Matrix", output_dir=output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(isgamma=False,matrix=np.imag(tau[:,:,-2]), title="The one before last Tau Matrix", output_dir=output_dir))
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
        plt.plot(theta_vals/np.pi, (Vx_values[:, i] + omega * hbar) - Va_values[:, i], label=f'Vx[{i+1}] - Va[{i+1}]')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Vx + omega*hbar - Va Components')
    plt.title('Vx + omega*hbar - Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Vx_plus_omega_hbar_minus_Va_components.png')
    print("Vx - Va plots saved to figures directory.")
    plt.close()


    #plot the substract of Vx-Va
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals/np.pi, -(Vx_values[:, i] + omega * hbar) + Va_values[:, i], label=f'Vx[{i+1}] - Va[{i+1}]')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Va - Vx Components + omega*hbar')
    plt.title('Va - Vx Components + omega*hbar vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Va_minus_Vx_components_plus_omega_hbar.png')
    print("Va - Vx plots saved to figures directory.")
    plt.close()

    #plot the differences withouut the hbar*omega
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals/np.pi, -(Vx_values[:, i]) + Va_values[:, i], label=f'Vx[{i+1}] - Va[{i+1}]')
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Va - Vx Components')
    plt.title('Va - Vx Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Va_minus_Vx_components.png')
    print("Va - Vx plots saved to figures directory.")
    plt.close()

    #plot the differences withouut the hbar*omega
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals/np.pi, (Vx_values[:, i]) - Va_values[:, i], label=f'Vx[{i+1}] - Va[{i+1}]')
    plt.xlabel('Theta (θ/π)')
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
    plt.plot(theta_vals[:len(raw_trace)]/np.pi, raw_trace, 'b-', linewidth=2, label='Raw Trace')
    plt.title("Raw Trace of γ Matrix")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Trace Value')
    plt.grid(True)
    plt.legend()
    
    # Plot the individual diagonal elements
    plt.subplot(2, 2, 2)
    for i in range(M):
        plt.plot(theta_vals[:len(gamma[i,i,:])]/np.pi, gamma[i, i, :], label=f'γ[{i+1},{i+1}]')
    plt.title("Diagonal Elements of γ Matrix")
    plt.xlabel('Theta (θ/π)')
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
        plt.plot(theta_vals[:gamma.shape[2]]/np.pi, berry_phases[i, :], label=f'State {i+1}')
    plt.title("Berry Phase per State")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Berry Phase')
    plt.grid(True)
    plt.legend()
    
    # Plot the sum of Berry phases
    plt.subplot(2, 2, 4)
    sum_berry_phases = np.sum(berry_phases, axis=0)
    plt.plot(theta_vals[:gamma.shape[2]]/np.pi, sum_berry_phases, 'r-', linewidth=2)
    plt.title("Sum of Berry Phases")
    plt.xlabel('Theta (θ/π)')
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
    print(f"Berry phase matrix calculated trace: {berry_phase_from_trace}")
    
    # Also create a plot showing just the trace with annotations
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals[:len(raw_trace)]/np.pi, raw_trace, 'b-', linewidth=2)
    
    # Add markers and annotations at key points
    key_indices = [0, len(raw_trace)//4, len(raw_trace)//2, 3*len(raw_trace)//4, -2]
    
    plt.scatter([theta_vals[i]/np.pi for i in key_indices], 
                [raw_trace[i] for i in key_indices], 
                color='red', s=80, zorder=5)
    
    # Add annotations
    for i, idx in enumerate(key_indices):
        plt.annotate(f'{raw_trace[idx]:.6f}', 
                    (theta_vals[idx]/np.pi, raw_trace[idx]), 
                    textcoords="offset points", 
                    xytext=(0, 10 if i % 2 == 0 else -20), 
                    ha='center')
    
    # Also add a point at the very end to show the jump
    plt.scatter([theta_vals[-2]/np.pi], [raw_trace[-1]], color='green', s=100, zorder=5)
    plt.annotate(f'{raw_trace[-1]:.6f}', 
                (theta_vals[-2]/np.pi, raw_trace[-1]), 
                textcoords="offset points", 
                xytext=(30, 0), 
                ha='left', 
                arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.title("Trace of γ Matrix vs θ")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Trace Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/trace_gamma_vs_theta_annotated.png')
    plt.close()
    
    # Let's simplify the plotting code to just focus on what's important
    # Create a plot showing the trace of gamma
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals[:len(raw_trace)]/np.pi, raw_trace, linewidth=2, label='Trace of γ')
    plt.title("Tr(γ) vs θ")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Tr_gamma_vs_theta_full.png')
    plt.close()
    
    # Add annotations to the trace plot
    key_indices = [0, len(raw_trace)//4, len(raw_trace)//2, 3*len(raw_trace)//4, -1]
    plt.scatter([theta_vals[i]/np.pi for i in key_indices], 
                [raw_trace[i] for i in key_indices], 
                color='red', s=80, zorder=5)
    
    # Add annotations
    for i, idx in enumerate(key_indices):
        plt.annotate(f'{raw_trace[idx]:.2f}', 
                     (theta_vals[idx]/np.pi, raw_trace[idx]), 
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
    plt.plot(theta_vals[:cutoff]/np.pi, raw_trace[:cutoff], linewidth=2)
    plt.title("Start of Tr(γ) vs θ")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(theta_vals[-cutoff-1:-1]/np.pi, raw_trace[-cutoff:], linewidth=2)
    plt.title("End of Tr(γ) vs θ")
    plt.xlabel('Theta (θ/π)')
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
    plt.plot(extended_theta_vals/np.pi, np.imag(tau[1,2,:]), label="Im(τ₁₂)")
    plt.scatter(extended_theta_vals[0]/np.pi, np.imag(tau[1,2,0]), color='green', label="First Point")
    plt.scatter(extended_theta_vals[-1]/np.pi, np.imag(tau[1,2,-1]), color='red', label="Last Point")
    plt.legend()
    plt.title("Tau continuity check (should be smooth)")
    plt.xlabel('Theta (θ/π)')
    plt.ylabel('Im(τ₁₂)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/tau_continuity_check.png')
    plt.close()

    #put it here: 
    stat_dir = os.path.join(output_dir, 'statistics')
    os.makedirs(stat_dir, exist_ok=True)

    # --- System resource and runtime statistics ---
    
    stats = {}
    stats['platform'] = platform.platform()
    stats['python_version'] = platform.python_version()
    
    # Memory info
    vm = psutil.virtual_memory()
    stats['memory'] = {
        'total': vm.total,
        'available': vm.available,
        'percent': vm.percent,
        'used': vm.used,
        'free': vm.free
    }

    # CPU info
    stats['cpu'] = {
        'count': psutil.cpu_count(),
        'percent': psutil.cpu_percent(interval=1),
        'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }

    # Runtime info (if you want to measure total runtime, wrap main() calls with time.time())
    # Here, just save current timestamp
    stats['Runned at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    current_time = time.time()
    elapsed = current_time - start_time
    stats['runtime'] = str(datetime.timedelta(seconds=elapsed))
    stats['runtime_seconds'] = elapsed  # also save the raw float if you want
    
    # GPU/VRAM/Temperature (try torch, fallback to nvidia-smi, else skip)
    gpu_stats = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_stats.append({
                    'name': torch.cuda.get_device_name(i),
                    'vram_total': torch.cuda.get_device_properties(i).total_memory,
                    'vram_allocated': torch.cuda.memory_allocated(i),
                    'vram_reserved': torch.cuda.memory_reserved(i),
                    'vram_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                    'temperature': torch.cuda.get_device_properties(i).temperature if hasattr(torch.cuda.get_device_properties(i), 'temperature') else None
                })
    except ImportError:
        # Try nvidia-smi via subprocess
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                name, mem_total, mem_used, mem_free, temp = line.split(', ')
                gpu_stats.append({
                    'name': name,
                    'vram_total': int(mem_total)*1024*1024,
                    'vram_used': int(mem_used)*1024*1024,
                    'vram_free': int(mem_free)*1024*1024,
                    'temperature': float(temp)
                })
        except Exception:
            gpu_stats = None
    except Exception:
        gpu_stats = None
    stats['gpu'] = gpu_stats

    # Temperatures (CPU, etc.)
    try:
        temps = psutil.sensors_temperatures()
        stats['temperatures'] = {k: [t._asdict() for t in v] for k, v in temps.items()}
    except Exception:
        stats['temperatures'] = None
    
    # output_dir sizes
    # get the size of the output_dir in bytes
    # make it the human readable format
    # and also the number of files in the output_dir
    try:
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        )
        stats['output_dir_sizes'] = {
            'total': total_size,
            'totalh': humanize.naturalsize(total_size),
            'files': len(os.listdir(output_dir))
        }
    except Exception:
        stats['output_dir_sizes'] = None

    # Save to output_dir as .arg file
    try:
        output_dir = locals().get('output_dir', 'statistics')
        stats_file = os.path.join(output_dir, 'statistics/run_statistics.arg')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"System statistics saved to {stats_file}")
    except Exception as e:
        print(f"Failed to save system statistics: {e}")
    
    return tau, gamma, eigvecs, theta_vals

if __name__ == '__main__':
    def dataset(val):
        if val == 1:
            return {
                'd': 0.07,#0.06123724356957945,
                'aVx': 1.0,
                'aVa': 3.0,
                'c_const': 0.1,
                'x_shift': 0.1,
                'theta_min': 0,
                'theta_max': 2 * np.pi,
                'omega': 0.1,
                'num_points': 5000,
                'R_0': (0, 0, 0)
            }
    
        elif val == 2:
            #let a be an aVx and an aVa parameter
            return {
                'd': 0.06120,
                'aVx': 1.0,
                'aVa': 3.0,
                'c_const': 0.01,
                'x_shift': 0.01,
                'theta_min': 0,
                'theta_max': 2 * np.pi,
                'omega': 0.1,
                'num_points': 5000,
                'R_0': (0, 0, 0)
            }    
    
    #run the main function
    main(dataset(1)['d'], dataset(1)['aVx'], dataset(1)['aVa'], dataset(1)['c_const'], dataset(1)['x_shift'], dataset(1)['theta_min'], dataset(1)['theta_max'], dataset(1)['omega'], dataset(1)['num_points'], dataset(1)['R_0'], extended=False)
    
    #main(dataset(2)['d'], dataset(2)['aVx'], dataset(2)['aVa'], dataset(2)['c_const'], dataset(2)['x_shift'], dataset(2)['theta_min'], dataset(2)['theta_max'], dataset(2)['omega'], dataset(2)['num_points'], dataset(2)['R_0'], extended=False)

    