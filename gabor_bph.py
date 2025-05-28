import numpy as np
import matplotlib.pyplot as plt
import os
from new_bph import Hamiltonian
#import an rk4 integrator
from scipy.integrate import odeint
from os.path import join
from generalized.vector_utils import multiprocessing_create_perfect_orthogonal_circle
from perfect_orthogonal_circle import verify_circle_properties, visualize_perfect_orthogonal_circle
from scaled_circle_visualizer import create_and_visualize_scaled_circle
from scipy.constants import hbar
import copy
    
def visualize_vectorz(R_0, d, num_points, theta_min, theta_max, save_dir):
    #use the perfect_orthogonal_circle.py script to visualize the R_theta vectors
    
    #visualize the R_theta vectors
    points = multiprocessing_create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max) #we already have a method for this
    #points = create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max)
    print(points.shape)
    
    # Pass R_0 to both functions so they know where the center is
    visualize_perfect_orthogonal_circle(points, save_dir, R_0=R_0)
    verify_circle_properties(d, num_points, points, save_dir, R_0=R_0)
    
    # Also create a scaled visualization for better visibility
    # Use a scale factor of 60 to make the small circle (d=0.001) visible
    create_and_visualize_scaled_circle(R_0, d, num_points, theta_min, theta_max, save_dir, scale_factor=60)

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
    elements = [(0, 1), (1, 2), (2, 3)]
    
    # Plot real and imaginary parts of tau
    plt.subplot(2, 1, 1)
    for i, j in elements:
        plt.plot(theta_vals/np.pi, np.real(tau[i, j, :]),
                label=f'Re(τ_{i+1}{j+1})', linestyle='-')
        plt.plot(theta_vals/np.pi, np.imag(tau[i, j, :]),
                label=f'Im(τ_{i+1}{j+1})', linestyle='--')
    plt.xlabel('θ')
    plt.ylabel('τ')
    plt.title('Evolution of τ matrix elements')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/tau_matrix_elements.png')
    
    
    plt.subplot(2, 1, 2)
    for i, j in elements:
        plt.plot(theta_vals/np.pi, np.real(gamma[i, j, :]/np.pi),
                label=f'Re(γ_{i+1}{j+1})', linestyle='-')
        plt.plot(theta_vals/np.pi, np.imag(gamma[i, j, :]/np.pi),
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
        
def compute_berry_phase(eigvectors_all, theta_vals, output_dir=None):
    """
    Note: the first and the last tau, gamma matrices are seemingly wrong!
    
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.
    Enhanced with additional diagnostics and visualizations.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - theta_vals: ndarray of shape (N,), parameter values along the path
    - output_dir: Directory to save diagnostic plots and data (optional)

    Returns:
    - tau: ndarray of shape (M, M, N), Berry connection for each eigenstate in radians
    - gamma: ndarray of shape (M, M, N), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    
    tau = np.zeros((M, M, N), dtype=np.complex128)  # Changed to complex to store both real and imaginary parts
    gamma = np.zeros((M, M, N), dtype=np.float64)
    
    # Arrays for diagnostics
    tau_imag_magnitudes = np.zeros((M, M, N), dtype=np.float64)
    tau_real_magnitudes = np.zeros((M, M, N), dtype=np.float64)
    gamma_increments = np.zeros((M, M, N-1), dtype=np.float64)
    
    # Tracking largest values for diagnostics
    largest_tau_imag = 0.0
    largest_tau_imag_loc = (0, 0, 0)
    largest_gamma_increment = 0.0
    largest_gamma_increment_loc = (0, 0, 0)

    for n in range(M):
        for m in range(M):
            for i in range(N): 
                # Handle boundary conditions for the finite difference
                # Inside compute_berry_phase

                im1 = i - 1
                ip1 = i + 1
                if i == 0:     im1 = -2
                if i == N - 1: ip1 =  1
                psi_prev = eigvectors_all[im1, :, n]
                psi_next = eigvectors_all[ip1, :, n]
                delta_theta_for_grad = theta_vals[2] - theta_vals[0]
                if ip1 - im1 != 2:
                    if np.vdot(psi_prev,psi_next).real < 0:
                        if i == 0:
                            psi_prev = -1 * copy.deepcopy(psi_prev)
                        else:
                            psi_next = -1 * copy.deepcopy(psi_next)

                '''
                if i == 0:
                    psi_prev = eigvectors_all[N - 2, :, n]  # Vector at theta_max (which is theta_0 if path is closed)
                                                            # OR eigvectors_all[N-2,:,n] if using N-1 points to define the distinct loop points (0 to N-2) and N-1 is same as 0
                                                            # Let's assume N points, theta_vals[N-1] is distinct from theta_vals[0] but psi(theta_vals[N-1]) is "before" psi(theta_vals[0])
                    psi_next = eigvectors_all[1, :, n]
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                elif i == N - 1:
                    psi_prev = eigvectors_all[N - 2, :, n]
                    psi_next = eigvectors_all[2, :, n] # Vector at theta_0 (which is theta_N-1 + step if path is closed)
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                else:
                    psi_prev = eigvectors_all[i - 1, :, n]
                    psi_next = eigvectors_all[i + 1, :, n]
                    delta_theta_for_grad = theta_vals[i + 1] - theta_vals[i - 1]
                #'''
                psi_curr = eigvectors_all[i, :, m]
                # Normalize for safety (elvileg 1-gyel osztunk itt, mivel a vektorok eigh-val számolva)
                psi_prev = psi_prev / np.linalg.norm(psi_prev)
                psi_next = psi_next / np.linalg.norm(psi_next)
                psi_curr = psi_curr / np.linalg.norm(psi_curr)

                # Finite difference approximation of ∇_theta |ψ>
                delta_psi = psi_next - psi_prev
                grad_psi = delta_psi / (delta_theta_for_grad) # Corrected delta_theta

                # τ = ⟨ψ_i | ∇_theta | ψ_{i-1}⟩  (Corrected index for tau)
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                tau[n, m, i] = tau_val
                
                # Store magnitudes for diagnostics
                tau_imag_magnitudes[n, m, i] = np.abs(np.imag(tau_val))
                tau_real_magnitudes[n, m, i] = np.abs(np.real(tau_val))
                
                # Track largest values
                if np.abs(np.imag(tau_val)) > largest_tau_imag:
                    largest_tau_imag = np.abs(np.imag(tau_val))
                    largest_tau_imag_loc = (n, m, i)
                
                # · d_theta to integrate. 
                if i == 0:
                   gamma[n, m, i] = 0.0
                else:
                    delta_theta_integrate = theta_vals[i] - theta_vals[i-1]
                    
                    # Use trapezoidal rule for more accurate integration
                    gamma_increment = np.imag(tau_val + tau[n, m, i - 1]) * delta_theta_integrate / 2.0
                    gamma[n, m, i] = gamma[n, m, i - 1] + gamma_increment
                    
                    # Store increment for diagnostics
                    gamma_increments[n, m, i-1] = gamma_increment
                    
                    # Track largest increment
                    if np.abs(gamma_increment) > largest_gamma_increment:
                        largest_gamma_increment = np.abs(gamma_increment)
                        largest_gamma_increment_loc = (n, m, i-1)
    
    # Print diagnostic information about large tau and gamma values
    print(f"\nDiagnostic Information:")
    print(f"Largest imaginary tau value: {largest_tau_imag:.8f} at (n={largest_tau_imag_loc[0]+1}, m={largest_tau_imag_loc[1]+1}, theta_idx={largest_tau_imag_loc[2]})")
    print(f"Largest imaginary tau value / π: {largest_tau_imag/np.pi:.8f}π at (n={largest_tau_imag_loc[0]+1}, m={largest_tau_imag_loc[1]+1}, theta_idx={largest_tau_imag_loc[2]})")
    print(f"Largest gamma increment: {largest_gamma_increment:.8f} at (n={largest_gamma_increment_loc[0]+1}, m={largest_gamma_increment_loc[1]+1}, theta_idx={largest_gamma_increment_loc[2]})")
    print(f"Largest gamma increment / π: {largest_gamma_increment/np.pi:.8f}π at (n={largest_gamma_increment_loc[0]+1}, m={largest_gamma_increment_loc[1]+1}, theta_idx={largest_gamma_increment_loc[2]})")
    
    # Find the matrix elements with the largest contribution to the trace
    tau_imag_sum = np.sum(tau_imag_magnitudes, axis=2)
    gamma_final_abs = np.abs(gamma[:,:,-1])
    
    # Get the indices of the top 5 contributors
    tau_flat_indices = np.argsort(tau_imag_sum.flatten())[-5:]
    gamma_flat_indices = np.argsort(gamma_final_abs.flatten())[-5:]
    
    print("\nTop 5 contributors to tau (imaginary part sum):")
    for idx in tau_flat_indices[::-1]:
        n, m = np.unravel_index(idx, tau_imag_sum.shape)
        print(f"  Tau[{n+1},{m+1}]: Sum of abs(imag) = {tau_imag_sum[n,m]:.8f}, Final value = {np.imag(tau[n,m,-1]):.8f}")
        print(f"                   Sum of abs(imag) / π = {tau_imag_sum[n,m]/np.pi:.8f}π, Final value / π = {np.imag(tau[n,m,-1])/np.pi:.8f}π")
    
    print("\nTop 5 contributors to gamma (final values):")
    for idx in gamma_flat_indices[::-1]:
        n, m = np.unravel_index(idx, gamma_final_abs.shape)
        print(f"  Gamma[{n+1},{m+1}]: Final value = {gamma[n,m,-1]:.8f}, Final value / π = {gamma[n,m,-1]/np.pi:.8f}π")
    
    # If output directory is provided, create diagnostic plots
    if output_dir is not None:
        # Create diagnostic plots directory
        diag_dir = os.path.join(output_dir, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        
        # Plot the evolution of the largest contributors
        plt.figure(figsize=(12, 8))
        for idx in gamma_flat_indices[-3:]:
            n, m = np.unravel_index(idx, gamma_final_abs.shape)
            plt.plot(theta_vals[1:]/np.pi, gamma[n,m,1:]/np.pi, label=f'Gamma[{n+1},{m+1}]/π')
        plt.title('Evolution of Largest Gamma Contributors (in units of π)')
        plt.xlabel('θ/π')
        plt.ylabel('Gamma Value/π')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{diag_dir}/largest_gamma_evolution.png', dpi=300)
        plt.close()
        
        # Plot the evolution of tau imaginary part for largest contributors
        plt.figure(figsize=(12, 8))
        for idx in tau_flat_indices[-3:]:
            n, m = np.unravel_index(idx, tau_imag_sum.shape)
            plt.plot(theta_vals[1:]/np.pi, np.imag(tau[n,m,1:]), label=f'Imag(Tau[{n+1},{m+1}])')
        plt.title('Evolution of Largest Tau Imaginary Parts')
        plt.xlabel('θ/π')
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
            plt.plot(theta_vals[1:-1]/np.pi, increments/np.pi, label=f'Gamma[{n+1},{m+1}] increments/π')
        plt.title('Gamma Increments for Largest Contributors (in units of π)')
        plt.xlabel('θ/π')
        plt.ylabel('Increment Value/π')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{diag_dir}/gamma_increments.png', dpi=300)
        plt.close()
        
        # Plot a heatmap of the final gamma matrix
        plt.figure(figsize=(10, 8))
        # Use RdBu_r colormap (red-white-blue reversed) for better visualization
        # Normalize the colormap to make zero values white and set symmetric limits
        gamma_max = np.max(np.abs(gamma[:,:,-1]))
        im = plt.imshow(gamma[:,:,-1]/np.pi, cmap='RdBu_r', vmin=-gamma_max/np.pi, vmax=gamma_max/np.pi)
        plt.colorbar(im, label='Gamma Value / π')
        plt.title('Final Gamma Matrix Heatmap (in units of π)')
        plt.xlabel('m')
        plt.ylabel('n')
        # Ensure that the x and y labels are integers
        plt.xticks(np.arange(0, M, 1))
        plt.yticks(np.arange(0, M, 1))
        # Add text annotations with more digits
        for i in range(M):
            for j in range(M):
                # Format as π units with more precision
                value_text = f'{gamma[i,j,-1]/np.pi:.4f}π'
                # Choose text color based on background intensity
                text_color = 'w' if abs(gamma[i,j,-1]/np.pi) > 1.5 else 'k'
                text = plt.text(j, i, value_text,
                              ha="center", va="center", color=text_color, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{diag_dir}/gamma_final_heatmap.png', dpi=300)
        plt.close()
        
        # Also create a heatmap of the tau imaginary part
        plt.figure(figsize=(10, 8))
        tau_imag_final = np.imag(tau[:,:,-1])
        tau_max = np.max(np.abs(tau_imag_final))
        im = plt.imshow(tau_imag_final, cmap='RdBu_r', vmin=-tau_max, vmax=tau_max)
        plt.colorbar(im, label='Imag(Tau)')
        plt.title('Final Tau Imaginary Part Heatmap')
        plt.xlabel('m')
        plt.ylabel('n')
        #ensure that the x and y labels are even numbers
        plt.xticks(np.arange(0, M, 1))
        plt.yticks(np.arange(0, M, 1))
        # Add text annotations
        for i in range(M):
            for j in range(M):
                value_text = f'{tau_imag_final[i,j]:.4f}'
                text_color = 'w' if abs(tau_imag_final[i,j]) > 0.3 else 'k'
                text = plt.text(j, i, value_text,
                              ha="center", va="center", color=text_color, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{diag_dir}/tau_imag_final_heatmap.png', dpi=300)
        plt.close()
        
        # Check eigenvector continuity
        continuity_data = []
        print("\nEigenvector continuity (should be ~1):")
        for i in range(1, N):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all[i - 1, :, n], eigvectors_all[i, :, n]))
                continuity_data.append((i, n, overlap))
                if overlap < 0.9:  # Only print problematic overlaps
                    print(f"  Overlap between θ_{i-1} and θ_{i} for state {n+1}: {overlap:.6f}")
        
        # Plot continuity data
        plt.figure(figsize=(12, 6))
        for n in range(M):
            overlaps = [data[2] for data in continuity_data if data[1] == n]
            theta_indices = [data[0] for data in continuity_data if data[1] == n]
            plt.plot(theta_vals[theta_indices], overlaps, label=f'State {n+1}')
        plt.title('Eigenvector Continuity Between Adjacent Points')
        plt.xlabel('θ')
        plt.ylabel('Overlap Magnitude')
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{diag_dir}/eigenvector_continuity.png')
        plt.close()
        
        # Save the gamma matrix to a text file
        with open(f'{diag_dir}/gamma_matrix.txt', 'w') as f:
            f.write("Final Gamma Matrix:\n")
            for i in range(M):
                for j in range(M):
                    f.write(f"Gamma[{i+1},{j+1}] = {gamma[i,j,-1]}\n")
    
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

def main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0):
    
    theta_vals = theta_range = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range)
    H_thetas = hamiltonian.H_thetas()
    R_thetas = hamiltonian.R_thetas()
    
    
    #create a directory for the output
    output_dir = os.path.join(os.path.dirname(__file__), 'berry_phase_corrected_run_n_minus_1')
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
    
    # Pass the output directory to compute_berry_phase for enhanced diagnostics
    tau, gamma = compute_berry_phase(eigvecs, theta_vals, output_dir)
    
    # Print the last gamma matrix
    print("Gamma[:,:,-1]:\n", gamma[:,:,-1])
    
    # Create a report on the gamma matrix
    with open(f'{output_dir}/gamma_report.txt', "w") as f:
        f.write("Gamma matrix report:\n===========================================\n")
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                f.write(f"Gamma[{i+1},{j+1}]: {gamma[i,j,-2]}\n")
                f.write(f"Tau[{i+1},{j+1}]: {tau[i,j,-2]}\n")
            f.write("\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(gamma[:,:,-2]/np.pi, "Berry Phase Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(tau[:,:,-2], "Tau Matrix", output_dir))
        f.write("\n\n")
        
        # Add additional diagnostic information to the report
        f.write("===========================================\n")
        f.write("Additional Diagnostics:\n")
        f.write("===========================================\n")
        
        # Add information about the largest contributors to gamma
        gamma_final_abs = np.abs(gamma[:,:,-1])
        gamma_flat_indices = np.argsort(gamma_final_abs.flatten())[-5:]
        
        f.write("\nTop 5 contributors to gamma (final values):\n")
        for idx in gamma_flat_indices[::-1]:
            n, m = np.unravel_index(idx, gamma_final_abs.shape)
            f.write(f"  Gamma[{n+1},{m+1}]: Final value = {gamma[n,m,-1]:.6f}\n")
        
        # Add information about the eigenvector continuity
        f.write("\nEigenvector continuity summary:\n")
        continuity_issues = 0
        for i in range(1, len(theta_vals)):
            for n in range(gamma.shape[0]):
                overlap = np.abs(np.vdot(eigvecs[i-1,:,n], eigvecs[i,:,n]))
                if overlap < 0.9:
                    f.write(f"  Low overlap between θ_{i-1} and θ_{i} for state {n+1}: {overlap:.6f}\n")
                    continuity_issues += 1
        
        if continuity_issues == 0:
            f.write("  No significant continuity issues detected.\n")
        f.write("===========================================\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(gamma[:,:,-1]/np.pi, "The last Berry Phase Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(tau[:,:,-1], "The last Tau Matrix", output_dir))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write(f"(i,Theta): ({0}, {theta_vals[0]/np.pi:.10}) ({1}, {theta_vals[1]/np.pi:.10}) ({2}, {theta_vals[2]/np.pi:.10})\n")
        f.write(f"(i,Theta): ({-3}, {theta_vals[-3]/np.pi:.10}) ({-2}, {theta_vals[-2]/np.pi:.10}) ({-1}, {theta_vals[-1]/np.pi:.10}) ")
        f.write("\n")
        Va_values = np.array(hamiltonian.Va_theta_vals(R_thetas))
        Vx_values = np.array(hamiltonian.Vx_theta_vals(R_thetas))
        for i in range(3):
            f.write(f"Va[theta=0]-Vx[theta=0], i={i}:\n{(Vx_values[0, i] + omega * hbar) - Va_values[0, i]}")
            f.write(f"  {(Vx_values[0, i] + omega * hbar) - Va_values[0, i] - ((Vx_values[0, 0] + omega * hbar) - Va_values[0, 0])}\n")
            f.write(f"{R_thetas[0]}\n")
        f.write("===========================================\n")

    #print the gamma matrix
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            print(f"Gamma[{i+1},{j+1}]: {gamma[i,j,-2]}")
            print(f"Tau[{i+1},{j+1}]: {tau[i,j,-2]}")
    

    #save the tau and gamma matrices
    np.save(os.path.join(npy_dir, 'tau.npy'), tau)
    np.save(os.path.join(npy_dir, 'gamma.npy'), gamma)

    #save the eigvecs
    np.save(os.path.join(npy_dir, 'eigvecs.npy'), eigvecs)

    #save the theta_vals
    np.save(os.path.join(npy_dir, 'theta_vals.npy'), theta_vals)


    #plot the gamma and tau matrices
    plot_matrix_elements(tau, gamma, theta_vals, plot_dir)
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
    
    return tau, gamma, eigvecs, theta_vals

if __name__ == '__main__':
    dataset = 2

    if dataset == 1:
        d = 0.001 #radius of the circle
        aVx = 1.0
        aVa = 1.3
        c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
        x_shift = 0.1  # Shift in x direction
        theta_min = 0
        theta_max = 2 * np.pi
        omega = 0.1
        num_points = 50001
        R_0 = (0, 0, 0)
    
    elif dataset == 2:
        #let a be an aVx and an aVa parameter
        aVx = 1.0
        aVa = 1.3
        c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
        x_shift = 0.1  # Shift in x direction
        theta_min = 0
        theta_max = 2 * np.pi
        omega = 0.1
        num_points = 5001
        x_prime = (aVa/aVx) / (aVa/aVx-1) * x_shift
        r0 = x_prime * 1
        x = (2 * (x_prime - r0)) * 1
        d = 0.001 # 1e-10 # 0.06123724356957945 ...  0.06123724356957950
        n_CI = 0
        if n_CI<3:
            R_0 = [r0+x+x if i == n_CI else r0-x for i in range(3)]
        else:
            R_0 = (r0, r0, r0)
        print("R_0:",R_0, "\tr0:", r0, "\tsum(R_0)/3:", sum(R_0)/3)

    
    main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0)


# 01: Vx = x**2
# 02: Va = a * (x-x_shift)**2 + c0

# 03: Va - Vx = (a-1) * x**2 -2*a*x_shift*x +c1
# 04: Va - Vx = (a-1) * (x - x_prime)**2 +c1
# 05: x_prime = (a/(a-1)) * x_shift

# 06: (Va - Vx)(r1) = (Va - Vx)(r2) ==>
# 07: r1 = x_prime - delta; r2 = x_prime + delta
# let r0+r1+r2 = X
# we can say that r0 = 2*r1 if r1 = r2, because of only two can be equal of r0, r1, r2 at the same time to get a CI --> three points for an R0=(r0,r1,r2)
# so we can actually prove why is our point 8 is as it is
# 08: r2 - r0 = -2 * (r1 - r0) ==>
# 09: 2*(r0-r1) = r2-r0
# 10: 2*r0 -2*x_prime + 2*delta = x_prime + delta - r0
# 11: delta = 3 * (x_prime - r0)

# 12: r1 = 3*r0 - 2*x_prime = r0 - 2 * (x_prime - r0)
# 13: r2 = 4*x_prime - 3*r0 = r0 + 4 * (x_prime - r0)

# 14: d_CI = (r2 - r0) * sqrt(6) / 2 = 4*(x_prime-r0) * sqrt(6) / 2
# 15: d_CI = 2 * sqrt(6) * (x_prime-r0)