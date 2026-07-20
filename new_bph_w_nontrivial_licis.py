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
    create_and_visualize_scaled_circle(R_0, d, num_points, theta_min, theta_max, save_dir, scale_factor=100)

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
    """Format a matrix with ASCII characters"""
    n, m = matrix.shape
    max_len = max(len(f"{x:.4f}") for row in matrix for x in row)
    
    # Calculate width based on max number length and matrix dimensions
    width = (max_len + 3) * m + 1
    
    lines = []
    if title:
        lines.append(f"    |{title:^{width-2}}|")
    
    # Top border
    lines.append("    |" + "-" * (width-2) + "|")
    
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
    lines.append("    |" + "-" * (width-2) + "|")
    
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

def main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, R_0, OUT_DIR):
    
    theta_vals = theta_range = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range)
    H_thetas = hamiltonian.H_thetas()
    R_thetas = hamiltonian.R_thetas()
    
    
    #create a directory for the output
    if OUT_DIR is None:
        output_dir = os.path.join(os.path.dirname(__file__), f'd_{d}')
    else:
        output_dir = os.path.join(os.path.dirname(__file__), OUT_DIR, f'd_{d}')
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
    #save the eigvals, but calculate them as well
    np.save(os.path.join(npy_dir, 'eigvals.npy'), eigenvalues)

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

def find_nontrivial_licis(aVx, aVa, x_shift, r0=None):
    """
    Find the three non-trivial LICI points around a trivial LICI.
    
    Parameters:
    - aVx: Parameter for Vx potential
    - aVa: Parameter for Va potential  
    - x_shift: Shift parameter for Va potential
    - r0: Optional, the trivial LICI position. If None, calculated as x_prime
    
    Returns:
    - Dictionary containing:
        * 'trivial_lci': (r0, r0, r0) - the trivial LICI
        * 'nontrivial_licis': List of three non-trivial LICI points
        * 'r0', 'r1', 'r2': The calculated bond distances
        * 'x_prime': The position of the potential minimum
        * 'd_CI': Distance from trivial to non-trivial LICI
    """
    # Calculate x_prime from the potential parameters
    x_prime = (aVa / aVx) / ((aVa / aVx) - 1) * x_shift
    
    # If r0 is not provided, use x_prime (trivial LICI at potential minimum)
    if r0 is None:
        r0 = x_prime
    
    # Calculate r1 and r2 using the derived formulas
    r1 = 3 * r0 - 2 * x_prime  # = r0 - 2*(x_prime - r0)
    r2 = 4 * x_prime - 3 * r0  # = r0 + 4*(x_prime - r0)
    
    # The three non-trivial LICI configurations
    nontrivial_licis = [
        (r1, r1, r2),  # LICI_1: molecules 1&2 equal, molecule 3 different
        (r1, r2, r1),  # LICI_2: molecules 1&3 equal, molecule 2 different
        (r2, r1, r1)   # LICI_3: molecules 2&3 equal, molecule 1 different
    ]
    
    # Calculate distance from trivial to non-trivial LICI
    d_CI = 2 * np.sqrt(6) * abs(x_prime - r0)
    
    return {
        'trivial_lci': (x_prime, x_prime, x_prime),
        'nontrivial_licis': nontrivial_licis,
        'r0': r0,
        'r1': r1,
        'r2': r2,
        'x_prime': x_prime,
        'd_CI': d_CI
    }

def print_lici_info(lici_data):
    """Print formatted information about the LICI points."""
    print("\n" + "="*60)
    print("LICI Point Information")
    print("="*60)
    print(f"x_prime: {lici_data['x_prime']:.6f}")
    print(f"r0 (trivial LICI): {lici_data['r0']:.6f}")
    print(f"r1: {lici_data['r1']:.6f}")
    print(f"r2: {lici_data['r2']:.6f}")
    print(f"Distance d_CI: {lici_data['d_CI']:.6f}")
    print("\nTrivial LICI:")
    print(f"  (r0, r0, r0) = {lici_data['trivial_lci']}")
    print("\nNon-trivial LICIs:")
    for i, lici in enumerate(lici_data['nontrivial_licis'], 1):
        print(f"  LICI_{i}: {lici}")
    print("="*60 + "\n")

def verify_lici_properties(lici_data, aVx, aVa, x_shift, c_const):
    """
    Verify that the found LICI points satisfy the CI condition.
    
    Parameters:
    - lici_data: Dictionary from find_nontrivial_licis
    - aVx, aVa, x_shift, c_const: Potential parameters
    """
    def Vx(x):
        return aVx * x**2
    
    def Va(x):
        return aVa * (x - x_shift)**2 + c_const
    
    print("\nVerifying CI condition (Va - Vx ≈ 0 for all coordinates):")
    print("-" * 60)
    
    # Check trivial LICI
    trivial = lici_data['trivial_lci']
    diff_trivial = [Va(r) - Vx(r) for r in trivial]
    print(f"Trivial LICI {trivial}:")
    print(f"  Va-Vx values: {[f'{d:.6f}' for d in diff_trivial]}")
    print(f"  Max difference: {max(abs(d) for d in diff_trivial):.8f}")
    
    # Check non-trivial LICIs
    for i, lici in enumerate(lici_data['nontrivial_licis'], 1):
        diff = [Va(r) - Vx(r) for r in lici]
        print(f"\nLICI_{i} {lici}:")
        print(f"  Va-Vx values: {[f'{d:.6f}' for d in diff]}")
        print(f"  Max difference: {max(abs(d) for d in diff):.8f}")
        print(f"  All equal? {all(abs(d - diff[0]) < 1e-10 for d in diff)}")
    
    print("-" * 60 + "\n")

def plot_ci_points_on_orthogonal_plane(ci_points, R_0, R_thetas, save_dir):
    """
    Plot CI points on the orthogonal plane using basis1 and basis2 as axes.
    Basis vectors are constructed dynamically (like the 'r0' branch in
    plot_vectors_2d_projection) as two orthogonal vectors spanning the plane
    perpendicular to the (1,1,1) direction.
    
    Parameters:
    - ci_points: Dictionary with 'trivial_ci' and 'nontrivial_licis' keys
    - R_0: Center point of the circle
    - R_thetas: Array of 3D points (shape (N, 3)) from hamiltonian.R_thetas(),
                tracing the actual circle orthogonal to (1,1,1)
    - save_dir: Directory to save the plot
    """
    
    # Normal to the plane: the (1,1,1) direction
    normal = np.array([1, 1, 1])
    normal = normal / np.linalg.norm(normal)
    
    # Build two orthogonal in-plane basis vectors via cross products,
    # same approach as the 'r0' branch above
    if not np.allclose(normal, np.array([1, 0, 0])):
        basis1 = np.cross(normal, np.array([1, 0, 0]))
    else:
        basis1 = np.cross(normal, np.array([0, 1, 0]))
    basis1 = basis1 / np.linalg.norm(basis1)
    
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    def project_to_plane(point):
        rel_point = np.array(point) - np.array(R_0)
        x_coord = np.dot(rel_point, basis1)
        y_coord = np.dot(rel_point, basis2)
        return x_coord, y_coord
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all CI points
    all_ci_points = [
        ('Origin', (0, 0, 0)),
        ('CI_0', (0.433, 0.433, 0.433)),
    ] + [(f'CI_{i+1}', ci) for i, ci in enumerate(ci_points['nontrivial_licis'])]
    for label, ci in all_ci_points:
        ci_x, ci_y = project_to_plane(ci)
        color = 'blue' if label == 'Origin' else 'red'
        ax.scatter(ci_x, ci_y, c=color, s=200, label=label,
                   marker='o' if label == 'Origin' else 's', zorder=5)
    
    # Plot center point R_0
    r0_x, r0_y = project_to_plane(R_0)
    ax.scatter(r0_x, r0_y, c='black', s=100, label='R_0 (center)', marker='*', zorder=6)
    
    # Plot the ACTUAL circle from R_thetas, projected onto the same plane
    R_thetas = np.asarray(R_thetas)  # expected shape (N, 3)
    circle_x, circle_y = zip(*[project_to_plane(pt) for pt in R_thetas])
    ax.plot(circle_x, circle_y, 'k--', alpha=0.5, label='Circle (R_thetas)', linewidth=2)
    
    ax.set_xlabel('Basis1 direction (orthogonal to (1,1,1))')
    ax.set_ylabel('Basis2 direction (orthogonal to (1,1,1))')
    ax.set_title('CI Points on Orthogonal Plane (centered at R_0)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/ci_points_orthogonal_plane.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CI points visualization saved to {save_dir}/ci_points_orthogonal_plane.png")
def dataset(val):
    """Return parameters for different datasets"""
    params = {}
    
    if val == 1:
        # Original dataset
        params['aVx'] = 1.0
        params['aVa'] = 1.3
        params['c_const'] = 0.01
        params['x_shift'] = 0.1
        params['theta_min'] = 0
        params['theta_max'] = 2 * np.pi
        params['omega'] = 0.1
        params['num_points'] = 5001
        params['d'] = 0.001
        params['R_0'] = (0.433, 0.433, 0.433)  # Default R_0 at r0
        
    elif val == 2:
        # Dataset with R_0 at r0
        params['aVx'] = 1.0
        params['aVa'] = 1.3
        params['c_const'] = 0.01
        params['x_shift'] = 0.1
        params['theta_min'] = 0
        params['theta_max'] = 2 * np.pi
        params['omega'] = 0.1
        params['num_points'] = 5001
        params['d'] = 0.001
        
        # Calculate r0, r1, r2
        x_prime = (params['aVa']/params['aVx']) / (params['aVa']/params['aVx']-1) * params['x_shift']
        r0 = x_prime
        x = 0  # Since r0 = x_prime, x = 2*(x_prime-r0) = 0
        
        # For dataset 2, use R_0 = (r0, r0, r0)
        params['R_0'] = (r0, r0, r0)
        
    elif val == 3:
        # Dataset with R_0 at r1
        params['aVx'] = 1.0
        params['aVa'] = 1.3
        params['c_const'] = 0.01
        params['x_shift'] = 0.1
        params['theta_min'] = 0
        params['theta_max'] = 2 * np.pi
        params['omega'] = 0.1
        params['num_points'] = 5001
        params['d'] = 0.001
        
        # Calculate r0, r1, r2
        x_prime = (params['aVa']/params['aVx']) / (params['aVa']/params['aVx']-1) * params['x_shift']
        r0 = x_prime
        
        # Using the formula derived in the notes: r1 = 3*r0 - 2*x_prime
        r1 = 3*r0 - 2*x_prime
        
        # For dataset 3, use R_0 = (r1, r1, r1)
        params['R_0'] = (r1, r1, r1)
        
    elif val == 4:
        # Dataset with R_0 at r2
        params['aVx'] = 1.0
        params['aVa'] = 1.3
        params['c_const'] = 0.01
        params['x_shift'] = 0.1
        params['theta_min'] = 0
        params['theta_max'] = 2 * np.pi
        params['omega'] = 0.1
        params['num_points'] = 5001
        params['d'] = 0.001
        
        # Calculate r0, r1, r2
        x_prime = (params['aVa']/params['aVx']) / (params['aVa']/params['aVx']-1) * params['x_shift']
        r0 = x_prime
        
        # Using the formula derived in the notes: r2 = 4*x_prime - 3*r0
        r2 = 4*x_prime - 3*r0
        
        # For dataset 4, use R_0 = (r2, r2, r2)
        params['R_0'] = (r2, r2, r2)
    
    return params

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers 3d projection

def plot_ci_seam_3d(x_prime, r0_range=(-3.0, 3.0), r0_marker=0.0, save_dir=None):
    """
    Plot the trivial CI seam and the three nontrivial-CI branch lines in
    full 3D (r1, r2, r3) space, showing all three branches converge on
    the trivial CI.
    """

    plt.close('all')

    r0_vals = np.linspace(r0_range[0], r0_range[1], 200)
    r1_vals = 3*r0_vals - 2*x_prime
    r2_vals = 4*x_prime - 3*r0_vals

    branch1 = np.stack([r1_vals, r1_vals, r2_vals], axis=1)
    branch2 = np.stack([r1_vals, r2_vals, r1_vals], axis=1)
    branch3 = np.stack([r2_vals, r1_vals, r1_vals], axis=1)

    CI_0 = np.array([x_prime, x_prime, x_prime])
    seam_t = np.linspace(r0_range[0], r0_range[1], 50)
    seam = np.stack([seam_t, seam_t, seam_t], axis=1)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*seam.T, 'k--', alpha=0.4, label='Trivial CI seam (1,1,1)')
    for branch, color, name in [(branch1, 'tab:red', 'CI_1 branch'),
                                  (branch2, 'tab:blue', 'CI_2 branch'),
                                  (branch3, 'tab:green', 'CI_3 branch')]:
        ax.plot(*branch.T, color=color, linewidth=2, label=name)

    ax.scatter(*CI_0, c='black', s=50, marker='x', zorder=5, label='CI_T')
    
    r1_m, r2_m = 3*r0_marker - 2*x_prime, 4*x_prime - 3*r0_marker
    for pt in [(r1_m, r1_m, r2_m), (r1_m, r2_m, r1_m), (r2_m, r1_m, r1_m)]:
        ax.scatter(*pt, c='orange', s=75, marker='o', zorder=5)
    
    ax.set_xlabel('r1'); ax.set_ylabel('r2'); ax.set_zlabel('r3')
    ax.set_title('Nontrivial CI branches converging on the trivial CI')
    ax.legend(loc='best')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/ci_seam_3d.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    # Common parameters
    d = 0.001  # radius of the circle
    aVx = 1.0
    aVa = 1.3
    c_const = 0.01
    x_shift = 0.1
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 5001
    
    # Find the actual trivial CI (at x_prime)
    tci_data = find_nontrivial_licis(aVx, aVa, x_shift, r0=None)
    
    # Find non-trivial CIs with R_0 = (0, 0, 0) as center
    R_0_center = (0, 0, 0)
    ci_data = find_nontrivial_licis(aVx, aVa, x_shift, r0=0)
    
    print("\n" + "="*60)
    print("ACTUAL TRIVIAL CI (at x_prime):")
    print("="*60)
    print(f"Trivial CI: {tci_data['trivial_lci']}")
    print("="*60 + "\n")
    
    print_lici_info(ci_data)
    verify_lici_properties(ci_data, aVx, aVa, x_shift, c_const)
    
    # Collect all CI points to analyze
    # Include both the actual trivial CI at x_prime and the non-trivial CIs
    all_ci_points = [
        ('Origin', (0, 0, 0)),  # Origin point for comparison
        ('CI_0', (0.433, 0.433, 0.433)),  # Trivial CI at r0
        ('CI_1', ci_data['nontrivial_licis'][0]),
        ('CI_2', ci_data['nontrivial_licis'][1]),
        ('CI_3', ci_data['nontrivial_licis'][2])
    ]

    all_ci_points_no_origin = [
        ('CI_0', (0.433, 0.433, 0.433)),  # Trivial CI at r0
        ('CI_1', ci_data['nontrivial_licis'][0]),
        ('CI_2', ci_data['nontrivial_licis'][1]),
        ('CI_3', ci_data['nontrivial_licis'][2])
    ]
    
    #calculate R_thetas using the Hamiltonian for R_0 = (0, 0, 0)
    theta_vals = theta_range = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0_center, d, theta_range)
    R_thetas = hamiltonian.R_thetas()

    # Create visualization of all CI points on orthogonal plane
    viz_dir = os.path.join(os.path.dirname(__file__), 'ci_analysis')
    plot_ci_points_on_orthogonal_plane(ci_data, R_0_center, R_thetas, viz_dir)
    
    # Run calculations for each CI point with d=0.001
    LEFU_DIR = os.path.join(os.path.dirname(__file__), 'lefutasok')
    
    for ci_name, ci_point in all_ci_points_no_origin:
        # Create folder name: d_0.001_point1_point2_point3
        point_str = f"{ci_point[0]:.3f}_{ci_point[1]:.3f}_{ci_point[2]:.3f}"
        folder_name = f"d_{d}_{point_str}"
        
        print(f"\n{'='*60}")
        print(f"Running calculation for {ci_name}: {ci_point}")
        print(f"Folder: {folder_name}")
        print(f"{'='*60}\n")
        
        # Run main calculation with this CI point as R_0
        main(d, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, ci_point, folder_name)

    """
    not needed, origin is not a ci but on a ci seam ???
    # Run additional calculation for Origin with d=0.433 to enclose CIs
    d_large = 0.433
    origin_point = (0, 0, 0)
    point_str = f"{origin_point[0]:.3f}_{origin_point[1]:.3f}_{origin_point[2]:.3f}"
    folder_name_large = f"d_{d_large}_{point_str}"
    

    print(f"\n{'='*60}")
    print(f"Running calculation for Origin with large d={d_large}: {origin_point}")
    print(f"Folder: {folder_name_large}")
    print(f"{'='*60}\n")
    
    main(d_large, aVx, aVa, c_const, x_shift, theta_min, theta_max, omega, num_points, origin_point, folder_name_large)
    """

    x_prime = (aVa / aVx) / ((aVa / aVx) - 1) * x_shift

    plot_ci_seam_3d(x_prime, r0_range=(-3.0, 3.0), r0_marker=0.0, save_dir='ci_analysis')


    for i, ci in enumerate(ci_data['nontrivial_licis'], start=1):
        ci = np.array(ci)
        dist = np.linalg.norm(ci - np.array([0, 0, 0]))
        print(f"\n\nCI_{i}: {ci}, distance from R_0: {dist:.6f}")
    
    for i, ci in enumerate(ci_data['nontrivial_licis'], start=1):
        ci = np.array(ci)
        dist = np.linalg.norm(ci - np.array([0.433, 0.433, 0.433]))
        print(f"\n\njust for curiosity\n\nCI_{i}: {ci}, distance from TCI: {dist:.6f}")


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



# Gabor ezt erre javitotta ki:
"""
# 06: (Va - Vx)(r1) = (Va - Vx)(r2) ==>
# 07: r1 = x_prime - delta; r2 = x_prime + delta
# let r0+r1+r2 = X
# we can say that r0 = 2*r1 if r1 = r2, because of only two can be equal of r0, r1, r2 at the same time to get a CI --> three points for an R0=(r0,r1,r2)
# so we can actually prove why is our point 8 is as it is 
Itt valamit nagyon félreértettél. Legalábbis a megjegyzésedből erre következtetek.
Ez a része a levezetésemnek arról szól, hogy keresem az összes LICI geometriát 
(három molekulára is azonos (Va-Vx)(ri) értéket kapunk) abban a síkban, amelyik 
tartalmazza az (r0,r0,r0) pontot és merőleges az r_1=r_2=r_3 egyenesre. 
Először is a jelölések: 
-- r0: ez határozza meg, hogy melyik síkban keresem a LICI-ket
-- r_1, r_2, r_3: a három molekula kötési távolsága.
	ezekre a levezetésben ilyen direkt módon nem hivatkoztam,
	csak most a magyarázatban.
-- r1, r2: a vizsgált síkban elhelyezkedő (r_1, r_2, r_3) pontok r_1, r_2, r_3 
	távolságai ezekből az értékekből kerülnek ki.

Egy picit átdoldoztam a levezetést, hogy jobban érthető legyen: 
# 01: Vx = x**2
# 02: Va = a * (x-x_shift)**2 + c0

# Trivial LICIs: (r_1, r_2, r_3) with r_1=r_2=r_3(=r0).

# Searching for nontrivial LICIs:
# If we can find an r1!=r2 with (Va - Vx)(r1) = (Va - Vx)(r2),
# the following configurations will provide an nontrivial LICI:
# -- LICI_1: (r_1,r_2,r_3)=(r1,r1,r2)
# -- LICI_2: (r_1,r_2,r_3)=(r1,r2,r1)
# -- LICI_3: (r_1,r_2,r_3)=(r2,r1,r1)
# -- LICI_4: (r_1,r_2,r_3)=(r2,r2,r1)
# -- LICI_5: (r_1,r_2,r_3)=(r2,r1,r2)
# -- LICI_6: (r_1,r_2,r_3)=(r1,r2,r2)

# As the trivial LICIs form a seam with the property of r_1=r_2=r_3,
# we used to set up our closed paths in the plane perpendicular to this
# "trivial" seem of LICIs. All points of this plane have the property of
# r_1 + r_2 + r_3 = 3 * r0, where (r0,r0,r0) is the trivial LICI in the plane.
# It is easy to see that LICI_1, LICI_2, LICI_3 are in the same plane, and
# similarly, LICI_4, LICI_5, LICI_6 are also in the same plane.

# Next we will look for nontrivial LICIs of type LICI_1, LICI_2, LICI_3
# with an additional requirement: r1+r1+r2=3*r0 with a predefined r0.

# 03: Va - Vx = (a-1) * x**2 -2*a*x_shift*x +c1
# 04: Va - Vx = (a-1) * (x - x_prime)**2 +c1
# 05: x_prime = (a/(a-1)) * x_shift

# 06: (Va - Vx)(r1) = (Va - Vx)(r2) and r1 != r2 ==>
# 07: r1 = x_prime - delta; r2 = x_prime + delta
#	(with arbitrary delta)

# 08: r1 + r1 + r2 = 3 * r0
# 09: 3 * x_prime - delta = 3 * r0
# 10: delta = 3 * (x_prime - r0)

# 11: r1 = x_prime - delta = 3*r0 - 2*x_prime = r0 - 2 * (x_prime - r0)
# 12: r2 = x_prime + delta = 4*x_prime - 3*r0 = r0 + 4 * (x_prime - r0)

# We need to know how far are the nontrivial LICIs from the trivial one
# in order to know how many LICIs are enclosed by a given circle.

# 13: d_CI = |r2 - r0| * sqrt(6) / 2 = 4*|x_prime-r0| * sqrt(6) / 2
# 14: d_CI = 2 * sqrt(6) * |x_prime-r0|

# Similar expressions could be derived for LICI_4, LICI_5, LICI_6,
# but here we will got:
# delta_prime = - 3 * (x_prime - r0) = - delta
# Taking this value to (11) and (12), we will get:
# r1_prime=r2 and r2_prime=r1.
# Combining this with the "definitions" of nontrivial LICI_i-s
# we should realize, that these LICIs are the same as LICI_1, LICI_2, LICI_3.
# More specifically: LICI_4 = LICI_1; LICI_5 = LICI_2; LICI_6 = LICI_3; 
"""