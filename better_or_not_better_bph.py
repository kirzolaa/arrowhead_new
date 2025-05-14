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

    # Optional: normalize eigenvectors for safety
    eigvectors_all = eigvectors_all / np.linalg.norm(eigvectors_all, axis=1, keepdims=True)

    for i in range(N):
        i_prev = (i - 1)
        i_next = (i + 1) % N

        for n in range(M):
            for m in range(M):
                if i == 0:
                    i_prev = N - 1
                else:
                    i_prev = i - 1
                if i == N - 1:
                    i_next = 0
                else:
                    i_next = len(eigvectors_all) - 1
                
                psi_prev = eigvectors_all[i_prev, :, n] # (THETA, COMPONENT, STATE)
                psi_next = eigvectors_all[i_next, :, n]
                psi_curr = eigvectors_all[i, :, m]

                grad_psi = (psi_next - psi_prev) / (delta_theta)
                tau_val = 1j * np.vdot(psi_curr, grad_psi)
                """ BAD, VERY BAD
                if theta_vals[i] == theta_max:
                    tau_val *= 2
                elif theta_vals[i] == theta_min:
                    tau_val *= 2
                else:
                    tau_val *= 1
                """
                tau[n, m, i] = tau_val

                # Accumulate γ (imaginary part of τ)
                if i > 0:
                    # Use trapezoidal rule for more accurate integration
                    if i > 0:
                        gamma[n, m, i] = gamma[n, m, i - 1] + 0.5 * np.imag(tau_val + tau[n, m, i - 1]) * delta_theta

    # Remove padded τ/γ at final point to keep shape = original
    tau = tau[:, :, :N_orig]
    gamma = gamma[:, :, :N_orig]
    """
    # Zero out diagonals of τ and γ (optional but helps stability)
    for n in range(M):
        tau[n, n, :] = 0.0
        gamma[n, n, :] = 0.0
    """
    if continuity_check:
        print("Eigenvector continuity (should be ~1):")
        for i in range(1, N_orig):
            for n in range(M):
                overlap = np.abs(np.vdot(eigvectors_all[i - 1, :, n], eigvectors_all[i, :, n]))
                if overlap < 0.99:
                    print(f"Theta index {i}, state {n+1}: overlap = {overlap:.6f}")

    # Report total Berry phase per eigenstate
    gamma_closed_loop = gamma[:, :, -1]
    berry_phase_per_state = np.angle(np.exp(1j * gamma_closed_loop.diagonal()))
    print("Berry phase per state:", berry_phase_per_state)

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

    #plot the trace of gamma
    plt.figure(figsize=(12, 6))
    trace_gamma = np.trace(gamma[:, :, :], axis1=0, axis2=1)
    # Use the same extended_theta_vals to match dimensions
    plt.plot(extended_theta_vals, trace_gamma)
    plt.title("Tr(γ) vs θ")
    plt.xlabel('Theta (θ)')
    plt.ylabel('Tr(γ)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Tr_gamma_vs_theta.png')
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
    