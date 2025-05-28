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
        
def compute_berry_phase(eigvectors_all, theta_vals):
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
                tau_val = np.vdot(psi_curr, grad_psi)
                tau[n, m, i] = tau_val.real
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
    
    tau, gamma = compute_berry_phase(eigvecs, theta_vals)
    #print("Tau:", tau)
    print("Gamma[:,:,-1]:\n", gamma[:,:,-1]) #print the last gamma matrix
    #create a report on the gamma matrix
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