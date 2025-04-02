import numpy as np
import os
import sys
sys.path.append('/home/zoli/arrowhead_new/completely_new/arrowhead_new/generalized')
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle
from main import *
print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")
import datetime

class Hamiltonian:
    def __init__(self, omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range):
        self.omega = omega
        self.aVx = aVx
        self.aVa = aVa
        self.x_shift = x_shift
        self.c = c_const
        self.R_0 = R_0
        self.d = d
        self.theta_range = theta_range

    def R_theta(self, theta):
        """
        Create a vector that traces a perfect circle orthogonal to the x=y=z line using the
        create_perfect_orthogonal_vectors function from the Arrowhead/generalized package.
        
        Parameters:
        d (float): The radius of the circle
        theta (float): The angle parameter
        
        Returns:
        numpy.ndarray: A 3D vector orthogonal to the x=y=z line
        """
        # Generate the perfect orthogonal vector
        return create_perfect_orthogonal_vectors(self.R_0, self.d, theta)

    def V_x(self, R_theta_val):
        return self.aVx * (R_theta_val**2)

    def V_a(self, R_theta_val):
        return self.aVa * ((R_theta_val - self.x_shift)**2 + self.c)
    
    def position_matrix(self, R_theta_val):
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

    def transitional_dipole_moment(self, eigvec_i, eigvec_f, position_operator):
        return np.vdot(eigvec_f, np.dot(position_operator, eigvec_i))

    def construct_matrix(self, theta):
        R_theta_val = self.R_theta(theta)
        Vx = [self.aVx * (R_theta_val[i] ** 2) for i in range(3)]
        Va = [self.aVa * (R_theta_val[i] ** 2) for i in range(3)]
        
        H = np.zeros((4, 4), dtype=complex)
        sumVx = sum(Vx)
        H[0, 0] = self.omega + sumVx
        for i in range(1, len(H)):
            H[i, i] = H[0, 0] + Va[i-1] - Vx[i-1]
            
        eigvals, eigvecs = np.linalg.eigh(H)
        sorted_indices = np.argsort(eigvals)
        sorted_eigvals = eigvals[sorted_indices]
        sorted_eigvecs = eigvecs[:, sorted_indices]
        
        pos_mat = self.position_matrix(R_theta_val)
        c10 = self.transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 1], pos_mat)
        c20 = self.transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 2], pos_mat)
        c30 = self.transitional_dipole_moment(sorted_eigvecs[:, 0], sorted_eigvecs[:, 3], pos_mat)
        
        H[0, 1] = H[1, 0] = c10
        H[0, 2] = H[2, 0] = c20
        H[0, 3] = H[3, 0] = c30

        return H

    def H_thetas(self):
        return [self.construct_matrix(theta) for theta in self.theta_range]

    def R_thetas(self):
        return [self.R_theta(theta) for theta in self.theta_range]

class BerryPhaseCalculator:
    def __init__(self, hamiltonian, R_thetas, eigenstates):
        self.hamiltonian = hamiltonian
        self.R_thetas = R_thetas
        self.eigenstates = eigenstates

    def calculate_berry_connection(self):
        """
        Calculate Berry connection A_n(R_theta) = <n(R_theta)|i∂_R|n(R_theta)>
        """
        num_states = self.eigenstates.shape[1]
        A_R = np.zeros((len(self.R_thetas)-1, num_states), dtype=complex)

        for i in range(len(self.R_thetas)-1):
            dR = self.R_thetas[i+1] - self.R_thetas[i]
            for n in range(num_states):
                v = self.eigenstates[i][:, n]
                dv_dR = (self.eigenstates[i+1][:, n] - v) / np.linalg.norm(dR)
                A_R[i, n] = np.vdot(v, 1j * dv_dR)

        return A_R

    def calculate_berry_phase(self):
        """
        Calculate Berry phase γ_n = ∫ A_n(R_theta) dR_theta
        """
        A_R = self.calculate_berry_connection()
        # Calculate berry phase for each state by integrating the Berry connection
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        berry_phases = np.zeros(num_states)
        
        for n in range(num_states):
            berry_phases[n] = np.sum(np.real(A_R[:, n]))
            
        return berry_phases

if __name__ == "__main__":
    #let a be an aVx and an aVa parameter
    aVx = 1.0
    aVa = 5.0
    c_const = 1.0  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 1.0  # Shift in x direction
    d = 0.001  # Radius of the circle
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 50
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    # Calculate Hamiltonians and eigenvectors at each theta value, explicitly including endpoint
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    H_thetas = H_theta = hamiltonian.H_thetas()
    r_theta = hamiltonian.R_thetas()
    
    # Calculate eigenvectors
    eigenvectors = eigvecs_all = np.array([np.linalg.eigh(H)[1] for H in H_theta])
    eigvals_all = np.array([np.linalg.eigh(H)[0] for H in H_theta])
    # Calculate Berry phase
    berry_phase_calculator = BerryPhaseCalculator(hamiltonian, r_theta, eigenvectors)
    berry_phase = berry_phase_calculator.calculate_berry_phase()
    
    # Print results for all of the eigenstates
    for i in range(len(berry_phase)):
        print(f"Berry phase: {berry_phase[i]} for eigenstate {i}")
    
    # Save results
    output_dir = f'results_thetamin_{theta_min:.2f}_thetamax_{theta_max:.2f}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    plot_dir = f'{output_dir}/plots'
    abs_dir = f'{plot_dir}/abs'
    real_dir = f'{plot_dir}/real'
    imag_dir = f'{plot_dir}/imag'
    total_sum_dir = f'{plot_dir}/total_sum'
    npy_dir = f'{output_dir}/npy'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(abs_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(imag_dir, exist_ok=True)
    os.makedirs(total_sum_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    
    for state in range(eigvecs_all.shape[2]):
        # Calculate H*v for each theta value
        Hv_results = np.zeros((len(theta_vals), eigvecs_all.shape[1]), dtype=complex)
        #get eigenvaluesof each H_theta, it is not theta vals
        #calculate H_thetas array by calculating H_theta, it should be a (num_points, 4, 4) array, like (theta_value, 4, 4)
        #H_thetas = np.array([hamiltonian(theta, omega, aVx, aVa, c_const, x_shift, d)[0] for theta in theta_vals])
        #print(H_thetas.shape)
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

