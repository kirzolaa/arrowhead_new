#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import hbar
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle

# Import the Hamiltonian class from new_bph.py
from new_bph import Hamiltonian

def plot_va_vx_plus_omega(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_min, theta_max, num_points):
    """
    Plot Va and Vx+hbar*omega of the molecule for different theta values.
    
    Parameters:
    omega (float): Angular frequency of the oscillator
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    theta_min (float): Minimum theta value in radians
    theta_max (float): Maximum theta value in radians
    num_points (int): Number of points to generate
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'results_thetamin_{theta_min:.2f}_thetamax_{theta_max:.2f}_{timestamp}'
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Generate theta values
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
    
    # Create Hamiltonian instance
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    
    # Generate R vectors for each theta
    R_thetas = np.array(hamiltonian.R_thetas())
    
    # Calculate Va and Vx values for each component of R
    Va_values = []
    Vx_values = []
    Vx_plus_hbar_omega_values = []
    
    for i, R_theta in enumerate(R_thetas):
        # Calculate Va and Vx for each component of the R vector
        Va_components = hamiltonian.Va(R_theta)
        Vx_components = hamiltonian.Vx(R_theta)
        
        # Add hbar*omega to Vx
        Vx_plus_hbar_omega = [Vx + hbar * omega for Vx in Vx_components]
        
        Va_values.append(Va_components)
        Vx_values.append(Vx_components)
        Vx_plus_hbar_omega_values.append(Vx_plus_hbar_omega)
    
    # Convert to numpy arrays for easier manipulation
    Va_values = np.array(Va_values)
    Vx_values = np.array(Vx_values)
    Vx_plus_hbar_omega_values = np.array(Vx_plus_hbar_omega_values)
    
    # Plot Va and Vx+hbar*omega vs theta for each component
    plt.figure(figsize=(12, 8))
    
    # Plot Va components
    for i in range(Va_values.shape[1]):
        plt.plot(theta_vals, Va_values[:, i], label=f'Va component {i}')
    
    # Plot Vx+hbar*omega components
    for i in range(Vx_plus_hbar_omega_values.shape[1]):
        plt.plot(theta_vals, Vx_plus_hbar_omega_values[:, i], '--', label=f'Vx+hbar*omega component {i}')
    
    plt.xlabel('Theta (radians)')
    plt.ylabel('Potential Energy')
    plt.title('Va and Vx+hbar*omega vs Theta')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/plots/Va_Vx_plus_hbar_omega.png')
    
    # Plot Va and Vx+hbar*omega in 3D
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D parametric plot for Va and Vx+hbar*omega
    for i in range(Va_values.shape[1]):
        ax.plot(R_thetas[:, 0], R_thetas[:, 1], Va_values[:, i], label=f'Va component {i}')
        ax.plot(R_thetas[:, 0], R_thetas[:, 1], Vx_plus_hbar_omega_values[:, i], '--', label=f'Vx+hbar*omega component {i}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential Energy')
    ax.set_title('3D Plot of Va and Vx+hbar*omega')
    ax.legend()
    plt.savefig(f'{output_dir}/plots/Va_Vx_plus_hbar_omega_3D.png')
    
    # Save the data for further analysis
    np.savetxt(f'{output_dir}/Va_values.txt', Va_values)
    np.savetxt(f'{output_dir}/Vx_values.txt', Vx_values)
    np.savetxt(f'{output_dir}/Vx_plus_hbar_omega_values.txt', Vx_plus_hbar_omega_values)
    np.savetxt(f'{output_dir}/theta_vals.txt', theta_vals)
    np.savetxt(f'{output_dir}/R_thetas.txt', R_thetas.reshape(-1, R_thetas.shape[-1]))
    
    print(f"Va and Vx+hbar*omega plots saved to {output_dir}/plots directory.")
    return output_dir, R_thetas, Va_values, Vx_values, Vx_plus_hbar_omega_values, theta_vals

if __name__ == "__main__":
    # Set parameters
    omega = 1.0
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.0
    c_const = 0.1
    R_0 = np.array([0, 0, 0])
    d = 0.1
    theta_min = 0.0
    theta_max = 2 * np.pi
    num_points = 100
    
    # Plot Va and Vx+hbar*omega
    output_dir, R_thetas, Va_values, Vx_values, Vx_plus_hbar_omega_values, theta_vals = plot_va_vx_plus_omega(
        omega, aVx, aVa, x_shift, c_const, R_0, d, theta_min, theta_max, num_points
    )
    
    plt.show()
