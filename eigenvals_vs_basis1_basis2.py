from new_bph import Hamiltonian

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cupy as cp

def getting_eigenvalues_gpu_drange(drange, H_thetas_drange):
    # Convert drange to cupy array for GPU processing
    d_values = cp.array(drange)
    
    # Initialize an array to store all eigenvalues
    num_d = len(drange)
    num_points = H_thetas_drange.shape[1]
    eigvals_all = cp.zeros((num_d, num_points, 4), dtype=cp.float64)  # 4 eigenvalues per Hamiltonian
    
    # Convert all Hamiltonians to cupy arrays once
    H_thetas_gpu = cp.array(H_thetas_drange)
    
    # Calculate eigenvalues for each d value
    for i in range(num_d):
        # Get the Hamiltonians for this d value
        H_thetas_for_d = H_thetas_gpu[i]
        
        # Calculate eigenvalues using cupy
        eigvals_gpu = cp.array([cp.linalg.eigh(H)[0] for H in H_thetas_for_d])
        
        # Store the eigenvalues directly in the CuPy array
        eigvals_all[i] = eigvals_gpu
    
    return cp.asnumpy(eigvals_all)

if __name__ == "__main__":
    #let a be an aVx and an aVa parameter
    aVx = 1.0
    aVa = 5.0
    c_const = 0.1  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 0.1  # Shift in x direction
    d = 0.2  # Radius of the circle, use unit circle for bigger radius
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 5000
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    #create a directory for the output
    output_dir = 'eigenvals_vs_basis1_basis2'
    os.makedirs(output_dir, exist_ok=True)
    
    #create a directory for the npy files
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    #create a directory for the plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    #get the eigenvalues
    # Example usage
    drange = np.linspace(0.1, 1.0, 100)  # 100 different d values from 0.1 to 1.0
    H_thetas_drange = []
    R_thetas_drange = []
    for d in drange:
        hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
        H_thetas_drange.append(hamiltonian.H_thetas())
        R_thetas_drange.append(hamiltonian.R_thetas())
    H_thetas_drange = np.array(H_thetas_drange)
    R_thetas_drange = np.array(R_thetas_drange)
    print(H_thetas_drange.shape)
    print(R_thetas_drange.shape)
    
    eigvals_all = getting_eigenvalues_gpu_drange(
        drange, H_thetas_drange
    )
    print(eigvals_all.shape) # (100, 5000, 4)

    # Define the 3D basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1, 0])
    basis2 = np.array([1, 1, -2])

    # Normalize the basis vectors
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)

    # Create plots for each d value
    for i, d in enumerate(drange):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        eigvals_d = eigvals_all[i]       # (5000, 4)
        R_d = R_thetas_drange[i]         # (5000, 3)

        # Project 3D R vectors into 2D plane orthogonal to (1,1,1)
        proj1 = R_d @ basis1             # (5000,)
        proj2 = R_d @ basis2             # (5000,)

        for j in range(4):
            ax.scatter(
                proj1,
                proj2,
                eigvals_d[:, j],
                label=f'Eigenvalue {j+1}',
                alpha=0.7,
                s=10
            )

        ax.legend()
        ax.set_xlabel('Basis 1 projection')
        ax.set_ylabel('Basis 2 projection')
        ax.set_zlabel('Eigenvalues')
        ax.set_title(f'Eigenvalues vs Projected Coordinates (d={d:.2f})')

        plot_path = os.path.join(plot_dir, f'eigenvalues_projected_d{d:.2f}.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)

    print("All projected eigenvalue plots have been generated.")
