from new_bph import Hamiltonian
from threeD_fig_eigenvec_comp_comps import fix_sign
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cupy as cp

def getting_eigenvalues_gpu_drange(drange, H_thetas_drange):
    num_d = len(drange)
    num_points = H_thetas_drange.shape[1]
    eigvals_all = cp.zeros((num_d, num_points, 4), dtype=cp.float64)  # 4 eigenvalues per Hamiltonian
    eigvecs_all = cp.zeros((num_d, num_points, 4, 4), dtype=cp.float64)  # 4 eigenvectors per Hamiltonian
    # Convert all Hamiltonians to cupy arrays once
    H_thetas_gpu = cp.array(H_thetas_drange)
    
    for i, d in enumerate(drange):
        print(f"Processing d = {d:.2f}")
        H_thetas_for_d = H_thetas_gpu[i]
        
        # Calculate eigenvalues using cupy
        eigvals_gpu = cp.array([cp.linalg.eigh(H)[0] for H in H_thetas_for_d])
        # Calculate eigenvectors using cupy
        eigvecs_gpu = cp.array([cp.linalg.eigh(H)[1] for H in H_thetas_for_d])
        
        # Fix eigenvector signs on GPU
        eigvecs_gpu = fix_sign(eigvecs_gpu, 0)
        eigvecs_gpu = fix_sign(eigvecs_gpu, 0)
        
        # Store the eigenvalues directly in the CuPy array
        eigvals_all[i] = eigvals_gpu
        # Store the eigenvectors directly in the CuPy array
        eigvecs_all[i] = eigvecs_gpu
    
    return cp.asnumpy(eigvals_all), cp.asnumpy(eigvecs_all)

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
    output_dir = 'eigenvals_vs_basis1_basis2_centered_R_0'
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
    
    eigvals_all, eigvecs_all = getting_eigenvalues_gpu_drange(
        drange, H_thetas_drange
    )
    print(eigvals_all.shape) # (100, 5000, 4)
    print(eigvecs_all.shape) # (100, 5000, 4, 4)

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
        R_d_centered = R_d - R_0  # make sure R_0 is shape (3,)
        proj1 = R_d_centered @ basis1
        proj2 = R_d_centered @ basis2

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


    # Directory to save the plots
    gap_plot_dir = "energy_gap_plots"
    os.makedirs(gap_plot_dir, exist_ok=True)

    # Loop over all d values
    for i, d in enumerate(drange):
        eigvals_d = eigvals_all[i]  # shape: (5000, 4)

        # Sort eigenvalues just to be sure (optional if already sorted)
        eigvals_d_sorted = np.sort(eigvals_d, axis=1)

        # Calculate all energy gaps
        gaps = {
            'E2-E1': eigvals_d_sorted[:, 1] - eigvals_d_sorted[:, 0],
            'E3-E2': eigvals_d_sorted[:, 2] - eigvals_d_sorted[:, 1],
            'E4-E3': eigvals_d_sorted[:, 3] - eigvals_d_sorted[:, 2]
        }

        # Plot each energy gap
        for gap_name, gap_values in gaps.items():
            plt.figure(figsize=(10, 4))
            plt.plot(gap_values, color='navy')
            plt.title(f"Energy Gap {gap_name} (d={d:.2f})")
            plt.xlabel("Angle Index")
            plt.ylabel(f"{gap_name}")
            plt.grid(True)

            # Save the figure
            gap_path = os.path.join(gap_plot_dir, f'gap_{gap_name}_d{d:.2f}.png')
            plt.savefig(gap_path)
            print(f"Gap plot saved to: {gap_path}")

            plt.close()

    # Create directory for all gap plots
    all_gap_dir = os.path.join(gap_plot_dir, "all_gap_plots")
    os.makedirs(all_gap_dir, exist_ok=True)
    
    # Sort all eigenvalues at once
    eigvals_all_sorted = np.sort(eigvals_all, axis=2)
    
    # Calculate all energy gaps for all d values
    gaps = {
        'E2-E1': eigvals_all_sorted[:, :, 1] - eigvals_all_sorted[:, :, 0],
        'E3-E2': eigvals_all_sorted[:, :, 2] - eigvals_all_sorted[:, :, 1],
        'E4-E3': eigvals_all_sorted[:, :, 3] - eigvals_all_sorted[:, :, 2]
    }
    
    # Create meshgrid for d values and angle indices
    d_grid, theta_grid = np.meshgrid(drange, np.arange(num_points))
    
    # Plot each energy gap in 3D
    for gap_name, gap_values in gaps.items():
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(
            d_grid, theta_grid, gap_values.T,
            cmap='viridis', edgecolor='none', alpha=0.8
        )
        
        # Add color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set labels and title
        ax.set_xlabel('d value')
        ax.set_ylabel('Angle Index')
        ax.set_zlabel(f'Energy Gap {gap_name}')
        ax.set_title(f'Energy Gap {gap_name} vs d and Angle')
        
        # Save the plot
        plot_path = os.path.join(all_gap_dir, f'all_{gap_name}.png')
        plt.savefig(plot_path)
        print(f"3D plot saved to: {plot_path}")
        
        plt.close()
    
    print("All 3D energy gap plots have been generated.")

    # Project each eigenvector (not eigenvalue!) onto basis1 and basis2
    # Assuming eigvecs_all[i] is of shape (5000, 4, 4) — 4 eigenvectors per point
    projected_dir = os.path.join(output_dir, 'projected_eigenvalues')
    os.makedirs(projected_dir, exist_ok=True)

    # Define the 3D basis vectors orthogonal to the (1,1,1,0) direction
    basis1 = np.array([1, -1, 0, 0])
    basis2 = np.array([1, 1, -2, 0])
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)

    for i, d in enumerate(drange):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        eigvals_d = eigvals_all[i]         # shape: (5000, 4)
        eigvecs_d = eigvecs_all[i]         # shape: (5000, 4, 4)

        for j in range(4):
            # Project each eigenvector for this eigenvalue onto basis1 and basis2
            proj1 = []
            proj2 = []
            for k in range(eigvecs_d.shape[0]):
                vec = eigvecs_d[k, j, :]  # j-th eigenvector at point k
                proj1.append(np.dot(vec, basis1))
                proj2.append(np.dot(vec, basis2))
            
            # Plot points
            ax.scatter(
                proj1,
                proj2,
                eigvals_d[:, j],
                label=f'Eigenvalue {j+1}',
                alpha=0.7,
                s=10
            )
            
            # Plot lines connecting the points
            ax.plot(
                proj1,
                proj2,
                eigvals_d[:, j],
                alpha=0.5,
                color='gray'
            )

        ax.legend()
        ax.set_xlabel('Basis 1 projection')
        ax.set_ylabel('Basis 2 projection')
        ax.set_zlabel('Eigenvalues')
        ax.set_title(f'Eigenvalues vs Projected Coordinates (d={d:.2f})')

        plot_path = os.path.join(projected_dir, f'eigenvalues_projected_d{d:.2f}.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)

    print("All projected eigenvalue plots have been generated.")
