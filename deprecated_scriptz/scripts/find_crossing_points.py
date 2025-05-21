#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import hbar
import datetime
import multiprocessing
from functools import partial
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle

# Import the Hamiltonian class from new_bph.py
from new_bph import Hamiltonian

def generate_basis_vectors():
    """
    Generate normalized basis vectors orthogonal to the (1,1,1) direction
    
    Returns:
    tuple: (basis1, basis2) - The two basis vectors
    """
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([+2, -1, -1])  # First basis vector
    basis2 = np.array([ 0, -1, +1])   # Second basis vector
    
    # Normalize the basis vectors
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    return basis1, basis2

def generate_R_vectors(R_0, d, alpha, beta):
    """
    Generate R vectors using the formula R = R_0 + d * (alpha*basis1 + beta*basis2)
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    alpha (float): Coefficient for basis1
    beta (float): Coefficient for basis2
    
    Returns:
    numpy.ndarray: The resulting R vector
    """
    basis1, basis2 = generate_basis_vectors()
    return R_0 + d * (alpha * basis1 + beta * basis2)

def find_crossing_points(omega, aVx, aVa, x_shift, c_const, R_0, d, alpha_range, beta_range, grid_size=100):
    """
    Find points where the three Va-Vx differences cross each other
    
    Parameters:
    omega (float): Angular frequency of the oscillator
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    alpha_range (tuple): (min_alpha, max_alpha) range to search
    beta_range (tuple): (min_beta, max_beta) range to search
    grid_size (int): Size of the grid for alpha and beta
    
    Returns:
    list: List of (alpha, beta) points where the three differences cross
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'crossing_points_search_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Create a dummy theta range for the Hamiltonian (not used in this function)
    theta_vals = np.array([0.0])
    
    # Create Hamiltonian instance
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    
    # Generate grid of alpha and beta values
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_vals = np.linspace(beta_range[0], beta_range[1], grid_size)
    alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)
    
    # Initialize arrays to store results
    Va_minus_Vx_values = np.zeros((grid_size, grid_size, 3))
    R_vectors = np.zeros((grid_size, grid_size, 3))
    
    # Calculate Va-Vx for each point in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = alpha_vals[i]
            beta = beta_vals[j]
            
            # Generate R vector
            R = generate_R_vectors(R_0, d, alpha, beta)
            R_vectors[i, j] = R
            
            # Calculate Va and Vx for each component of the R vector
            Va_components = hamiltonian.Va(R)
            Vx_components = hamiltonian.Vx(R)
            
            # Calculate Va - Vx
            Va_minus_Vx = [Va - Vx for Va, Vx in zip(Va_components, Vx_components)]
            Va_minus_Vx_values[i, j] = Va_minus_Vx
    
    # Find where the three differences are approximately equal
    # This indicates a crossing point
    crossing_points = []
    tolerance = 1e-4
    
    for i in range(grid_size):
        for j in range(grid_size):
            diffs = Va_minus_Vx_values[i, j]
            # Check if all three differences are approximately equal
            if (np.abs(diffs[0] - diffs[1]) < tolerance and 
                np.abs(diffs[1] - diffs[2]) < tolerance and 
                np.abs(diffs[0] - diffs[2]) < tolerance):
                crossing_points.append((alpha_vals[i], beta_vals[j], diffs[0]))
    
    # Create plots to visualize the differences
    # Plot each component of Va-Vx as a contour plot
    for component in range(3):
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(alpha_grid, beta_grid, Va_minus_Vx_values[:, :, component].T, 
                     levels=50, cmap='viridis')
        plt.colorbar(contour, label=f'Va - Vx (Component {component+1})')
        
        # Mark the crossing points
        if crossing_points:
            cross_alphas = [p[0] for p in crossing_points]
            cross_betas = [p[1] for p in crossing_points]
            plt.scatter(cross_alphas, cross_betas, color='red', marker='x', s=100, 
                       label=f'Crossing Points ({len(crossing_points)})')
        
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.title(f'Va - Vx (Component {component+1}) vs Alpha and Beta')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{output_dir}/plots/Va_minus_Vx_component_{component+1}.png')
    
    # Create a plot showing the difference between the components
    plt.figure(figsize=(10, 8))
    diff_01 = np.abs(Va_minus_Vx_values[:, :, 0] - Va_minus_Vx_values[:, :, 1]).T
    diff_12 = np.abs(Va_minus_Vx_values[:, :, 1] - Va_minus_Vx_values[:, :, 2]).T
    diff_02 = np.abs(Va_minus_Vx_values[:, :, 0] - Va_minus_Vx_values[:, :, 2]).T
    
    # Sum of all differences - should be close to zero at crossing points
    total_diff = (diff_01 + diff_12 + diff_02)
    
    contour = plt.contourf(alpha_grid, beta_grid, total_diff, 
                 levels=50, cmap='viridis')
    plt.colorbar(contour, label='Sum of differences between components')
    
    # Mark the crossing points
    if crossing_points:
        cross_alphas = [p[0] for p in crossing_points]
        cross_betas = [p[1] for p in crossing_points]
        plt.scatter(cross_alphas, cross_betas, color='red', marker='x', s=100, 
                   label=f'Crossing Points ({len(crossing_points)})')
    
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Sum of Differences Between Va-Vx Components')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/plots/sum_of_differences.png')
    
    # Create a 3D plot showing the three Va-Vx surfaces
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for 3D plotting
    X, Y = np.meshgrid(alpha_vals, beta_vals)
    
    # Plot each component as a surface
    surf1 = ax.plot_surface(X, Y, Va_minus_Vx_values[:, :, 0].T, 
                           alpha=0.7, cmap='viridis', label='Component 1')
    surf2 = ax.plot_surface(X, Y, Va_minus_Vx_values[:, :, 1].T, 
                           alpha=0.7, cmap='plasma', label='Component 2')
    surf3 = ax.plot_surface(X, Y, Va_minus_Vx_values[:, :, 2].T, 
                           alpha=0.7, cmap='inferno', label='Component 3')
    
    # Add a proxy artist for the legend
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc='blue')
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc='green')
    proxy3 = plt.Rectangle((0, 0), 1, 1, fc='red')
    
    # Mark the crossing points in 3D
    if crossing_points:
        cross_alphas = [p[0] for p in crossing_points]
        cross_betas = [p[1] for p in crossing_points]
        cross_values = [p[2] for p in crossing_points]
        ax.scatter(cross_alphas, cross_betas, cross_values, color='red', marker='o', s=100)
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Va - Vx')
    ax.set_title('3D Visualization of Va-Vx Components')
    ax.legend([proxy1, proxy2, proxy3], ['Component 1', 'Component 2', 'Component 3'])
    plt.savefig(f'{output_dir}/plots/3D_components.png')
    
    # Save the crossing points to a file
    if crossing_points:
        with open(f'{output_dir}/crossing_points.txt', 'w') as f:
            f.write("Alpha, Beta, Va-Vx Value\n")
            for point in crossing_points:
                f.write(f"{point[0]}, {point[1]}, {point[2]}\n")
        
        print(f"Found {len(crossing_points)} crossing points:")
        for i, point in enumerate(crossing_points):
            print(f"Point {i+1}: alpha={point[0]:.6f}, beta={point[1]:.6f}, Va-Vx={point[2]:.6f}")
    else:
        print("No crossing points found within the tolerance.")
    
    # Create plots showing the difference between each pair of components
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    contour1 = plt.contourf(alpha_grid, beta_grid, diff_01, levels=50, cmap='viridis')
    plt.colorbar(contour1, label='|Component 1 - Component 2|')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Difference between Components 1 and 2')
    
    plt.subplot(1, 3, 2)
    contour2 = plt.contourf(alpha_grid, beta_grid, diff_12, levels=50, cmap='viridis')
    plt.colorbar(contour2, label='|Component 2 - Component 3|')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Difference between Components 2 and 3')
    
    plt.subplot(1, 3, 3)
    contour3 = plt.contourf(alpha_grid, beta_grid, diff_02, levels=50, cmap='viridis')
    plt.colorbar(contour3, label='|Component 1 - Component 3|')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Difference between Components 1 and 3')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/component_differences.png')
    
    # Create a plot showing the three R vectors at each crossing point
    if crossing_points:
        for i, point in enumerate(crossing_points):
            alpha, beta, _ = point
            
            # Generate the three R vectors at the symmetric theta points
            theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
            R_triple = []
            
            for theta in theta_triple:
                # Convert from (alpha, beta) to (d, theta) coordinates
                # For this specific point
                d_point = d * np.sqrt(alpha**2 + beta**2)
                theta_point = np.arctan2(beta, alpha)
                
                # Rotate by the current theta
                theta_rotated = theta_point + theta
                alpha_rotated = d_point * np.cos(theta_rotated) / d
                beta_rotated = d_point * np.sin(theta_rotated) / d
                
                # Generate the R vector
                R = generate_R_vectors(R_0, d, alpha_rotated, beta_rotated)
                R_triple.append(R)
            
            R_triple = np.array(R_triple)
            
            # Calculate Va-Vx for each R vector
            Va_minus_Vx_triple = []
            for R in R_triple:
                Va_components = hamiltonian.Va(R)
                Vx_components = hamiltonian.Vx(R)
                Va_minus_Vx = [Va - Vx for Va, Vx in zip(Va_components, Vx_components)]
                Va_minus_Vx_triple.append(Va_minus_Vx)
            
            Va_minus_Vx_triple = np.array(Va_minus_Vx_triple)
            
            # Plot the three R vectors in 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the three R vectors
            ax.scatter(R_triple[:, 0], R_triple[:, 1], R_triple[:, 2], 
                      c=['r', 'g', 'b'], marker='o', s=100)
            
            # Connect the three points to form a triangle
            ax.plot([R_triple[0, 0], R_triple[1, 0]], 
                   [R_triple[0, 1], R_triple[1, 1]], 
                   [R_triple[0, 2], R_triple[1, 2]], 'k-')
            ax.plot([R_triple[1, 0], R_triple[2, 0]], 
                   [R_triple[1, 1], R_triple[2, 1]], 
                   [R_triple[1, 2], R_triple[2, 2]], 'k-')
            ax.plot([R_triple[2, 0], R_triple[0, 0]], 
                   [R_triple[2, 1], R_triple[0, 1]], 
                   [R_triple[2, 2], R_triple[0, 2]], 'k-')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Crossing Point {i+1}: R Vectors at Symmetric Theta Points')
            plt.savefig(f'{output_dir}/plots/crossing_point_{i+1}_R_vectors.png')
            
            # Plot the Va-Vx values for each component
            plt.figure(figsize=(10, 6))
            theta_labels = ['0', '2π/3', '4π/3']
            
            for component in range(3):
                plt.plot(range(3), Va_minus_Vx_triple[:, component], 'o-', 
                        label=f'Component {component+1}')
            
            plt.xticks(range(3), theta_labels)
            plt.xlabel('Theta')
            plt.ylabel('Va - Vx')
            plt.title(f'Crossing Point {i+1}: Va-Vx Values at Symmetric Theta Points')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_dir}/plots/crossing_point_{i+1}_Va_minus_Vx.png')
    
    return crossing_points, Va_minus_Vx_values, R_vectors, output_dir

if __name__ == "__main__":
    # Set parameters
    omega = 1.0
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.0
    c_const = 0.1
    R_0 = np.array([0, 0, 0])
    d = 0.005  # Use the value we found in the previous script
    
    # Define search range for alpha and beta
    # We know one solution is at alpha = beta = 0
    # Let's search around that point
    alpha_range = (-0.5, 0.5)
    beta_range = (-0.5, 0.5)
    
    print(f"Searching for crossing points with d = {d}, alpha ∈ [{alpha_range[0]}, {alpha_range[1]}], beta ∈ [{beta_range[0]}, {beta_range[1]}]")
    
    # Find crossing points
    crossing_points, Va_minus_Vx_values, R_vectors, output_dir = find_crossing_points(
        omega, aVx, aVa, x_shift, c_const, R_0, d, alpha_range, beta_range, grid_size=200
    )
    
    print(f"Results saved to {output_dir}")
    plt.show()
