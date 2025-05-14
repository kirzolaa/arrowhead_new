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

def process_alpha_slice(alpha_idx, alpha, beta_range, R_0, d, hamiltonian):
    """
    Process a slice of the alpha-beta grid for a single alpha value
    
    Parameters:
    alpha_idx (int): Index of the current alpha value
    alpha (float): Current alpha value
    beta_range (numpy.ndarray): Range of beta values
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    hamiltonian (Hamiltonian): Hamiltonian instance
    
    Returns:
    tuple: (alpha_idx, R_vectors_slice, Va_values_slice, Vx_values_slice, Va_minus_Vx_values_slice)
    """
    num_beta = len(beta_range)
    R_vectors_slice = np.zeros((num_beta, 3))
    Va_values_slice = np.zeros((num_beta, 3))
    Vx_values_slice = np.zeros((num_beta, 3))
    Va_minus_Vx_values_slice = np.zeros((num_beta, 3))
    
    for j, beta in enumerate(beta_range):
        # Generate R vector
        R = generate_R_vectors(R_0, d, alpha, beta)
        R_vectors_slice[j] = R
        
        # Calculate Va and Vx for each component of the R vector
        Va_components = hamiltonian.Va(R)
        Vx_components = hamiltonian.Vx(R)
        
        # Calculate Va - Vx
        Va_minus_Vx = [Va - Vx for Va, Vx in zip(Va_components, Vx_components)]
        
        Va_values_slice[j] = Va_components
        Vx_values_slice[j] = Vx_components
        Va_minus_Vx_values_slice[j] = Va_minus_Vx
    
    return alpha_idx, R_vectors_slice, Va_values_slice, Vx_values_slice, Va_minus_Vx_values_slice

def search_conical_intersections(omega, aVx, aVa, x_shift, c_const, R_0, d, alpha_range, beta_range):
    """
    Search for conical intersections (CI) by plotting Vx-Va for different R vectors
    using multiprocessing to speed up calculations for large grids
    
    Parameters:
    omega (float): Angular frequency of the oscillator
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    alpha_range (numpy.ndarray): Range of alpha values
    beta_range (numpy.ndarray): Range of beta values
    
    Returns:
    tuple: (output_dir, R_vectors, Va_minus_Vx_values)
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'ci_search_results_{timestamp}'
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Create a dummy theta range for the Hamiltonian (not used in this function)
    theta_vals = np.array([0.0])
    
    # Create Hamiltonian instance
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    
    # Generate grid of alpha and beta values
    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
    
    # Initialize arrays to store results
    num_alpha = len(alpha_range)
    num_beta = len(beta_range)
    R_vectors = np.zeros((num_alpha, num_beta, 3))
    Va_values = np.zeros((num_alpha, num_beta, 3))
    Vx_values = np.zeros((num_alpha, num_beta, 3))
    Va_minus_Vx_values = np.zeros((num_alpha, num_beta, 3))
    
    # Determine number of processes to use (leave one core free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel computation")
    
    # Create a partial function with fixed parameters
    process_slice = partial(process_alpha_slice, 
                           beta_range=beta_range, 
                           R_0=R_0, 
                           d=d, 
                           hamiltonian=hamiltonian)
    
    # Process alpha slices in parallel
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a list of (alpha_idx, alpha) tuples to process
        alpha_items = list(enumerate(alpha_range))
        
        # Use tqdm to show progress
        for result in tqdm.tqdm(pool.starmap(process_slice, alpha_items), total=len(alpha_items), desc="Processing grid"):
            results.append(result)
    
    # Combine results
    for alpha_idx, R_slice, Va_slice, Vx_slice, Va_minus_Vx_slice in results:
        R_vectors[alpha_idx] = R_slice
        Va_values[alpha_idx] = Va_slice
        Vx_values[alpha_idx] = Vx_slice
        Va_minus_Vx_values[alpha_idx] = Va_minus_Vx_slice
    
    # Plot Va - Vx as a function of alpha and beta for each component
    for component in range(3):
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(alpha_grid, beta_grid, Va_minus_Vx_values[:, :, component].T, 
                     levels=50, cmap='viridis')
        plt.colorbar(contour, label=f'Va - Vx (Component {component})')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.title(f'Va - Vx (Component {component}) vs Alpha and Beta')
        plt.grid(True)
        plt.savefig(f'{output_dir}/plots/Va_minus_Vx_component_{component}.png')
    
    # Find where Va - Vx is close to zero for all components (potential CIs)
    tolerance = 0.01  # Adjust based on your needs
    potential_cis = []
    
    for i in range(num_alpha):
        for j in range(num_beta):
            # Check if Va - Vx is close to zero for all components
            if all(abs(val) < tolerance for val in Va_minus_Vx_values[i, j]):
                potential_cis.append((alpha_range[i], beta_range[j], R_vectors[i, j], Va_minus_Vx_values[i, j]))
    
    # Save potential CIs to a file
    if potential_cis:
        with open(f'{output_dir}/potential_cis.txt', 'w') as f:
            f.write("Potential Conical Intersections (CIs):\n")
            f.write("Alpha, Beta, R_vector, Va-Vx values\n")
            for ci in potential_cis:
                f.write(f"{ci[0]}, {ci[1]}, {ci[2]}, {ci[3]}\n")
        print(f"Found {len(potential_cis)} potential conical intersections.")
    else:
        print("No potential conical intersections found within the tolerance.")
    
    # Plot 3D visualization of Va - Vx = 0 isosurfaces
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for 3D plotting
    alpha_3d, beta_3d, component_3d = np.meshgrid(alpha_range, beta_range, np.arange(3), indexing='ij')
    
    # Reshape Va_minus_Vx_values for 3D plotting
    Va_minus_Vx_3d = Va_minus_Vx_values.reshape(-1)
    
    # Create scatter plot with color based on Va - Vx values
    scatter = ax.scatter(alpha_3d.flatten(), beta_3d.flatten(), component_3d.flatten(), 
                        c=Va_minus_Vx_3d, cmap='coolwarm', alpha=0.5)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Va - Vx')
    
    # Mark potential CIs
    if potential_cis:
        ci_alphas = [ci[0] for ci in potential_cis]
        ci_betas = [ci[1] for ci in potential_cis]
        ci_components = np.zeros(len(potential_cis))  # Place at component 0 for visibility
        ax.scatter(ci_alphas, ci_betas, ci_components, color='red', s=100, marker='*', label='Potential CIs')
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Component')
    ax.set_title('3D Visualization of Va - Vx Values')
    ax.legend()
    plt.savefig(f'{output_dir}/plots/Va_minus_Vx_3D.png')
    
    # Plot in d-theta coordinates
    # Convert alpha-beta to d-theta
    d_values = np.sqrt(alpha_grid**2 + beta_grid**2) * d
    theta_values = np.arctan2(beta_grid, alpha_grid)
    
    for component in range(3):
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(d_values, theta_values, Va_minus_Vx_values[:, :, component].T, 
                     levels=50, cmap='viridis')
        plt.colorbar(contour, label=f'Va - Vx (Component {component})')
        plt.xlabel('d')
        plt.ylabel('theta (radians)')
        plt.title(f'Va - Vx (Component {component}) vs d and theta')
        plt.grid(True)
        plt.savefig(f'{output_dir}/plots/Va_minus_Vx_d_theta_component_{component}.png')
    
    # Save the data for further analysis
    np.save(f'{output_dir}/R_vectors.npy', R_vectors)
    np.save(f'{output_dir}/Va_values.npy', Va_values)
    np.save(f'{output_dir}/Vx_values.npy', Vx_values)
    np.save(f'{output_dir}/Va_minus_Vx_values.npy', Va_minus_Vx_values)
    np.save(f'{output_dir}/alpha_grid.npy', alpha_grid)
    np.save(f'{output_dir}/beta_grid.npy', beta_grid)
    np.save(f'{output_dir}/d_values.npy', d_values)
    np.save(f'{output_dir}/theta_values.npy', theta_values)
    
    print(f"Va - Vx plots saved to {output_dir}/plots directory.")
    return output_dir, R_vectors, Va_minus_Vx_values

def find_symmetric_solutions(R_0, d, potential_cis):
    """
    Find symmetric solutions for R0, R1, R2 where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2)
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    potential_cis (list): List of potential conical intersections
    
    Returns:
    list: List of symmetric solutions
    """
    symmetric_solutions = []
    
    # We know that alpha = beta = 0 is one solution (as mentioned in the notes)
    origin_solution = (0, 0, R_0, np.zeros(3))
    
    # Check for symmetric solutions
    if len(potential_cis) >= 3:
        # Try all combinations of 3 points
        from itertools import combinations
        for combo in combinations(potential_cis, 3):
            # Check if they form a symmetric pattern
            # For simplicity, we'll check if they form an equilateral triangle in alpha-beta space
            alpha_beta_points = [(ci[0], ci[1]) for ci in combo]
            
            # Calculate distances between points
            distances = []
            for i in range(3):
                for j in range(i+1, 3):
                    p1 = alpha_beta_points[i]
                    p2 = alpha_beta_points[j]
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    distances.append(dist)
            
            # Check if all distances are approximately equal (equilateral triangle)
            if np.std(distances) < 0.01:  # Adjust tolerance as needed
                symmetric_solutions.append(combo)
    
    return symmetric_solutions

if __name__ == "__main__":
    # Set parameters
    omega = 1.0
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    R_0 = np.array([0, 0, 0])
    d = 0.1
    
    # Define range for alpha and beta
    # For very large grids, consider using a coarser grid first to identify regions of interest
    # then refine the search in those regions
    print("Warning: Using a 10000x10000 grid will require significant memory and computation time.")
    print("Consider using a smaller grid (e.g., 500x500) for initial exploration.")
    
    # Ask user for confirmation or alternative grid size
    response = input("Enter grid size (e.g., '500' for 500x500 grid) or press Enter to continue with 10000x10000: ")
    
    try:
        if response.strip():
            grid_size = int(response.strip())
            print(f"Using {grid_size}x{grid_size} grid")
        else:
            grid_size = 10000
            print(f"Continuing with {grid_size}x{grid_size} grid")
    except ValueError:
        grid_size = 500
        print(f"Invalid input. Using default {grid_size}x{grid_size} grid")
    
    alpha_range = np.linspace(-1.0, 1.0, grid_size)
    beta_range = np.linspace(-1.0, 1.0, grid_size)
    
    # Search for conical intersections with multiprocessing
    print("Starting conical intersection search with multiprocessing...")
    output_dir, R_vectors, Va_minus_Vx_values = search_conical_intersections(
        omega, aVx, aVa, x_shift, c_const, R_0, d, alpha_range, beta_range
    )
    
    print(f"Search complete. Results saved to {output_dir}")
    plt.show()
