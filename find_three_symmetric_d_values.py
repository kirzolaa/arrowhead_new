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

def find_symmetric_triple_d_values(omega, aVx, aVa, x_shift, c_const, R_0, d_range, epsilon=1e-5):
    """
    Find three d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2) is exactly satisfied
    for three symmetric points R0, R1, R2 at theta = 0, 2π/3, 4π/3
    
    Parameters:
    omega (float): Angular frequency of the oscillator
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    R_0 (numpy.ndarray): The origin vector
    d_range (numpy.ndarray): Range of d values to search
    epsilon (float): Tolerance for equality comparison
    
    Returns:
    list: List of d values where the equation is satisfied
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'symmetric_triple_d_values_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Define the three symmetric theta values
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    # Store results for all d values
    all_delta_Vs = []
    all_d_vals = []
    
    # Store the three d values where the equation is satisfied
    symmetric_d_values = []
    
    # Search through d values
    for d in tqdm.tqdm(d_range, desc="Searching for symmetric triple d values"):
        h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
        R_thetas = np.array(h.R_thetas())
        Va_vals = np.array(h.Va_theta_vals(R_thetas))  # shape (3, 3)
        Vx_vals = np.array(h.Vx_theta_vals(R_thetas))  # shape (3, 3)

        # Compute ΔV = Va - Vx componentwise at θ = 0, 2π/3, 4π/3
        delta_Vs = Va_vals - Vx_vals  # shape (3, 3)
        
        # Store results for plotting
        all_delta_Vs.append(delta_Vs)
        all_d_vals.append(d)

        # For each component (i=0,1,2), check if ΔV[i] is same across all 3 points
        componentwise_equal = [
            np.allclose(delta_Vs[:, i], delta_Vs[0, i], atol=epsilon)
            for i in range(3)
        ]

        if all(componentwise_equal):
            print(f"🎯 Found exact triple-point degeneracy for d = {d:.6f}")
            print(f"ΔV values per point:\n{delta_Vs}")
            symmetric_d_values.append(d)
            
            # If we've found 3 d values, we can stop
            if len(symmetric_d_values) >= 3:
                break
    
    # Convert results to numpy arrays
    all_delta_Vs = np.array(all_delta_Vs)
    all_d_vals = np.array(all_d_vals)
    
    # Plot the results
    if len(all_d_vals) > 0:
        # Calculate how close each d value is to having equal delta_Vs
        closeness_scores = []
        for i, delta_Vs in enumerate(all_delta_Vs):
            # For each component, calculate the standard deviation across the three points
            component_stds = [np.std(delta_Vs[:, j]) for j in range(3)]
            # Average the standard deviations
            avg_std = np.mean(component_stds)
            closeness_scores.append(avg_std)
        
        closeness_scores = np.array(closeness_scores)
        
        # Plot the closeness score vs d
        plt.figure(figsize=(10, 6))
        plt.plot(all_d_vals, closeness_scores)
        
        # Mark the found d values
        if symmetric_d_values:
            for d in symmetric_d_values:
                idx = np.abs(all_d_vals - d).argmin()
                plt.scatter([all_d_vals[idx]], [closeness_scores[idx]], color='red', s=100)
        
        plt.xlabel('d')
        plt.ylabel('Average Standard Deviation of ΔV Components')
        plt.title('Closeness to Symmetric Triple Point')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'closeness_scores.png'))
        plt.close()
    
    # For each found d value, create detailed plots
    for d in symmetric_d_values:
        # Create a directory for this d value
        d_dir = os.path.join(output_dir, f'd_{d:.6f}')
        os.makedirs(d_dir, exist_ok=True)
        
        # Create Hamiltonian with this d value
        h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
        R_thetas = np.array(h.R_thetas())
        Va_vals = np.array(h.Va_theta_vals(R_thetas))
        Vx_vals = np.array(h.Vx_theta_vals(R_thetas))
        delta_Vs = Va_vals - Vx_vals
        
        # Plot the three R vectors in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the three R vectors
        ax.scatter(R_thetas[:, 0], R_thetas[:, 1], R_thetas[:, 2], 
                  c=['r', 'g', 'b'], marker='o', s=100)
        
        # Connect the three points to form a triangle
        ax.plot([R_thetas[0, 0], R_thetas[1, 0]], 
               [R_thetas[0, 1], R_thetas[1, 1]], 
               [R_thetas[0, 2], R_thetas[1, 2]], 'k-')
        ax.plot([R_thetas[1, 0], R_thetas[2, 0]], 
               [R_thetas[1, 1], R_thetas[2, 1]], 
               [R_thetas[1, 2], R_thetas[2, 2]], 'k-')
        ax.plot([R_thetas[2, 0], R_thetas[0, 0]], 
               [R_thetas[2, 1], R_thetas[0, 1]], 
               [R_thetas[2, 2], R_thetas[0, 2]], 'k-')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'R Vectors at Symmetric Theta Points (d = {d:.6f})')
        plt.savefig(os.path.join(d_dir, 'R_vectors_3d.png'))
        plt.close()
        
        # Plot the Va-Vx values for each component
        plt.figure(figsize=(10, 6))
        theta_labels = ['0', '2π/3', '4π/3']
        
        for component in range(3):
            plt.plot(range(3), delta_Vs[:, component], 'o-', 
                    label=f'Component {component+1}')
        
        plt.xticks(range(3), theta_labels)
        plt.xlabel('Theta')
        plt.ylabel('Va - Vx')
        plt.title(f'Va-Vx Values at Symmetric Theta Points (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(d_dir, 'Va_minus_Vx.png'))
        plt.close()
        
        # Plot Va and Vx at full resolution over theta ∈ [0, 2π]
        theta_full = np.linspace(0, 2 * np.pi, 1000)
        h_full = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_full)
        R_full = np.array(h_full.R_thetas())
        Va_full = np.array(h_full.Va_theta_vals(R_full))  # shape (1000, 3)
        Vx_full = np.array(h_full.Vx_theta_vals(R_full))  # shape (1000, 3)
        
        # Plot each component of Va
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, Va_full[:, i], label=f'Component {i+1}')
        plt.xlabel('Theta')
        plt.ylabel('Va')
        plt.title(f'Va Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(d_dir, 'Va_components.png'))
        plt.close()
        
        # Plot each component of Vx
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, Vx_full[:, i], label=f'Component {i+1}')
        plt.xlabel('Theta')
        plt.ylabel('Vx')
        plt.title(f'Vx Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(d_dir, 'Vx_components.png'))
        plt.close()
        
        # Plot Va-Vx for each component
        delta_V_full = Va_full - Vx_full
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, delta_V_full[:, i], label=f'Component {i+1}')
        plt.xlabel('Theta')
        plt.ylabel('Va - Vx')
        plt.title(f'Va-Vx vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(d_dir, 'Va_minus_Vx_full.png'))
        plt.close()
        
        # Mark the three theta points on the Va-Vx plot
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, delta_V_full[:, i], label=f'Component {i+1}')
            plt.scatter(theta_triple, delta_Vs[:, i], color='red', s=100)
        plt.xlabel('Theta')
        plt.ylabel('Va - Vx')
        plt.title(f'Va-Vx vs Theta with Symmetric Points (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(d_dir, 'Va_minus_Vx_with_points.png'))
        plt.close()
    
    # Save the found d values to a file
    with open(os.path.join(output_dir, 'symmetric_d_values.txt'), 'w') as f:
        f.write("d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2):\n")
        for d in symmetric_d_values:
            f.write(f"{d:.10f}\n")
    
    return symmetric_d_values, output_dir

# Define a function to process a chunk of d values
def process_chunk(args):
    d_chunk, omega, aVx, aVa, x_shift, c_const, R_0, epsilon = args
    
    # Define the three symmetric theta values
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    # Store the d values where the equation is satisfied
    symmetric_d_values = []
    
    # Search through d values
    for d in d_chunk:
        h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
        R_thetas = np.array(h.R_thetas())
        Va_vals = np.array(h.Va_theta_vals(R_thetas))  # shape (3, 3)
        Vx_vals = np.array(h.Vx_theta_vals(R_thetas))  # shape (3, 3)

        # Compute ΔV = Va - Vx componentwise at θ = 0, 2π/3, 4π/3
        delta_Vs = Va_vals - Vx_vals  # shape (3, 3)

        # For each component (i=0,1,2), check if ΔV[i] is same across all 3 points
        componentwise_equal = [
            np.allclose(delta_Vs[:, i], delta_Vs[0, i], atol=epsilon)
            for i in range(3)
        ]

        if all(componentwise_equal):
            print(f"🎯 Found exact triple-point degeneracy for d = {d:.6f}")
            print(f"ΔV values per point:\n{delta_Vs}")
            symmetric_d_values.append((d, delta_Vs))
    
    return symmetric_d_values

def parallel_search_symmetric_triple_d_values(omega, aVx, aVa, x_shift, c_const, R_0, d_start, d_end, num_points, epsilon=1e-5, num_processes=None):
    """
    Search for three d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2) is exactly satisfied
    using parallel processing
    
    Parameters:
    omega, aVx, aVa, x_shift, c_const, R_0: Parameters for the Hamiltonian
    d_start (float): Start of d range
    d_end (float): End of d range
    num_points (int): Number of points in the range
    epsilon (float): Tolerance for equality comparison
    num_processes (int): Number of processes to use, defaults to CPU count - 1
    
    Returns:
    list: List of d values where the equation is satisfied
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} processes for parallel search")
    
    # Create d range
    d_range = np.linspace(d_start, d_end, num_points)
    
    # Split the d range into chunks for parallel processing
    chunk_size = len(d_range) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    d_chunks = []
    for i in range(0, len(d_range), chunk_size):
        chunk = d_range[i:i+chunk_size]
        d_chunks.append((chunk, omega, aVx, aVa, x_shift, c_const, R_0, epsilon))
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, d_chunks)
    
    # Combine results
    symmetric_d_values = []
    for result in results:
        symmetric_d_values.extend(result)
    
    # Sort by d value
    symmetric_d_values.sort(key=lambda x: x[0])
    
    # Extract just the d values
    d_values = [d for d, _ in symmetric_d_values]
    
    # If we found at least 3 d values, create plots for them
    if len(d_values) >= 3:
        # Take the first 3 d values
        d_values = d_values[:3]
        
        # Create plots for each d value
        find_symmetric_triple_d_values(omega, aVx, aVa, x_shift, c_const, R_0, np.array(d_values), epsilon)
    
    return d_values

if __name__ == "__main__":
    # Set parameters
    omega = 1.0
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.0
    c_const = 0.1
    R_0 = np.array([0, 0, 0])
    
    # Ask user which search mode to use
    print("Choose search mode:")
    print("1. Search for three d values in a wide range (d = 0.001 to 0.1)")
    print("2. Search for three d values in a narrow range around d = 0.005 (d = 0.001 to 0.01)")
    print("3. Custom range search")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        # Wide range search
        d_start = 0.001
        d_end = 0.1
        num_points = 10000
        epsilon = 1e-4
    elif choice == "2":
        # Narrow range search around d = 0.005
        d_start = 0.001
        d_end = 0.01
        num_points = 10000
        epsilon = 1e-5  # Higher precision
    elif choice == "3":
        # Custom range search
        d_start = float(input("Enter start d value: "))
        d_end = float(input("Enter end d value: "))
        num_points = int(input("Enter number of points: "))
        epsilon = float(input("Enter tolerance (epsilon): "))
    else:
        # Default to narrow range search
        print("Invalid choice. Using narrow range search.")
        d_start = 0.001
        d_end = 0.01
        num_points = 10000
        epsilon = 1e-5
    
    print(f"Searching for three symmetric d values with d ∈ [{d_start}, {d_end}], {num_points} points, epsilon = {epsilon}")
    
    # Use parallel search
    d_values = parallel_search_symmetric_triple_d_values(
        omega, aVx, aVa, x_shift, c_const, R_0, d_start, d_end, num_points, epsilon
    )
    
    if len(d_values) >= 3:
        print(f"Found three symmetric d values: {d_values[:3]}")
    elif len(d_values) > 0:
        print(f"Found {len(d_values)} symmetric d values: {d_values}")
    else:
        print("No symmetric d values found within the given tolerance.")
