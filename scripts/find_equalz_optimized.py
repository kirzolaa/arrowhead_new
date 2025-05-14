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

def process_d_chunk(args):
    """
    Process a chunk of d values to find symmetric triple points
    
    Parameters:
    args: tuple containing (d_chunk, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)
    
    Returns:
    list: List of tuples (d, delta_Vs) for d values where the equation is satisfied
    """
    d_chunk, aVx, aVa, x_shift, c_const, omega, R_0, epsilon = args
    
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
            symmetric_d_values.append((d, delta_Vs))
    
    return symmetric_d_values

def find_exact_symmetric_triple_d(d_range, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5, save_plots=True, num_processes=None):
    """
    Find d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2) is exactly satisfied
    for three symmetric points R0, R1, R2 at theta = 0, 2π/3, 4π/3
    
    Uses multiprocessing for efficient searching across large d ranges
    
    Parameters:
    d_range (numpy.ndarray): Range of d values to search
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    omega (float): Angular frequency of the oscillator
    R_0 (tuple or numpy.ndarray): The origin vector
    epsilon (float): Tolerance for equality comparison
    save_plots (bool): Whether to save plots
    num_processes (int): Number of processes to use, defaults to CPU count - 1
    
    Returns:
    list: List of d values where the equation is satisfied
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} processes for parallel search across {len(d_range)} d values")
    
    # Convert R_0 to numpy array if it's a tuple
    R_0 = np.array(R_0)
    
    # Split the d range into chunks for parallel processing
    chunk_size = len(d_range) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    d_chunks = []
    for i in range(0, len(d_range), chunk_size):
        chunk = d_range[i:i+chunk_size]
        d_chunks.append((chunk, aVx, aVa, x_shift, c_const, omega, R_0, epsilon))
    
    # Process chunks in parallel
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap(process_d_chunk, d_chunks), total=len(d_chunks), desc="Searching for symmetric triple points"):
            results.extend(result)
    
    # Sort results by d value
    results.sort(key=lambda x: x[0])
    
    # Extract just the d values
    d_values = [d for d, _ in results]
    
    if results:
        print(f"Found {len(results)} symmetric triple points")
        
        # Create timestamp for output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f'symmetric_triple_points_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the found d values to a file
        with open(os.path.join(output_dir, 'symmetric_d_values.txt'), 'w') as f:
            f.write("d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2):\n")
            for d, delta_Vs in results:
                f.write(f"{d:.10f}: {np.mean(delta_Vs[:, 0]):.10f}\n")
        
        if save_plots:
            # Create plots for each found d value
            for d, delta_Vs in results:
                # Create a directory for this d value
                plot_dir = f'plots_triple_d_{d:.6f}'
                os.makedirs(plot_dir, exist_ok=True)
                
                print(f"🎯 Exact triple-point degeneracy found for d = {d:.6f}")
                print(f"ΔV values at θ = 0, 2π/3, 4π/3:\n{delta_Vs}")
                
                # Define the three symmetric theta values
                theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
                
                # Plot Va and Vx at full resolution over theta ∈ [0, 2π]
                theta_full = np.linspace(0, 2 * np.pi, 1000)
                h_full = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_full)
                R_full = np.array(h_full.R_thetas())
                Va_full = np.array(h_full.Va_theta_vals(R_full))  # shape (1000, 3)
                Vx_full = np.array(h_full.Vx_theta_vals(R_full))  # shape (1000, 3)
                
                # Calculate Va-Vx
                delta_full = Va_full - Vx_full
                
                # Plot Va-Vx for each component with vertical lines at the three theta points
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, delta_full[:, i], label=fr"$\Delta V_{{{i+1}}}$")
                for θ in theta_triple:
                    plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
                plt.xlabel(r"$\theta$")
                plt.ylabel(r"$V_a - V_x$")
                plt.title(f"ΔV components at d = {d:.6f}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"deltaV_full_{d:.6f}.png"))
                plt.close()
                
                # Plot the three R vectors in 3D
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Generate the R vectors at the three theta points
                h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
                R_thetas = np.array(h.R_thetas())
                
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
                plt.savefig(os.path.join(plot_dir, 'R_vectors_3d.png'))
                plt.close()
                
                # Plot Va and Vx components separately
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, Va_full[:, i], label=f'Va component {i+1}')
                plt.xlabel('Theta')
                plt.ylabel('Va')
                plt.title(f'Va Components vs Theta (d = {d:.6f})')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, 'Va_components.png'))
                plt.close()
                
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, Vx_full[:, i], label=f'Vx component {i+1}')
                plt.xlabel('Theta')
                plt.ylabel('Vx')
                plt.title(f'Vx Components vs Theta (d = {d:.6f})')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, 'Vx_components.png'))
                plt.close()
                
                # Create a combined plot with all components in separate subplots
                fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                
                # Plot Va components
                for i in range(3):
                    axes[0].plot(theta_full, Va_full[:, i], label=f'Va component {i+1}')
                axes[0].set_ylabel('Va')
                axes[0].set_title(f'Va Components at d = {d:.6f}')
                axes[0].legend()
                axes[0].grid(True)
                
                # Plot Vx components
                for i in range(3):
                    axes[1].plot(theta_full, Vx_full[:, i], label=f'Vx component {i+1}')
                axes[1].set_ylabel('Vx')
                axes[1].set_title(f'Vx Components at d = {d:.6f}')
                axes[1].legend()
                axes[1].grid(True)
                
                # Plot deltaV components
                for i in range(3):
                    axes[2].plot(theta_full, delta_full[:, i], label=f'ΔV component {i+1}')
                axes[2].set_xlabel('Theta')
                axes[2].set_ylabel('Va - Vx')
                axes[2].set_title(f'ΔV Components at d = {d:.6f}')
                axes[2].legend()
                axes[2].grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'all_components_{d:.6f}.png'))
                plt.close()
    else:
        print("No symmetric triple points found within the given tolerance.")
    
    return d_values

if __name__ == "__main__":
    # Set parameters with default values
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    omega = 0.1
    R_0 = (0, 0, 0)
    epsilon = 1e-5
    
    # Ask user which search mode to use
    print("Choose search mode:")
    print("1. Quick search (d = 0.001 to 0.1, 10,000 points)")
    print("2. Medium search (d = 0.001 to 0.5, 100,000 points)")
    print("3. Comprehensive search (d = 0.001 to 1.0, 1,000,000 points)")
    print("4. Custom range search")
    
    choice = input("Enter your choice (1/2/3/4): ")
    
    if choice == "1":
        # Quick search
        d_start = 0.001
        d_end = 0.1
        num_points = 10000
    elif choice == "2":
        # Medium search
        d_start = 0.001
        d_end = 0.5
        num_points = 100000
    elif choice == "3":
        # Comprehensive search
        d_start = 0.001
        d_end = 1.0
        num_points = 1000000
    elif choice == "4":
        # Custom range search
        d_start = float(input("Enter start d value: "))
        d_end = float(input("Enter end d value: "))
        num_points = int(input("Enter number of points: "))
        epsilon = float(input("Enter tolerance (epsilon): "))
        aVx = float(input("Enter aVx value (default 1.0): ") or "1.0")
        aVa = float(input("Enter aVa value (default 5.0): ") or "5.0")
        x_shift = float(input("Enter x_shift value (default 0.01): ") or "0.01")
        c_const = float(input("Enter c_const value (default 0.01): ") or "0.01")
        omega = float(input("Enter omega value (default 0.1): ") or "0.1")
    else:
        # Default to quick search
        print("Invalid choice. Using quick search.")
        d_start = 0.001
        d_end = 0.1
        num_points = 10000
    
    print(f"Searching for symmetric triple points with d ∈ [{d_start}, {d_end}], {num_points} points, epsilon = {epsilon}")
    print(f"Parameters: aVx={aVx}, aVa={aVa}, x_shift={x_shift}, c_const={c_const}, omega={omega}")
    
    # Create d range
    d_range = np.linspace(d_start, d_end, num_points, endpoint=True)
    
    # Find symmetric triple points
    d_values = find_exact_symmetric_triple_d(
        d_range, aVx=aVx, aVa=aVa, x_shift=x_shift, c_const=c_const, omega=omega, R_0=R_0, epsilon=epsilon
    )
    
    if d_values:
        print(f"Found {len(d_values)} symmetric triple points:")
        for i, d in enumerate(d_values[:10]):  # Show only first 10 if there are many
            print(f"{i+1}. d = {d:.6f}")
        
        if len(d_values) > 10:
            print(f"... and {len(d_values) - 10} more")
    else:
        print("No symmetric triple points found within the given tolerance.")
