#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import hbar
import datetime
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))

try:
    from vector_utils import create_perfect_orthogonal_vectors
    print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")
except ImportError:
    print("Failed to import from arrowhead/generalized package.")
    sys.exit(1)

# Import the Hamiltonian class from new_bph.py
from new_bph import Hamiltonian

def analyze_specific_d_value(d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5, plot_dir=None):
    """
    Analyze a specific d value and check if it satisfies the triple point condition
    
    Parameters:
    d (float): The d value to analyze
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    plot_dir: Directory to save plots, if None no plots are generated
    
    Returns:
    tuple: (is_triple_point, delta_Vs)
    """
    # Define the three symmetric theta values
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    # Create Hamiltonian and calculate potentials
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

    is_triple_point = all(componentwise_equal)
    
    if is_triple_point and plot_dir is not None:
        # Create plots directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        
        print(f"🎯 Exact triple-point degeneracy found for d = {d:.6f}")
        print(f"ΔV values at θ = 0, 2π/3, 4π/3:\n{delta_Vs}")
        
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
        
        # Plot Va components
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, Va_full[:, i], label=f'Va component {i+1}')
        plt.xlabel('Theta')
        plt.ylabel('Va')
        plt.title(f'Va Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Va_components_{d:.6f}.png'))
        plt.close()
        
        # Plot Vx components
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.plot(theta_full, Vx_full[:, i], label=f'Vx component {i+1}')
        plt.xlabel('Theta')
        plt.ylabel('Vx')
        plt.title(f'Vx Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Vx_components_{d:.6f}.png'))
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
        plt.savefig(os.path.join(plot_dir, f'R_vectors_3d_{d:.6f}.png'))
        plt.close()
    
    return is_triple_point, delta_Vs

def adaptive_d_search(d_start, d_end, small_step, large_step, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5):
    """
    Perform an adaptive search for triple points, using small_step normally
    but skipping ahead by large_step when a triple point is found
    
    Parameters:
    d_start, d_end: Range of d values to search
    small_step: Normal step size (e.g. 0.001)
    large_step: Skip step size after finding a point (e.g. 0.01)
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    
    Returns:
    list: List of d values where the triple point condition is satisfied
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'adaptive_d_search_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Store the d values where the equation is satisfied
    triple_points = []
    
    # Initialize d
    d = d_start
    
    # Create a progress bar
    # Estimate the number of steps (assuming no triple points found)
    estimated_steps = int((d_end - d_start) / small_step)
    pbar = tqdm.tqdm(total=estimated_steps, desc="Adaptive d search")
    
    # Save search parameters to a file
    with open(os.path.join(parent_dir, 'search_params.txt'), 'w') as f:
        f.write(f"Search parameters:\n")
        f.write(f"d_start = {d_start}\n")
        f.write(f"d_end = {d_end}\n")
        f.write(f"small_step = {small_step}\n")
        f.write(f"large_step = {large_step}\n")
        f.write(f"aVx = {aVx}\n")
        f.write(f"aVa = {aVa}\n")
        f.write(f"x_shift = {x_shift}\n")
        f.write(f"c_const = {c_const}\n")
        f.write(f"omega = {omega}\n")
        f.write(f"epsilon = {epsilon}\n")
    
    # Search through d values
    while d <= d_end:
        # Create plot directory for this d value
        plot_dir = os.path.join(parent_dir, f'd_{d:.6f}')
        
        # Check if this d value satisfies the triple point condition
        is_triple_point, delta_Vs = analyze_specific_d_value(
            d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, plot_dir
        )
        
        # Update progress bar
        pbar.update(1)
        
        if is_triple_point:
            triple_points.append((d, delta_Vs))
            
            # Skip ahead by large_step
            d += large_step
            
            # Update the progress bar to reflect the skip
            pbar.update(int(large_step / small_step) - 1)
        else:
            # Move to the next d value with small_step
            d += small_step
    
    pbar.close()
    
    # Save the found d values to a file
    with open(os.path.join(parent_dir, 'triple_points.txt'), 'w') as f:
        f.write("d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2):\n")
        for d, delta_Vs in triple_points:
            f.write(f"{d:.10f}: {np.mean(delta_Vs[:, 0]):.10f}\n")
    
    return triple_points

if __name__ == "__main__":
    # Set parameters with default values
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    omega = 0.1
    R_0 = (0, 0, 0)
    epsilon = 1e-5
    
    # Set search parameters
    d_start = 0.001
    d_end = 1000.0
    small_step = 0.0001  # Normal step size
    large_step = 0.01   # Skip step size after finding a point
    
    print(f"Starting adaptive search from d={d_start} to d={d_end}")
    print(f"Using small step={small_step}, large step={large_step} after finding a point")
    print(f"Parameters: aVx={aVx}, aVa={aVa}, x_shift={x_shift}, c_const={c_const}, omega={omega}")
    
    # Perform adaptive search
    triple_points = adaptive_d_search(
        d_start, d_end, small_step, large_step, 
        aVx, aVa, x_shift, c_const, omega, R_0, epsilon
    )
    
    # Print results
    if triple_points:
        print(f"\nFound {len(triple_points)} triple points:")
        for i, (d, _) in enumerate(triple_points):
            print(f"{i+1}. d = {d:.6f}")
    else:
        print("\nNo triple points found.")
