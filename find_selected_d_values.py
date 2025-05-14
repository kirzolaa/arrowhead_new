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
    
    return is_triple_point, delta_Vs

def explore_selected_d_values(d_values, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5):
    """
    Explore selected d values to find triple points
    
    Parameters:
    d_values (list): List of d values to explore
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    
    Returns:
    list: List of d values where the triple point condition is satisfied
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'selected_d_values_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Store the d values where the equation is satisfied
    triple_points = []
    
    # Explore each d value
    for d in tqdm.tqdm(d_values, desc="Exploring selected d values"):
        plot_dir = os.path.join(parent_dir, f'd_{d:.6f}')
        is_triple_point, delta_Vs = analyze_specific_d_value(
            d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, plot_dir
        )
        
        if is_triple_point:
            triple_points.append((d, delta_Vs))
    
    # Save the found d values to a file
    with open(os.path.join(parent_dir, 'triple_points.txt'), 'w') as f:
        f.write("d values where Va-Vx(R0) = Va-Vx(R1) = Va-Vx(R2):\n")
        for d, delta_Vs in triple_points:
            f.write(f"{d:.10f}: {np.mean(delta_Vs[:, 0]):.10f}\n")
    
    return triple_points

def search_around_d_value(center_d, width, num_points, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5):
    """
    Search for triple points around a specific d value
    
    Parameters:
    center_d (float): The center d value to search around
    width (float): The width of the search range
    num_points (int): Number of points to search
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    
    Returns:
    list: List of d values where the triple point condition is satisfied
    """
    # Create d range
    d_range = np.linspace(center_d - width/2, center_d + width/2, num_points)
    
    # Explore selected d values
    return explore_selected_d_values(d_range, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)

def main():
    # Set parameters with default values
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    omega = 0.1
    R_0 = (0, 0, 0)
    epsilon = 1e-5
    
    print("Select mode:")
    print("1. Analyze specific d values")
    print("2. Search around a known d value")
    print("3. Analyze three interesting d values")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        # Analyze specific d values
        print("Enter d values separated by spaces (e.g. 0.061235 0.065 0.07):")
        d_values_input = input("> ")
        d_values = [float(d) for d in d_values_input.split()]
        
        triple_points = explore_selected_d_values(d_values, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)
        
    elif choice == "2":
        # Search around a known d value
        center_d = float(input("Enter center d value (e.g. 0.061235): "))
        width = float(input("Enter search width (e.g. 0.001): "))
        num_points = int(input("Enter number of points to search (e.g. 100): "))
        
        triple_points = search_around_d_value(center_d, width, num_points, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)
        
    elif choice == "3":
        # Analyze three interesting d values with large steps
        # Starting from the known value 0.061235, explore with large steps
        d_values = [0.001, 0.01, 0.061235, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        triple_points = explore_selected_d_values(d_values, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Print results
    if triple_points:
        print(f"Found {len(triple_points)} triple points:")
        for i, (d, _) in enumerate(triple_points):
            print(f"{i+1}. d = {d:.6f}")
    else:
        print("No triple points found.")

if __name__ == "__main__":
    main()
