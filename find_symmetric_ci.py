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

def find_exact_symmetric_triple_d(
    d_vals, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-6, save_plots=False
):
    """
    Find exact symmetric triple points where Va-Vx is the same for all three components
    at three symmetric theta values (0, 2π/3, 4π/3).
    
    Parameters:
    d_vals (numpy.ndarray): Array of d values to search through
    aVx (float): Parameter of the potential Vx
    aVa (float): Parameter of the potential Va
    x_shift (float): Shift in the potential Va
    c_const (float): Constant in the potential
    omega (float): Angular frequency of the oscillator
    R_0 (numpy.ndarray): The origin vector
    epsilon (float): Tolerance for equality comparison
    save_plots (bool): Whether to save plots
    
    Returns:
    tuple: (found_d, delta_Vs) if a solution is found, else (None, None)
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'symmetric_ci_search_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the three symmetric theta values
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    # Store results for all d values
    all_delta_Vs = []
    all_d_vals = []
    
    # Search through d values
    for d in tqdm.tqdm(d_vals, desc="Searching for symmetric triple points"):
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
            
            if save_plots:
                plot_dir = os.path.join(output_dir, f'plots_triple_d_{d:.6f}')
                os.makedirs(plot_dir, exist_ok=True)
                
                # Plot Va and Vx at full resolution over theta ∈ [0, 2π]
                theta_full = np.linspace(0, 2 * np.pi, 1000)
                h_full = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_full)
                R_full = np.array(h_full.R_thetas())
                Va_full = np.array(h_full.Va_theta_vals(R_full))  # shape (1000, 3)
                Vx_full = np.array(h_full.Vx_theta_vals(R_full))  # shape (1000, 3)

                # Plot each component of Va
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, Va_full[:, i], label=fr'$V_{{a,{i+1}}}$')
                plt.xlabel(r'$\theta$')
                plt.ylabel(r'$V_a(R(\theta))$')
                plt.title(f'Va Components at d = {d:.6f}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'Va_full_{d:.6f}.png'))
                plt.close()

                # Plot each component of Vx
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, Vx_full[:, i], label=fr'$V_{{x,{i+1}}}$')
                plt.xlabel(r'$\theta$')
                plt.ylabel(r'$V_x(R(\theta))$')
                plt.title(f'Vx Components at d = {d:.6f}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'Vx_full_{d:.6f}.png'))
                plt.close()

                # ΔV = Va - Vx
                delta_V = Va_full - Vx_full  # shape (1000, 3)

                # Plot all three ΔV curves
                plt.figure(figsize=(10, 5))
                for i in range(3):
                    plt.plot(theta_full, delta_V[:, i], label=fr'$\Delta V_{{{i+1}}}$')
                plt.xlabel(r'$\theta$')
                plt.ylabel(r'$V_a(R(\theta)) - V_x(R(\theta))$')
                plt.title(f'Potential Difference Components at d = {d:.6f}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'deltaV_full_{d:.6f}.png'))
                plt.close()
                
                # Create a combined plot with all components in separate subplots
                fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                
                # Plot Va components
                for i in range(3):
                    axes[0].plot(theta_full, Va_full[:, i], label=fr'$V_{{a,{i+1}}}$')
                axes[0].set_ylabel(r'$V_a(R(\theta))$')
                axes[0].set_title(f'Va Components at d = {d:.6f}')
                axes[0].legend()
                axes[0].grid(True)
                
                # Plot Vx components
                for i in range(3):
                    axes[1].plot(theta_full, Vx_full[:, i], label=fr'$V_{{x,{i+1}}}$')
                axes[1].set_ylabel(r'$V_x(R(\theta))$')
                axes[1].set_title(f'Vx Components at d = {d:.6f}')
                axes[1].legend()
                axes[1].grid(True)
                
                # Plot deltaV components
                for i in range(3):
                    axes[2].plot(theta_full, delta_V[:, i], label=fr'$\Delta V_{{{i+1}}}$')
                axes[2].set_xlabel(r'$\theta$')
                axes[2].set_ylabel(r'$V_a - V_x$')
                axes[2].set_title(f'Potential Difference Components at d = {d:.6f}')
                axes[2].legend()
                axes[2].grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'all_potentials_vs_theta_{d:.6f}.png'))
                plt.close()
                
                # Also create individual plots for each component across all potentials
                for i in range(3):
                    plt.figure(figsize=(10, 6))
                    plt.plot(theta_full, Va_full[:, i], label=fr'$V_{{a,{i+1}}}$')
                    plt.plot(theta_full, Vx_full[:, i], label=fr'$V_{{x,{i+1}}}$')
                    plt.plot(theta_full, delta_V[:, i], label=fr'$\Delta V_{{{i+1}}}$')
                    plt.xlabel(r'$\theta$')
                    plt.ylabel('Potential')
                    plt.title(f'Component {i+1} Potentials at d = {d:.6f}')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'component_{i+1}_potentials_{d:.6f}.png'))
                    plt.close()
                
                # Plot the three points in 3D space
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(R_thetas[:, 0], R_thetas[:, 1], R_thetas[:, 2], c='r', marker='o', s=100)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Symmetric Triple Points at d = {d:.6f}')
                plt.savefig(os.path.join(plot_dir, f'triple_points_3d_{d:.6f}.png'))
                plt.close()
                
            return d, delta_Vs
    
    # If no exact match is found, plot the closest matches
    if save_plots and len(all_delta_Vs) > 0:
        # Convert to numpy arrays for easier manipulation
        all_delta_Vs = np.array(all_delta_Vs)
        all_d_vals = np.array(all_d_vals)
        
        # Calculate how close each d value is to having equal delta_Vs
        closeness_scores = []
        for i, delta_Vs in enumerate(all_delta_Vs):
            # For each component, calculate the standard deviation across the three points
            component_stds = [np.std(delta_Vs[:, j]) for j in range(3)]
            # Average the standard deviations
            avg_std = np.mean(component_stds)
            closeness_scores.append(avg_std)
        
        closeness_scores = np.array(closeness_scores)
        
        # Find the 3 closest matches
        closest_indices = np.argsort(closeness_scores)[:3]
        
        # Plot the closeness score vs d
        plt.figure(figsize=(10, 6))
        plt.plot(all_d_vals, closeness_scores)
        plt.scatter(all_d_vals[closest_indices], closeness_scores[closest_indices], color='red')
        plt.xlabel('d')
        plt.ylabel('Average Standard Deviation of ΔV Components')
        plt.title('Closeness to Symmetric Triple Point')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'closeness_scores.png'))
        plt.close()
        
        # Plot the closest matches
        for idx in closest_indices:
            d = all_d_vals[idx]
            delta_Vs = all_delta_Vs[idx]
            
            print(f"Close match at d = {d:.6f}, avg std = {closeness_scores[idx]:.6e}")
            print(f"ΔV values:\n{delta_Vs}")
            
            plot_dir = os.path.join(output_dir, f'plots_close_d_{d:.6f}')
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot the delta_Vs for this close match
            plt.figure(figsize=(8, 6))
            theta_labels = ['0', '2π/3', '4π/3']
            for i in range(3):
                plt.plot(range(3), delta_Vs[:, i], 'o-', label=f'Component {i+1}')
            plt.xticks(range(3), theta_labels)
            plt.xlabel('Theta')
            plt.ylabel('Va - Vx')
            plt.title(f'ΔV Components at d = {d:.6f}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'deltaV_components_{d:.6f}.png'))
            plt.close()
    
    print("No exact symmetric triple point found within the given tolerance.")
    return None, None

def process_d_range(d_start, d_end, num_points, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-6):
    """
    Process a range of d values in parallel
    
    Parameters:
    d_start (float): Start of d range
    d_end (float): End of d range
    num_points (int): Number of points in the range
    aVx, aVa, x_shift, c_const, omega, R_0, epsilon: Parameters for find_exact_symmetric_triple_d
    
    Returns:
    tuple: (found_d, delta_Vs) if a solution is found, else (None, None)
    """
    d_vals = np.linspace(d_start, d_end, num_points)
    return find_exact_symmetric_triple_d(
        d_vals, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, save_plots=False
    )

def parallel_search_symmetric_triple_d(
    d_start, d_end, num_points, aVx, aVa, x_shift, c_const, omega, R_0, 
    epsilon=1e-6, num_processes=None, save_plots=True
):
    """
    Search for symmetric triple points in parallel
    
    Parameters:
    d_start (float): Start of d range
    d_end (float): End of d range
    num_points (int): Number of points in the range
    aVx, aVa, x_shift, c_const, omega, R_0, epsilon: Parameters for find_exact_symmetric_triple_d
    num_processes (int): Number of processes to use, defaults to CPU count - 1
    save_plots (bool): Whether to save plots
    
    Returns:
    tuple: (found_d, delta_Vs) if a solution is found, else (None, None)
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} processes for parallel search")
    
    # Split the d range into chunks for parallel processing
    chunk_size = num_points // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    d_chunks = []
    for i in range(0, num_points, chunk_size):
        chunk_start = d_start + (d_end - d_start) * (i / num_points)
        chunk_end = d_start + (d_end - d_start) * (min(i + chunk_size, num_points) / num_points)
        chunk_points = min(chunk_size, num_points - i)
        d_chunks.append((chunk_start, chunk_end, chunk_points))
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for chunk_start, chunk_end, chunk_points in d_chunks:
            results.append(pool.apply_async(
                process_d_range, 
                (chunk_start, chunk_end, chunk_points, aVx, aVa, x_shift, c_const, omega, R_0, epsilon)
            ))
        
        # Wait for all processes to complete
        for i, result in enumerate(results):
            found_d, delta_Vs = result.get()
            if found_d is not None:
                # Found a solution
                if save_plots:
                    # Generate plots for the found solution
                    d_vals = np.array([found_d])
                    find_exact_symmetric_triple_d(
                        d_vals, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, save_plots=True
                    )
                return found_d, delta_Vs
    
    # If no solution found, do a final search with plots
    if save_plots:
        d_vals = np.linspace(d_start, d_end, min(num_points, 100))  # Use fewer points for the final search
        return find_exact_symmetric_triple_d(
            d_vals, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, save_plots=True
        )
    
    return None, None

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
    print("1. Wide range search (d = 0.01 to 1.0)")
    print("2. Narrow range search around d = 0.01 (d = 0.005 to 0.015)")
    print("3. Custom range search")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        # Wide range search
        d_start = 0.01
        d_end = 1.0
        num_points = 1000
        epsilon = 1e-4
    elif choice == "2":
        # Narrow range search around d = 0.01
        d_start = 0.005
        d_end = 0.015
        num_points = 1000
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
        d_start = 0.005
        d_end = 0.015
        num_points = 1000
        epsilon = 1e-5
    
    print(f"Searching for symmetric triple points with d ∈ [{d_start}, {d_end}], {num_points} points, epsilon = {epsilon}")
    
    # Use parallel search
    found_d, delta_Vs = parallel_search_symmetric_triple_d(
        d_start, d_end, num_points, aVx, aVa, x_shift, c_const, omega, R_0, 
        epsilon=epsilon, save_plots=True
    )
    
    if found_d is not None:
        print(f"Found symmetric triple point at d = {found_d:.6f}")
        print(f"ΔV values:\n{delta_Vs}")
    else:
        print("No symmetric triple point found within the given tolerance.")
