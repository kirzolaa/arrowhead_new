#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generalized'))

try:
    from vector_utils import create_perfect_orthogonal_vectors
    print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")
except ImportError:
    print("Failed to import from arrowhead/generalized package.")
    sys.exit(1)

# Import the Hamiltonian class from new_bph.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from new_bph import Hamiltonian

def analyze_vx_yz_equality(d, aVx, aVa, x_shift, c_const, omega, R_0, num_theta=500):
    """
    Analyze the equality between y and z components of Vx for a specific d value
    
    Parameters:
    d (float): The d value to analyze
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    num_theta: Number of theta values to check
    
    Returns:
    dict: Results dictionary with analysis information
    """
    # Generate theta values across the full range [0, 2π]
    theta_vals = np.linspace(0, 2 * np.pi, num_theta)
    
    # Create Hamiltonian and calculate potentials
    h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    R_thetas = np.array(h.R_thetas())
    Va_vals = np.array(h.Va_theta_vals(R_thetas))  # shape (num_theta, 3)
    Vx_vals = np.array(h.Vx_theta_vals(R_thetas))  # shape (num_theta, 3)
    
    # Calculate differences between y and z components of Vx
    Vx_y_z_diff = Vx_vals[:, 1] - Vx_vals[:, 2]
    max_diff = np.max(np.abs(Vx_y_z_diff))
    mean_diff = np.mean(np.abs(Vx_y_z_diff))
    
    results = {
        "d": d,
        "theta_vals": theta_vals,
        "Vx_vals": Vx_vals,
        "Vx_y_z_diff": Vx_y_z_diff,
        "max_diff": max_diff,
        "mean_diff": mean_diff
    }
    
    return results

def plot_vx_yz_comparison(results, output_dir):
    """
    Create plots comparing the y and z components of Vx
    
    Parameters:
    results: Results dictionary from analyze_vx_yz_equality
    output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    d = results["d"]
    theta_vals = results["theta_vals"]
    Vx_vals = results["Vx_vals"]
    Vx_y_z_diff = results["Vx_y_z_diff"]
    max_diff = results["max_diff"]
    mean_diff = results["mean_diff"]
    
    # Plot Vx components
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals, Vx_vals[:, 0], 'r-', label='Vx x component')
    plt.plot(theta_vals, Vx_vals[:, 1], 'g-', label='Vx y component')
    plt.plot(theta_vals, Vx_vals[:, 2], 'b-', label='Vx z component')
    plt.xlabel('Theta')
    plt.ylabel('Vx')
    plt.title(f'Vx Components vs Theta (d = {d:.6f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'Vx_components_{d:.6f}.png'))
    plt.close()
    
    # Plot y and z components of Vx with zoom
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals, Vx_vals[:, 1], 'g-', label='Vx y component')
    plt.plot(theta_vals, Vx_vals[:, 2], 'b-', label='Vx z component')
    plt.xlabel('Theta')
    plt.ylabel('Vx')
    plt.title(f'Vx y and z Components vs Theta (d = {d:.6f})\nMax diff = {max_diff:.10f}, Mean diff = {mean_diff:.10f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'Vx_y_z_components_{d:.6f}.png'))
    plt.close()
    
    # Plot difference between y and z components
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals, Vx_y_z_diff, 'k-')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Theta')
    plt.ylabel('Vx(y) - Vx(z)')
    plt.title(f'Difference between Vx y and z Components (d = {d:.6f})\nMax diff = {max_diff:.10f}, Mean diff = {mean_diff:.10f}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'Vx_y_z_diff_{d:.6f}.png'))
    plt.close()
    
    # Plot difference between y and z components with zoom
    plt.figure(figsize=(12, 6))
    plt.plot(theta_vals, Vx_y_z_diff, 'k-')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Theta')
    plt.ylabel('Vx(y) - Vx(z)')
    plt.title(f'Difference between Vx y and z Components (d = {d:.6f})\nMax diff = {max_diff:.10f}, Mean diff = {mean_diff:.10f}')
    plt.ylim(-max_diff*1.2, max_diff*1.2)  # Zoom in on the y-axis
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'Vx_y_z_diff_zoom_{d:.6f}.png'))
    plt.close()

def analyze_d_range(d_start, d_end, d_step, aVx, aVa, x_shift, c_const, omega, R_0, num_theta=500):
    """
    Analyze a range of d values to find where Vx y and z components are most equal
    
    Parameters:
    d_start, d_end: Range of d values to analyze
    d_step: Step size for d values
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    num_theta: Number of theta values to check
    
    Returns:
    dict: Results with best d value and analysis
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'vx_yz_equality_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Generate d values to analyze
    d_values = np.arange(d_start, d_end + d_step/2, d_step)
    
    # Track the d value with minimum difference
    min_diff = float('inf')
    best_d = None
    best_results = None
    
    # Create a progress bar
    pbar = tqdm.tqdm(total=len(d_values), desc="Analyzing d values")
    
    # Open a file to write results
    results_file = os.path.join(parent_dir, 'vx_yz_equality_results.txt')
    with open(results_file, 'w') as f:
        f.write("d\tMax Vx(y)-Vx(z)\tMean Vx(y)-Vx(z)\n")
    
    # Analyze each d value
    for d in d_values:
        # Analyze this d value
        results = analyze_vx_yz_equality(d, aVx, aVa, x_shift, c_const, omega, R_0, num_theta)
        
        # Update progress bar
        pbar.update(1)
        
        # Write results to file
        with open(results_file, 'a') as f:
            f.write(f"{d:.10f}\t{results['max_diff']:.10f}\t{results['mean_diff']:.10f}\n")
        
        # Check if this is the best d value so far
        if results['max_diff'] < min_diff:
            min_diff = results['max_diff']
            best_d = d
            best_results = results
            print(f"New minimum Vx y-z difference: {min_diff:.10f} at d = {best_d:.6f}")
    
    # Close the progress bar
    pbar.close()
    
    # Create plots for the best d value
    if best_d is not None:
        best_plot_dir = os.path.join(parent_dir, f'best_d_{best_d:.6f}')
        plot_vx_yz_comparison(best_results, best_plot_dir)
        
        # Also create plots for a few d values around the best one
        for offset in [-2*d_step, -d_step, d_step, 2*d_step]:
            nearby_d = best_d + offset
            if nearby_d >= d_start and nearby_d <= d_end:
                nearby_results = analyze_vx_yz_equality(nearby_d, aVx, aVa, x_shift, c_const, omega, R_0, num_theta)
                nearby_plot_dir = os.path.join(parent_dir, f'nearby_d_{nearby_d:.6f}')
                plot_vx_yz_comparison(nearby_results, nearby_plot_dir)
    
    # Create a plot showing how the maximum difference varies with d
    plt.figure(figsize=(12, 6))
    d_list = []
    max_diff_list = []
    with open(results_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            d_list.append(float(parts[0]))
            max_diff_list.append(float(parts[1]))
    
    plt.plot(d_list, max_diff_list, 'b-')
    plt.axvline(best_d, color='r', linestyle='--', label=f'Best d = {best_d:.6f}')
    plt.xlabel('d value')
    plt.ylabel('Max |Vx(y) - Vx(z)|')
    plt.title('Maximum Difference between Vx y and z Components vs d')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(parent_dir, 'max_diff_vs_d.png'))
    plt.close()
    
    print(f"Analysis complete. Best d value: {best_d:.10f} with max difference: {min_diff:.10f}")
    print(f"Results saved to {parent_dir}")
    
    return {"best_d": best_d, "min_diff": min_diff, "results_file": results_file}

if __name__ == "__main__":
    # Set parameters with default values
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    omega = 0.1
    
    # Default search range and step sizes
    d_start = 0.05
    d_end = 0.07
    d_step = 0.0001
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze equality between Vx y and z components.')
    parser.add_argument('--aVx', type=float, default=aVx, help=f'Parameter aVx (default: {aVx})')
    parser.add_argument('--aVa', type=float, default=aVa, help=f'Parameter aVa (default: {aVa})')
    parser.add_argument('--x_shift', type=float, default=x_shift, help=f'Parameter x_shift (default: {x_shift})')
    parser.add_argument('--c_const', type=float, default=c_const, help=f'Parameter c_const (default: {c_const})')
    parser.add_argument('--omega', type=float, default=omega, help=f'Parameter omega (default: {omega})')
    parser.add_argument('--d_start', type=float, default=d_start, help=f'Start of d range (default: {d_start})')
    parser.add_argument('--d_end', type=float, default=d_end, help=f'End of d range (default: {d_end})')
    parser.add_argument('--d_step', type=float, default=d_step, help=f'Step size for d values (default: {d_step})')
    parser.add_argument('--num_theta', type=int, default=500, help='Number of theta values to check (default: 500)')
    parser.add_argument('--analyze_d', type=float, help='Analyze a single d value in detail')
    args = parser.parse_args()
    
    # Update parameters from command line arguments
    aVx = args.aVx
    aVa = args.aVa
    x_shift = args.x_shift
    c_const = args.c_const
    omega = args.omega
    d_start = args.d_start
    d_end = args.d_end
    d_step = args.d_step
    num_theta = args.num_theta
    
    # Define the origin vector R_0
    R_0 = create_perfect_orthogonal_vectors(1.0)
    
    # If analyze_d is provided, just analyze that single d value
    if args.analyze_d is not None:
        d = args.analyze_d
        print(f"Analyzing single d value: {d}")
        
        # Create timestamp for output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f'vx_yz_analysis_d_{d:.6f}_{timestamp}'
        
        # Analyze the d value
        results = analyze_vx_yz_equality(d, aVx, aVa, x_shift, c_const, omega, R_0, num_theta)
        
        # Create plots
        plot_vx_yz_comparison(results, output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"Max difference between Vx y and z components: {results['max_diff']:.10f}")
        print(f"Mean difference between Vx y and z components: {results['mean_diff']:.10f}")
    else:
        # Analyze a range of d values
        print(f"Analyzing d values from {d_start} to {d_end} with step {d_step}")
        print(f"Parameters: aVx={aVx}, aVa={aVa}, x_shift={x_shift}, c_const={c_const}, omega={omega}")
        print(f"Using {num_theta} theta points")
        
        # Perform the analysis
        analyze_d_range(d_start, d_end, d_step, aVx, aVa, x_shift, c_const, omega, R_0, num_theta)
