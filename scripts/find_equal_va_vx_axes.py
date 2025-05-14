#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import hbar
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

def analyze_specific_d_value(d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5, plot_dir=None, check_method="exact"):
    """
    Analyze a specific d value and check if Va or Vx values are equal or close on both x and y axes
    
    Parameters:
    d (float): The d value to analyze
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    plot_dir: Directory to save plots, if None no plots are generated
    check_method: Method to check for equality - "exact" (isclose), "relative" (relative difference), or "min_diff" (minimum difference)
    
    Returns:
    tuple: (has_equal_components, results_dict)
    """
    # Define the three symmetric theta values
    theta_triple = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    # Create Hamiltonian and calculate potentials
    h = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_triple)
    R_thetas = np.array(h.R_thetas())
    Va_vals = np.array(h.Va_theta_vals(R_thetas))  # shape (3, 3)
    Vx_vals = np.array(h.Vx_theta_vals(R_thetas))  # shape (3, 3)

    # Calculate differences between x and y components
    Va_x_y_diff = np.abs(Va_vals[:, 0] - Va_vals[:, 1])
    Vx_x_y_diff = np.abs(Vx_vals[:, 0] - Vx_vals[:, 1])
    
    # Calculate relative differences
    Va_x_y_rel_diff = Va_x_y_diff / (np.abs(Va_vals[:, 0]) + np.abs(Va_vals[:, 1]) + 1e-10)
    Vx_x_y_rel_diff = Vx_x_y_diff / (np.abs(Vx_vals[:, 0]) + np.abs(Vx_vals[:, 1]) + 1e-10)
    
    # Check for equality between x and y components (indices 0 and 1)
    if check_method == "exact":
        Va_x_y_equal = [
            np.isclose(Va_vals[i, 0], Va_vals[i, 1], atol=epsilon)
            for i in range(3)
        ]
        
        Vx_x_y_equal = [
            np.isclose(Vx_vals[i, 0], Vx_vals[i, 1], atol=epsilon)
            for i in range(3)
        ]
        
        # Check if all points have equal x and y components for either Va or Vx
        all_Va_x_y_equal = all(Va_x_y_equal)
        all_Vx_x_y_equal = all(Vx_x_y_equal)
        
    elif check_method == "relative":
        # Check if relative differences are below threshold
        all_Va_x_y_equal = np.all(Va_x_y_rel_diff < epsilon)
        all_Vx_x_y_equal = np.all(Vx_x_y_rel_diff < epsilon)
        
    elif check_method == "min_diff":
        # Check if the maximum difference is below threshold
        all_Va_x_y_equal = np.max(Va_x_y_diff) < epsilon
        all_Vx_x_y_equal = np.max(Vx_x_y_diff) < epsilon
    
    has_equal_components = all_Va_x_y_equal or all_Vx_x_y_equal
    
    # Calculate mean differences for reporting
    mean_Va_x_y_diff = np.mean(Va_x_y_diff)
    mean_Vx_x_y_diff = np.mean(Vx_x_y_diff)
    max_Va_x_y_diff = np.max(Va_x_y_diff)
    max_Vx_x_y_diff = np.max(Vx_x_y_diff)
    
    results_dict = {
        "Va_x_y_equal": all_Va_x_y_equal,
        "Vx_x_y_equal": all_Vx_x_y_equal,
        "Va_vals": Va_vals,
        "Vx_vals": Vx_vals,
        "R_thetas": R_thetas,
        "Va_x_y_diff": Va_x_y_diff,
        "Vx_x_y_diff": Vx_x_y_diff,
        "mean_Va_x_y_diff": mean_Va_x_y_diff,
        "mean_Vx_x_y_diff": mean_Vx_x_y_diff,
        "max_Va_x_y_diff": max_Va_x_y_diff,
        "max_Vx_x_y_diff": max_Vx_x_y_diff
    }
    
    if has_equal_components and plot_dir is not None:
        # Create plots directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        
        if all_Va_x_y_equal:
            print(f"🎯 Equal Va x-y components found for d = {d:.6f}")
            print(f"Va values at θ = 0, 2π/3, 4π/3:\n{Va_vals}")
        
        if all_Vx_x_y_equal:
            print(f"🎯 Equal Vx x-y components found for d = {d:.6f}")
            print(f"Vx values at θ = 0, 2π/3, 4π/3:\n{Vx_vals}")
        
        # Plot Va and Vx at full resolution over theta ∈ [0, 2π]
        theta_full = np.linspace(0, 2 * np.pi, 1000)
        h_full = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_full)
        R_full = np.array(h_full.R_thetas())
        Va_full = np.array(h_full.Va_theta_vals(R_full))  # shape (1000, 3)
        Vx_full = np.array(h_full.Vx_theta_vals(R_full))  # shape (1000, 3)
        
        # Plot Va components
        plt.figure(figsize=(10, 5))
        plt.plot(theta_full, Va_full[:, 0], 'r-', label='Va x component')
        plt.plot(theta_full, Va_full[:, 1], 'g-', label='Va y component')
        plt.plot(theta_full, Va_full[:, 2], 'b-', label='Va z component')
        for θ in theta_triple:
            plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Va')
        plt.title(f'Va Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Va_components_{d:.6f}.png'))
        plt.close()
        
        # Plot Vx components
        plt.figure(figsize=(10, 5))
        plt.plot(theta_full, Vx_full[:, 0], 'r-', label='Vx x component')
        plt.plot(theta_full, Vx_full[:, 1], 'g-', label='Vx y component')
        plt.plot(theta_full, Vx_full[:, 2], 'b-', label='Vx z component')
        for θ in theta_triple:
            plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Vx')
        plt.title(f'Vx Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Vx_components_{d:.6f}.png'))
        plt.close()
        
        # Plot the difference between x and y components for Va
        plt.figure(figsize=(10, 5))
        plt.plot(theta_full, Va_full[:, 0] - Va_full[:, 1], 'r-', label='Va(x) - Va(y)')
        for θ in theta_triple:
            plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Va(x) - Va(y)')
        plt.title(f'Va x-y Difference vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Va_x_y_diff_{d:.6f}.png'))
        plt.close()
        
        # Plot the difference between x and y components for Vx
        plt.figure(figsize=(10, 5))
        plt.plot(theta_full, Vx_full[:, 0] - Vx_full[:, 1], 'b-', label='Vx(x) - Vx(y)')
        for θ in theta_triple:
            plt.axvline(θ, color='k', linestyle='--', alpha=0.3)
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Vx(x) - Vx(y)')
        plt.title(f'Vx x-y Difference vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Vx_x_y_diff_{d:.6f}.png'))
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
    
    return has_equal_components, results_dict

def adaptive_d_search(d_start, d_end, small_step, large_step, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-5, check_method="exact", track_min_diff=False):
    """
    Perform an adaptive search for d values where Va or Vx have equal x and y components,
    using small_step normally but skipping ahead by large_step when a point is found
    
    Parameters:
    d_start, d_end: Range of d values to search
    small_step: Normal step size (e.g. 0.001)
    large_step: Skip step size after finding a point (e.g. 0.01)
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    epsilon: Tolerance for equality comparison
    check_method: Method to check for equality - "exact", "relative", or "min_diff"
    track_min_diff: If True, track the d values with minimum differences even if not equal
    
    Returns:
    list: List of d values where the condition is satisfied
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'equal_va_vx_axes_search_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Store the d values where the equation is satisfied
    equal_component_points = []
    
    # If tracking minimum differences, keep track of the smallest differences
    if track_min_diff:
        min_Va_diff = float('inf')
        min_Vx_diff = float('inf')
        min_Va_d = None
        min_Vx_d = None
    
    # Initialize d
    d = d_start
    
    # Create a progress bar
    # Estimate the number of steps (assuming no points found)
    estimated_steps = int((d_end - d_start) / small_step)
    pbar = tqdm.tqdm(total=estimated_steps, desc="Adaptive d search")
    
    # Keep track of the number of steps taken
    steps_taken = 0
    
    # Open a file to write results in real-time
    results_file = os.path.join(parent_dir, 'equal_component_points.txt')
    with open(results_file, 'w') as f:
        f.write(f"d values where Va(x) = Va(y) or Vx(x) = Vx(y) for all three points (epsilon={epsilon}, method={check_method}):\n")
    
    # Open a file to track differences
    if track_min_diff:
        diff_file = os.path.join(parent_dir, 'component_differences.txt')
        with open(diff_file, 'w') as f:
            f.write("d\tMax Va(x)-Va(y)\tMax Vx(x)-Vx(y)\n")
    
    # Search through d values
    while d <= d_end:
        # Create a directory for plots for this d value if it satisfies the condition
        plot_dir = None  # Only create if needed
        
        # Check if this d value satisfies the condition
        has_equal_components, results = analyze_specific_d_value(
            d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, plot_dir, check_method
        )
        
        # Update progress bar
        steps_taken += 1
        pbar.update(1)
        
        # Track minimum differences if requested
        if track_min_diff:
            max_Va_diff = results["max_Va_x_y_diff"]
            max_Vx_diff = results["max_Vx_x_y_diff"]
            
            # Write to differences file every 100 steps to avoid excessive I/O
            if steps_taken % 100 == 0:
                with open(diff_file, 'a') as f:
                    f.write(f"{d:.6f}\t{max_Va_diff:.10f}\t{max_Vx_diff:.10f}\n")
            
            # Update minimum Va difference if found
            if max_Va_diff < min_Va_diff:
                min_Va_diff = max_Va_diff
                min_Va_d = d
                print(f"New minimum Va x-y difference: {min_Va_diff:.10f} at d = {min_Va_d:.6f}")
            
            # Update minimum Vx difference if found
            if max_Vx_diff < min_Vx_diff:
                min_Vx_diff = max_Vx_diff
                min_Vx_d = d
                print(f"New minimum Vx x-y difference: {min_Vx_diff:.10f} at d = {min_Vx_d:.6f}")
        
        if has_equal_components:
            # Create plot directory now that we found a match
            plot_dir = os.path.join(parent_dir, f'plots_d_{d:.6f}')
            
            # Re-analyze with plot directory to generate plots
            has_equal_components, results = analyze_specific_d_value(
                d, aVx, aVa, x_shift, c_const, omega, R_0, epsilon, plot_dir, check_method
            )
            
            # Add this d value to our list
            equal_component_points.append((d, results))
            
            # Append to the results file
            with open(results_file, 'a') as f:
                if results["Va_x_y_equal"]:
                    f.write(f"{d:.10f}: Va(x) = Va(y), max diff = {results['max_Va_x_y_diff']:.10f}\n")
                if results["Vx_x_y_equal"]:
                    f.write(f"{d:.10f}: Vx(x) = Vx(y), max diff = {results['max_Vx_x_y_diff']:.10f}\n")
            
            # Skip ahead by large_step
            d += large_step
        else:
            # Move to the next d value with small_step
            d += small_step
        
        # Update the progress bar description
        pbar.set_description(f"Adaptive d search (found {len(equal_component_points)} points)")
    
    # Close the progress bar
    pbar.close()
    
    print(f"Search complete. Found {len(equal_component_points)} points where Va(x) = Va(y) or Vx(x) = Vx(y).")
    
    return equal_component_points

if __name__ == "__main__":
    # Set parameters with default values
    aVx = 1.0
    aVa = 5.0
    x_shift = 0.01
    c_const = 0.01
    omega = 0.1
    
    # Default search range and step sizes
    d_start = 0.001
    d_end = 1000.0
    small_step = 0.0001
    large_step = 0.01
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Search for d values where Va(x) = Va(y) or Vx(x) = Vx(y).')
    parser.add_argument('--aVx', type=float, default=aVx, help=f'Parameter aVx (default: {aVx})')
    parser.add_argument('--aVa', type=float, default=aVa, help=f'Parameter aVa (default: {aVa})')
    parser.add_argument('--x_shift', type=float, default=x_shift, help=f'Parameter x_shift (default: {x_shift})')
    parser.add_argument('--c_const', type=float, default=c_const, help=f'Parameter c_const (default: {c_const})')
    parser.add_argument('--omega', type=float, default=omega, help=f'Parameter omega (default: {omega})')
    parser.add_argument('--d_start', type=float, default=d_start, help=f'Start of d range (default: {d_start})')
    parser.add_argument('--d_end', type=float, default=d_end, help=f'End of d range (default: {d_end})')
    parser.add_argument('--small_step', type=float, default=small_step, help=f'Small step size (default: {small_step})')
    parser.add_argument('--large_step', type=float, default=large_step, help=f'Large step size after finding a point (default: {large_step})')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Tolerance for equality comparison (default: 1e-5)')
    parser.add_argument('--check_method', type=str, default='exact', choices=['exact', 'relative', 'min_diff'], 
                        help='Method to check for equality (default: exact)')
    parser.add_argument('--track_min_diff', action='store_true', help='Track d values with minimum differences')
    parser.add_argument('--focus_d', type=float, help='Focus search around this d value with higher precision')
    parser.add_argument('--focus_range', type=float, default=0.01, help='Range around focus_d to search (default: 0.01)')
    parser.add_argument('--focus_step', type=float, default=0.00001, help='Step size for focus search (default: 0.00001)')
    args = parser.parse_args()
    
    # Update parameters from command line arguments
    aVx = args.aVx
    aVa = args.aVa
    x_shift = args.x_shift
    c_const = args.c_const
    omega = args.omega
    d_start = args.d_start
    d_end = args.d_end
    small_step = args.small_step
    large_step = args.large_step
    epsilon = args.epsilon
    check_method = args.check_method
    track_min_diff = args.track_min_diff
    
    # Define the origin vector R_0
    R_0 = create_perfect_orthogonal_vectors(1.0)
    
    # If focus_d is provided, perform a focused search around that value
    if args.focus_d is not None:
        focus_d = args.focus_d
        focus_range = args.focus_range
        focus_step = args.focus_step
        
        d_start = max(0.001, focus_d - focus_range)
        d_end = focus_d + focus_range
        small_step = focus_step
        large_step = focus_step * 10
        
        print(f"Performing focused search around d={focus_d} with range ±{focus_range}")
    
    print(f"Starting adaptive search from d={d_start} to d={d_end}")
    print(f"Using small step={small_step}, large step={large_step} after finding a point")
    print(f"Parameters: aVx={aVx}, aVa={aVa}, x_shift={x_shift}, c_const={c_const}, omega={omega}")
    print(f"Check method: {check_method}, epsilon: {epsilon}, track minimum differences: {track_min_diff}")
    
    # Perform the search
    equal_component_points = adaptive_d_search(
        d_start, d_end, small_step, large_step, 
        aVx, aVa, x_shift, c_const, omega, R_0, epsilon,
        check_method=check_method, track_min_diff=track_min_diff
    )
    
    # If we found points, analyze the first one in detail
    if equal_component_points:
        d, results = equal_component_points[0]
        print(f"\nDetailed analysis of first found point at d = {d:.10f}:")
        if results["Va_x_y_equal"]:
            print(f"Va(x) = Va(y) with max difference: {results['max_Va_x_y_diff']:.10f}")
            print(f"Va values at the three points:\n{results['Va_vals']}")
        if results["Vx_x_y_equal"]:
            print(f"Vx(x) = Vx(y) with max difference: {results['max_Vx_x_y_diff']:.10f}")
            print(f"Vx values at the three points:\n{results['Vx_vals']}")
    elif track_min_diff:
        print("\nNo exact matches found, but tracked minimum differences.")
        print("Check the component_differences.txt file for details.")
