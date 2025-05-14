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

def analyze_d_value_full_theta(d, aVx, aVa, x_shift, c_const, omega, R_0, num_theta=100, epsilon=1e-5, plot_dir=None):
    """
    Analyze a specific d value and check if Va or Vx components are equal across theta values
    
    Parameters:
    d (float): The d value to analyze
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    num_theta: Number of theta values to check
    epsilon: Tolerance for equality comparison
    plot_dir: Directory to save plots, if None no plots are generated
    
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
    
    # Check for equality between x and y components across all theta values
    Va_x_equals_y = np.allclose(Va_vals[:, 0], Va_vals[:, 1], atol=epsilon)
    Vx_x_equals_y = np.allclose(Vx_vals[:, 0], Vx_vals[:, 1], atol=epsilon)
    
    # Check for equality between x and z components across all theta values
    Va_x_equals_z = np.allclose(Va_vals[:, 0], Va_vals[:, 2], atol=epsilon)
    Vx_x_equals_z = np.allclose(Vx_vals[:, 0], Vx_vals[:, 2], atol=epsilon)
    
    # Check for equality between y and z components across all theta values
    Va_y_equals_z = np.allclose(Va_vals[:, 1], Va_vals[:, 2], atol=epsilon)
    Vx_y_equals_z = np.allclose(Vx_vals[:, 1], Vx_vals[:, 2], atol=epsilon)
    
    # Calculate maximum differences between components
    Va_x_y_max_diff = np.max(np.abs(Va_vals[:, 0] - Va_vals[:, 1]))
    Va_x_z_max_diff = np.max(np.abs(Va_vals[:, 0] - Va_vals[:, 2]))
    Va_y_z_max_diff = np.max(np.abs(Va_vals[:, 1] - Va_vals[:, 2]))
    
    Vx_x_y_max_diff = np.max(np.abs(Vx_vals[:, 0] - Vx_vals[:, 1]))
    Vx_x_z_max_diff = np.max(np.abs(Vx_vals[:, 0] - Vx_vals[:, 2]))
    Vx_y_z_max_diff = np.max(np.abs(Vx_vals[:, 1] - Vx_vals[:, 2]))
    
    # Check if any components are equal
    has_equal_components = (Va_x_equals_y or Va_x_equals_z or Va_y_equals_z or 
                           Vx_x_equals_y or Vx_x_equals_z or Vx_y_equals_z)
    
    results = {
        "d": d,
        "Va_x_equals_y": Va_x_equals_y,
        "Va_x_equals_z": Va_x_equals_z,
        "Va_y_equals_z": Va_y_equals_z,
        "Vx_x_equals_y": Vx_x_equals_y,
        "Vx_x_equals_z": Vx_x_equals_z,
        "Vx_y_equals_z": Vx_y_equals_z,
        "Va_x_y_max_diff": Va_x_y_max_diff,
        "Va_x_z_max_diff": Va_x_z_max_diff,
        "Va_y_z_max_diff": Va_y_z_max_diff,
        "Vx_x_y_max_diff": Vx_x_y_max_diff,
        "Vx_x_z_max_diff": Vx_x_z_max_diff,
        "Vx_y_z_max_diff": Vx_y_z_max_diff,
        "has_equal_components": has_equal_components,
        "theta_vals": theta_vals,
        "Va_vals": Va_vals,
        "Vx_vals": Vx_vals,
        "R_thetas": R_thetas
    }
    
    if has_equal_components and plot_dir is not None:
        # Create plots directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        
        # Report which components are equal
        if Va_x_equals_y:
            print(f"🎯 Va x=y components equal for all theta at d = {d:.6f}")
        if Va_x_equals_z:
            print(f"🎯 Va x=z components equal for all theta at d = {d:.6f}")
        if Va_y_equals_z:
            print(f"🎯 Va y=z components equal for all theta at d = {d:.6f}")
        if Vx_x_equals_y:
            print(f"🎯 Vx x=y components equal for all theta at d = {d:.6f}")
        if Vx_x_equals_z:
            print(f"🎯 Vx x=z components equal for all theta at d = {d:.6f}")
        if Vx_y_equals_z:
            print(f"🎯 Vx y=z components equal for all theta at d = {d:.6f}")
        
        # Plot Va components
        plt.figure(figsize=(10, 5))
        plt.plot(theta_vals, Va_vals[:, 0], 'r-', label='Va x component')
        plt.plot(theta_vals, Va_vals[:, 1], 'g-', label='Va y component')
        plt.plot(theta_vals, Va_vals[:, 2], 'b-', label='Va z component')
        plt.xlabel('Theta')
        plt.ylabel('Va')
        plt.title(f'Va Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Va_components_{d:.6f}.png'))
        plt.close()
        
        # Plot Vx components
        plt.figure(figsize=(10, 5))
        plt.plot(theta_vals, Vx_vals[:, 0], 'r-', label='Vx x component')
        plt.plot(theta_vals, Vx_vals[:, 1], 'g-', label='Vx y component')
        plt.plot(theta_vals, Vx_vals[:, 2], 'b-', label='Vx z component')
        plt.xlabel('Theta')
        plt.ylabel('Vx')
        plt.title(f'Vx Components vs Theta (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'Vx_components_{d:.6f}.png'))
        plt.close()
        
        # Plot component differences
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(theta_vals, Va_vals[:, 0] - Va_vals[:, 1], 'r-', label='Va(x) - Va(y)')
        plt.plot(theta_vals, Va_vals[:, 0] - Va_vals[:, 2], 'g-', label='Va(x) - Va(z)')
        plt.plot(theta_vals, Va_vals[:, 1] - Va_vals[:, 2], 'b-', label='Va(y) - Va(z)')
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Component Difference')
        plt.title(f'Va Component Differences (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(theta_vals, Vx_vals[:, 0] - Vx_vals[:, 1], 'r-', label='Vx(x) - Vx(y)')
        plt.plot(theta_vals, Vx_vals[:, 0] - Vx_vals[:, 2], 'g-', label='Vx(x) - Vx(z)')
        plt.plot(theta_vals, Vx_vals[:, 1] - Vx_vals[:, 2], 'b-', label='Vx(y) - Vx(z)')
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Theta')
        plt.ylabel('Component Difference')
        plt.title(f'Vx Component Differences (d = {d:.6f})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'component_differences_{d:.6f}.png'))
        plt.close()
    
    return results

def search_equal_components(d_start, d_end, d_step, aVx, aVa, x_shift, c_const, omega, R_0, 
                           num_theta=100, epsilon=1e-5, track_min_diff=True):
    """
    Search for d values where Va or Vx components are equal across all theta values
    
    Parameters:
    d_start, d_end: Range of d values to search
    d_step: Step size for d values
    aVx, aVa, x_shift, c_const, omega: Parameters for the Hamiltonian
    R_0: Origin vector
    num_theta: Number of theta values to check
    epsilon: Tolerance for equality comparison
    track_min_diff: Whether to track minimum differences
    
    Returns:
    list: List of d values and results where components are equal
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'equal_components_theta_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Store d values where components are equal
    equal_component_points = []
    
    # Track minimum differences if requested
    if track_min_diff:
        min_diffs = {
            "Va_x_y": {"diff": float('inf'), "d": None},
            "Va_x_z": {"diff": float('inf'), "d": None},
            "Va_y_z": {"diff": float('inf'), "d": None},
            "Vx_x_y": {"diff": float('inf'), "d": None},
            "Vx_x_z": {"diff": float('inf'), "d": None},
            "Vx_y_z": {"diff": float('inf'), "d": None}
        }
    
    # Generate d values to search
    d_values = np.arange(d_start, d_end + d_step/2, d_step)
    
    # Create a progress bar
    pbar = tqdm.tqdm(total=len(d_values), desc="Searching for equal components")
    
    # Open files to write results
    results_file = os.path.join(parent_dir, 'equal_component_points.txt')
    with open(results_file, 'w') as f:
        f.write(f"d values where components are equal across all theta (epsilon={epsilon}):\n")
    
    if track_min_diff:
        diff_file = os.path.join(parent_dir, 'component_differences.txt')
        with open(diff_file, 'w') as f:
            f.write("d\tVa_x_y\tVa_x_z\tVa_y_z\tVx_x_y\tVx_x_z\tVx_y_z\n")
    
    # Search through d values
    for d in d_values:
        # Create a directory for plots if we find equal components
        plot_dir = None
        
        # Analyze this d value
        results = analyze_d_value_full_theta(
            d, aVx, aVa, x_shift, c_const, omega, R_0, 
            num_theta=num_theta, epsilon=epsilon, plot_dir=plot_dir
        )
        
        # Update progress bar
        pbar.update(1)
        
        # Track minimum differences if requested
        if track_min_diff:
            # Check if we have a new minimum difference for any component pair
            if results["Va_x_y_max_diff"] < min_diffs["Va_x_y"]["diff"]:
                min_diffs["Va_x_y"]["diff"] = results["Va_x_y_max_diff"]
                min_diffs["Va_x_y"]["d"] = d
                print(f"New minimum Va x-y diff: {min_diffs['Va_x_y']['diff']:.10f} at d = {d:.6f}")
            
            if results["Va_x_z_max_diff"] < min_diffs["Va_x_z"]["diff"]:
                min_diffs["Va_x_z"]["diff"] = results["Va_x_z_max_diff"]
                min_diffs["Va_x_z"]["d"] = d
                print(f"New minimum Va x-z diff: {min_diffs['Va_x_z']['diff']:.10f} at d = {d:.6f}")
            
            if results["Va_y_z_max_diff"] < min_diffs["Va_y_z"]["diff"]:
                min_diffs["Va_y_z"]["diff"] = results["Va_y_z_max_diff"]
                min_diffs["Va_y_z"]["d"] = d
                print(f"New minimum Va y-z diff: {min_diffs['Va_y_z']['diff']:.10f} at d = {d:.6f}")
            
            if results["Vx_x_y_max_diff"] < min_diffs["Vx_x_y"]["diff"]:
                min_diffs["Vx_x_y"]["diff"] = results["Vx_x_y_max_diff"]
                min_diffs["Vx_x_y"]["d"] = d
                print(f"New minimum Vx x-y diff: {min_diffs['Vx_x_y']['diff']:.10f} at d = {d:.6f}")
            
            if results["Vx_x_z_max_diff"] < min_diffs["Vx_x_z"]["diff"]:
                min_diffs["Vx_x_z"]["diff"] = results["Vx_x_z_max_diff"]
                min_diffs["Vx_x_z"]["d"] = d
                print(f"New minimum Vx x-z diff: {min_diffs['Vx_x_z']['diff']:.10f} at d = {d:.6f}")
            
            if results["Vx_y_z_max_diff"] < min_diffs["Vx_y_z"]["diff"]:
                min_diffs["Vx_y_z"]["diff"] = results["Vx_y_z_max_diff"]
                min_diffs["Vx_y_z"]["d"] = d
                print(f"New minimum Vx y-z diff: {min_diffs['Vx_y_z']['diff']:.10f} at d = {d:.6f}")
            
            # Write to differences file every 10 steps to avoid excessive I/O
            if pbar.n % 10 == 0:
                with open(diff_file, 'a') as f:
                    f.write(f"{d:.6f}\t{results['Va_x_y_max_diff']:.10f}\t{results['Va_x_z_max_diff']:.10f}\t"
                            f"{results['Va_y_z_max_diff']:.10f}\t{results['Vx_x_y_max_diff']:.10f}\t"
                            f"{results['Vx_x_z_max_diff']:.10f}\t{results['Vx_y_z_max_diff']:.10f}\n")
        
        if results["has_equal_components"]:
            # Create plot directory now that we found a match
            plot_dir = os.path.join(parent_dir, f'plots_d_{d:.6f}')
            
            # Re-analyze with plot directory to generate plots
            results = analyze_d_value_full_theta(
                d, aVx, aVa, x_shift, c_const, omega, R_0, 
                num_theta=num_theta, epsilon=epsilon, plot_dir=plot_dir
            )
            
            # Add this d value to our list
            equal_component_points.append((d, results))
            
            # Append to the results file
            with open(results_file, 'a') as f:
                f.write(f"d = {d:.10f}:\n")
                if results["Va_x_equals_y"]:
                    f.write(f"  Va(x) = Va(y), max diff = {results['Va_x_y_max_diff']:.10f}\n")
                if results["Va_x_equals_z"]:
                    f.write(f"  Va(x) = Va(z), max diff = {results['Va_x_z_max_diff']:.10f}\n")
                if results["Va_y_equals_z"]:
                    f.write(f"  Va(y) = Va(z), max diff = {results['Va_y_z_max_diff']:.10f}\n")
                if results["Vx_x_equals_y"]:
                    f.write(f"  Vx(x) = Vx(y), max diff = {results['Vx_x_y_max_diff']:.10f}\n")
                if results["Vx_x_equals_z"]:
                    f.write(f"  Vx(x) = Vx(z), max diff = {results['Vx_x_z_max_diff']:.10f}\n")
                if results["Vx_y_equals_z"]:
                    f.write(f"  Vx(y) = Vx(z), max diff = {results['Vx_y_z_max_diff']:.10f}\n")
                f.write("\n")
        
        # Update the progress bar description
        pbar.set_description(f"Searching (found {len(equal_component_points)} points)")
    
    # Close the progress bar
    pbar.close()
    
    # Write summary of minimum differences if tracking
    if track_min_diff:
        min_diff_file = os.path.join(parent_dir, 'minimum_differences.txt')
        with open(min_diff_file, 'w') as f:
            f.write("Component pair\tMinimum difference\td value\n")
            for pair, data in min_diffs.items():
                f.write(f"{pair}\t{data['diff']:.10f}\t{data['d']:.10f}\n")
    
    print(f"Search complete. Found {len(equal_component_points)} points with equal components.")
    
    return equal_component_points, min_diffs if track_min_diff else None

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
    parser = argparse.ArgumentParser(description='Search for d values where Va or Vx components are equal across all theta.')
    parser.add_argument('--aVx', type=float, default=aVx, help=f'Parameter aVx (default: {aVx})')
    parser.add_argument('--aVa', type=float, default=aVa, help=f'Parameter aVa (default: {aVa})')
    parser.add_argument('--x_shift', type=float, default=x_shift, help=f'Parameter x_shift (default: {x_shift})')
    parser.add_argument('--c_const', type=float, default=c_const, help=f'Parameter c_const (default: {c_const})')
    parser.add_argument('--omega', type=float, default=omega, help=f'Parameter omega (default: {omega})')
    parser.add_argument('--d_start', type=float, default=d_start, help=f'Start of d range (default: {d_start})')
    parser.add_argument('--d_end', type=float, default=d_end, help=f'End of d range (default: {d_end})')
    parser.add_argument('--d_step', type=float, default=d_step, help=f'Step size for d values (default: {d_step})')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Tolerance for equality comparison (default: 1e-5)')
    parser.add_argument('--num_theta', type=int, default=100, help='Number of theta values to check (default: 100)')
    parser.add_argument('--no_track_min', action='store_true', help='Disable tracking of minimum differences')
    parser.add_argument('--focus_d', type=float, help='Focus search around this d value with higher precision')
    parser.add_argument('--focus_range', type=float, default=0.001, help='Range around focus_d to search (default: 0.001)')
    parser.add_argument('--focus_step', type=float, default=0.00001, help='Step size for focus search (default: 0.00001)')
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
    epsilon = args.epsilon
    num_theta = args.num_theta
    track_min_diff = not args.no_track_min
    
    # Define the origin vector R_0
    R_0 = create_perfect_orthogonal_vectors(1.0)
    
    # If analyze_d is provided, just analyze that single d value
    if args.analyze_d is not None:
        d = args.analyze_d
        print(f"Analyzing single d value: {d}")
        
        # Create timestamp for output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        parent_dir = f'analyze_d_{d:.6f}_{timestamp}'
        os.makedirs(parent_dir, exist_ok=True)
        
        plot_dir = os.path.join(parent_dir, f'plots')
        results = analyze_d_value_full_theta(
            d, aVx, aVa, x_shift, c_const, omega, R_0, 
            num_theta=num_theta, epsilon=epsilon, plot_dir=plot_dir
        )
        
        # Write detailed results to file
        results_file = os.path.join(parent_dir, f'analysis_d_{d:.6f}.txt')
        with open(results_file, 'w') as f:
            f.write(f"Analysis of d = {d:.10f}:\n\n")
            f.write("Component equality:\n")
            f.write(f"Va(x) = Va(y): {results['Va_x_equals_y']}, max diff = {results['Va_x_y_max_diff']:.10f}\n")
            f.write(f"Va(x) = Va(z): {results['Va_x_equals_z']}, max diff = {results['Va_x_z_max_diff']:.10f}\n")
            f.write(f"Va(y) = Va(z): {results['Va_y_equals_z']}, max diff = {results['Va_y_z_max_diff']:.10f}\n")
            f.write(f"Vx(x) = Vx(y): {results['Vx_x_equals_y']}, max diff = {results['Vx_x_y_max_diff']:.10f}\n")
            f.write(f"Vx(x) = Vx(z): {results['Vx_x_equals_z']}, max diff = {results['Vx_x_z_max_diff']:.10f}\n")
            f.write(f"Vx(y) = Vx(z): {results['Vx_y_equals_z']}, max diff = {results['Vx_y_z_max_diff']:.10f}\n")
        
        print(f"Analysis complete. Results saved to {results_file}")
        sys.exit(0)
    
    # If focus_d is provided, perform a focused search around that value
    if args.focus_d is not None:
        focus_d = args.focus_d
        focus_range = args.focus_range
        focus_step = args.focus_step
        
        d_start = max(0.001, focus_d - focus_range)
        d_end = focus_d + focus_range
        d_step = focus_step
        
        print(f"Performing focused search around d={focus_d} with range ±{focus_range}")
    
    print(f"Starting search from d={d_start} to d={d_end} with step={d_step}")
    print(f"Parameters: aVx={aVx}, aVa={aVa}, x_shift={x_shift}, c_const={c_const}, omega={omega}")
    print(f"Using {num_theta} theta points, epsilon={epsilon}, track_min_diff={track_min_diff}")
    
    # Perform the search
    equal_points, min_diffs = search_equal_components(
        d_start, d_end, d_step, aVx, aVa, x_shift, c_const, omega, R_0,
        num_theta=num_theta, epsilon=epsilon, track_min_diff=track_min_diff
    )
    
    # Print summary of results
    if equal_points:
        print("\nFound points with equal components:")
        for d, results in equal_points:
            print(f"d = {d:.10f}:")
            if results["Va_x_equals_y"]:
                print(f"  Va(x) = Va(y), max diff = {results['Va_x_y_max_diff']:.10f}")
            if results["Va_x_equals_z"]:
                print(f"  Va(x) = Va(z), max diff = {results['Va_x_z_max_diff']:.10f}")
            if results["Va_y_equals_z"]:
                print(f"  Va(y) = Va(z), max diff = {results['Va_y_z_max_diff']:.10f}")
            if results["Vx_x_equals_y"]:
                print(f"  Vx(x) = Vx(y), max diff = {results['Vx_x_y_max_diff']:.10f}")
            if results["Vx_x_equals_z"]:
                print(f"  Vx(x) = Vx(z), max diff = {results['Vx_x_z_max_diff']:.10f}")
            if results["Vx_y_equals_z"]:
                print(f"  Vx(y) = Vx(z), max diff = {results['Vx_y_z_max_diff']:.10f}")
    
    if track_min_diff:
        print("\nMinimum differences found:")
        for pair, data in min_diffs.items():
            print(f"{pair}: {data['diff']:.10f} at d = {data['d']:.10f}")
