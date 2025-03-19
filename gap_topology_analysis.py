#!/usr/bin/env python3
"""
Gap-Topology Analysis for Berry Phase System

This script extends the energy gap analysis to explore the relationship between
energy gaps and topological properties (Berry phases) in the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import argparse
from scipy.signal import find_peaks

# Import functions from improved_berry_phase.py
from improved_berry_phase import (
    hamiltonian, R_theta, V_x, V_a, 
    berry_connection_analytical, berry_phase_integration,
    hbar
)

# Import functions from energy_gap_analysis.py
from energy_gap_analysis import (
    calculate_energy_spectrum, calculate_energy_gaps, analyze_minimum_gaps,
    calculate_gap_statistics, calculate_adjacent_gaps, calculate_all_gaps
)

def scan_parameter_space(param_ranges, fixed_params, num_points=20):
    """
    Scan through parameter space and analyze how energy gaps and Berry phases change.
    
    Parameters:
    param_ranges (dict): Dictionary of parameter ranges to scan through
    fixed_params (dict): Dictionary of fixed parameters
    num_points (int): Number of points to sample for each parameter
    
    Returns:
    dict: Results of the parameter scan
    """
    # Determine which parameters to scan
    scan_params = list(param_ranges.keys())
    if len(scan_params) > 2:
        print("Warning: Only the first two parameters will be used for 2D scanning")
        scan_params = scan_params[:2]
    
    # Create parameter grids
    if len(scan_params) == 1:
        # 1D scan
        param1 = scan_params[0]
        param1_vals = np.linspace(param_ranges[param1][0], param_ranges[param1][1], num_points)
        param_grid = [(param1, val) for val in param1_vals]
        
        # Initialize results arrays
        adjacent_min_gaps = [None] * num_points
        all_min_gaps = [None] * num_points
        berry_phases = [None] * num_points
        
        # Scan through parameter space
        for i, (param, val) in enumerate(param_grid):
            print(f"Scanning {param}={val:.4f} ({i+1}/{len(param_grid)})")
            
            # Set up parameters for this point
            params = fixed_params.copy()
            params[param] = val
            
            # Calculate spectrum, gaps, and Berry phases
            result = calculate_single_point(params)
            
            # Store results
            adjacent_min_gaps[i] = result['adjacent_min_gaps']
            all_min_gaps[i] = result['all_min_gaps']
            berry_phases[i] = result['berry_phases']
        
        return {
            'type': '1D',
            'param1': param1,
            'param1_vals': param1_vals,
            'adjacent_min_gaps': adjacent_min_gaps,
            'all_min_gaps': all_min_gaps,
            'berry_phases': berry_phases
        }
    
    else:
        # 2D scan
        param1, param2 = scan_params
        param1_vals = np.linspace(param_ranges[param1][0], param_ranges[param1][1], num_points)
        param2_vals = np.linspace(param_ranges[param2][0], param_ranges[param2][1], num_points)
        
        # Initialize results arrays
        adjacent_min_gaps = np.empty((num_points, num_points), dtype=object)
        all_min_gaps = np.empty((num_points, num_points), dtype=object)
        berry_phases = np.empty((num_points, num_points), dtype=object)
        
        # Scan through parameter space
        total_points = num_points * num_points
        point_count = 0
        
        for i, val1 in enumerate(param1_vals):
            for j, val2 in enumerate(param2_vals):
                point_count += 1
                print(f"Scanning {param1}={val1:.4f}, {param2}={val2:.4f} ({point_count}/{total_points})")
                
                # Set up parameters for this point
                params = fixed_params.copy()
                params[param1] = val1
                params[param2] = val2
                
                # Calculate spectrum, gaps, and Berry phases
                result = calculate_single_point(params)
                
                # Store results
                adjacent_min_gaps[i, j] = result['adjacent_min_gaps']
                all_min_gaps[i, j] = result['all_min_gaps']
                berry_phases[i, j] = result['berry_phases']
        
        return {
            'type': '2D',
            'param1': param1,
            'param1_vals': param1_vals,
            'param2': param2,
            'param2_vals': param2_vals,
            'adjacent_min_gaps': adjacent_min_gaps,
            'all_min_gaps': all_min_gaps,
            'berry_phases': berry_phases
        }

def calculate_single_point(params):
    """
    Calculate energy spectrum, gaps, and Berry phases for a single parameter point.
    
    Parameters:
    params (dict): Dictionary of parameters
    
    Returns:
    dict: Results for this parameter point
    """
    # Calculate energy spectrum
    num_points = 100  # Reduced resolution for faster calculation
    
    # Prepare parameters for energy_gap_analysis functions
    adapted_params = {
        'c': params.get('c', 0.2),
        'omega': params.get('omega', 0.025),
        'a_vx': params.get('a_vx', 0.018),  # Coefficient for x^2 term in Vx
        'b_vx': params.get('b_vx', 0.0),    # Coefficient for x term in Vx
        'c_vx': params.get('c_vx', 0.0),    # Constant term in Vx
        'a_va': params.get('a_va', 0.42),   # Coefficient for x^2 term in Va
        'b_va': params.get('b_va', 0.0),    # Coefficient for x term in Va
        'c_va': params.get('c_va', 0.0),    # Constant term in Va
        'x_shift': params.get('x_shift', 22.5),
        'y_shift': params.get('y_shift', 547.7222222222222),
        'd': params.get('d', 0.005),
        'num_points': num_points
    }
    
    # Calculate energy spectrum
    theta_vals, eigenvalues, eigenstates, _, _, _ = calculate_energy_spectrum(adapted_params)
    
    # Calculate adjacent energy gaps
    adjacent_gaps, adjacent_gap_indices = calculate_adjacent_gaps(eigenvalues)
    
    # Calculate all energy gaps (including non-adjacent)
    all_gaps, all_gap_indices = calculate_all_gaps(eigenvalues)
    
    # Analyze minimum gaps for both adjacent and all gaps
    adjacent_min_gap_info = analyze_minimum_gaps(theta_vals, adjacent_gaps, adjacent_gap_indices)
    all_min_gap_info = analyze_minimum_gaps(theta_vals, all_gaps, all_gap_indices)
    
    # Calculate Berry connection and phase
    A = berry_connection_analytical(theta_vals, adapted_params['c'])
    
    # Calculate total Berry phase for each state
    # The berry_phase_integration function expects a 2D array of Berry connections
    # but returns a 1D array of phases, so we can use it directly
    total_berry_phases = berry_phase_integration(A, theta_vals)
    
    return {
        'eigenvalues': eigenvalues,
        'adjacent_gaps': adjacent_gaps,
        'adjacent_gap_indices': adjacent_gap_indices,
        'adjacent_min_gaps': adjacent_min_gap_info,
        'all_gaps': all_gaps,
        'all_gap_indices': all_gap_indices,
        'all_min_gaps': all_min_gap_info,
        'berry_phases': total_berry_phases
    }

def plot_1d_scan_results(results, output_dir=None):
    """
    Plot the results of a 1D parameter scan.
    
    Parameters:
    results (dict): Results from the parameter scan
    output_dir (str, optional): Directory to save the plots
    
    Returns:
    None
    """
    param1 = results['param1']
    param1_vals = results['param1_vals']
    adjacent_min_gaps = results['adjacent_min_gaps']
    all_min_gaps = results['all_min_gaps']
    berry_phases = results['berry_phases']
    
    # Create output subdirectories
    if output_dir:
        adjacent_dir = os.path.join(output_dir, "adjacent_gaps")
        all_gaps_dir = os.path.join(output_dir, "all_gaps")
        os.makedirs(adjacent_dir, exist_ok=True)
        os.makedirs(all_gaps_dir, exist_ok=True)
    
    # Plot adjacent gaps
    fig_adjacent, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot minimum adjacent gaps
    for gap_idx, info in adjacent_min_gaps[0].items():
        # Extract the minimum gap values for this gap across all parameter values
        min_gap_values = np.array([adjacent_min_gaps[i][gap_idx]['min_gap'] for i in range(len(param1_vals))])
        
        # Get state information if available
        if 'states' in info:
            state_i, state_j = info['states']
            label = f'Gap {state_i}-{state_j}'
        else:
            label = f'Gap {gap_idx}-{gap_idx+1}'
            
        ax1.plot(param1_vals, min_gap_values, label=label)
    
    ax1.set_ylabel('Minimum Energy Gap')
    ax1.set_title(f'Adjacent Minimum Energy Gaps vs {param1}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Berry phases
    for i in range(len(param1_vals)):
        # Extract Berry phases for all states at this parameter value
        phases = berry_phases[i]
        for state_idx, phase in enumerate(phases):
            if i == 0:  # Only add label for the first point to avoid duplicate labels
                ax2.plot(param1_vals[i], phase, 'o', label=f'State {state_idx}')
            else:
                ax2.plot(param1_vals[i], phase, 'o')
    
    ax2.set_xlabel(param1)
    ax2.set_ylabel('Berry Phase')
    ax2.set_title(f'Berry Phases vs {param1}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(adjacent_dir, f'adjacent_gaps_{param1}.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig_adjacent)
    
    # Plot all gaps (including non-adjacent)
    fig_all, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot minimum all gaps
    for gap_idx, info in all_min_gaps[0].items():
        # Extract the minimum gap values for this gap across all parameter values
        min_gap_values = np.array([all_min_gaps[i][gap_idx]['min_gap'] for i in range(len(param1_vals))])
        
        # Get state information if available
        if 'states' in info:
            state_i, state_j = info['states']
            # Only plot non-adjacent gaps
            if state_j - state_i > 1:
                label = f'Gap {state_i}-{state_j}'
                ax1.plot(param1_vals, min_gap_values, label=label)
    
    ax1.set_ylabel('Minimum Energy Gap')
    ax1.set_title(f'Non-Adjacent Minimum Energy Gaps vs {param1}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Berry phases (same as before)
    for i in range(len(param1_vals)):
        phases = berry_phases[i]
        for state_idx, phase in enumerate(phases):
            if i == 0:  # Only add label for the first point
                ax2.plot(param1_vals[i], phase, 'o', label=f'State {state_idx}')
            else:
                ax2.plot(param1_vals[i], phase, 'o')
    
    ax2.set_xlabel(param1)
    ax2.set_ylabel('Berry Phase')
    ax2.set_title(f'Berry Phases vs {param1}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(all_gaps_dir, f'all_gaps_{param1}.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig_all)
    
    # Create correlation plots between minimum gaps and Berry phase differences
    # For adjacent gaps
    plt.figure(figsize=(8, 6))
    
    # Find the gap between states 1 and 2 (if it exists)
    gap_1_2_values = None
    for gap_idx, info in adjacent_min_gaps[0].items():
        if 'states' in info and info['states'] == (1, 2):
            gap_1_2_values = np.array([adjacent_min_gaps[i][gap_idx]['min_gap'] for i in range(len(param1_vals))])
            break
    
    if gap_1_2_values is not None:
        # Calculate Berry phase difference between states 1 and 2
        berry_phase_diffs = []
        for i in range(len(param1_vals)):
            phases = berry_phases[i]
            if len(phases) > 2:  # Make sure we have at least 3 states
                berry_phase_diffs.append(np.abs(phases[1] - phases[2]))
        
        plt.scatter(gap_1_2_values, berry_phase_diffs, c=param1_vals, cmap='viridis')
        plt.colorbar(label=param1)
        
        plt.xlabel('Minimum Gap 1-2')
        plt.ylabel('|Berry Phase 1 - Berry Phase 2|')
        plt.title('Correlation between Adjacent Gap 1-2 and Berry Phase Difference')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(adjacent_dir, f'adjacent_gap_phase_correlation_{param1}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    plt.close()
    
    # For non-adjacent gaps
    plt.figure(figsize=(8, 6))
    
    # Find a non-adjacent gap (e.g., 0-2 or 1-3)
    non_adjacent_gap_values = None
    non_adjacent_gap_label = ""
    for gap_idx, info in all_min_gaps[0].items():
        if 'states' in info:
            state_i, state_j = info['states']
            if state_j - state_i > 1:  # Non-adjacent gap
                non_adjacent_gap_values = np.array([all_min_gaps[i][gap_idx]['min_gap'] for i in range(len(param1_vals))])
                non_adjacent_gap_label = f"Gap {state_i}-{state_j}"
                break
    
    if non_adjacent_gap_values is not None:
        # Calculate Berry phase difference between the corresponding states
        berry_phase_diffs = []
        for i in range(len(param1_vals)):
            phases = berry_phases[i]
            if len(phases) > state_j:  # Make sure we have enough states
                berry_phase_diffs.append(np.abs(phases[state_i] - phases[state_j]))
        
        plt.scatter(non_adjacent_gap_values, berry_phase_diffs, c=param1_vals, cmap='viridis')
        plt.colorbar(label=param1)
        
        plt.xlabel(f'Minimum {non_adjacent_gap_label}')
        plt.ylabel(f'|Berry Phase {state_i} - Berry Phase {state_j}|')
        plt.title(f'Correlation between Non-Adjacent {non_adjacent_gap_label} and Berry Phase Difference')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(all_gaps_dir, f'non_adjacent_gap_phase_correlation_{param1}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    plt.close()

def plot_2d_scan_results(results, output_dir=None):
    """
    Plot the results of a 2D parameter scan.
    
    Parameters:
    results (dict): Results from the parameter scan
    output_dir (str, optional): Directory to save the plots
    
    Returns:
    None
    """
    param1 = results['param1']
    param1_vals = results['param1_vals']
    param2 = results['param2']
    param2_vals = results['param2_vals']
    adjacent_min_gaps = results['adjacent_min_gaps']
    all_min_gaps = results['all_min_gaps']
    berry_phases = results['berry_phases']
    
    # Create output subdirectories
    if output_dir:
        adjacent_dir = os.path.join(output_dir, "adjacent_gaps")
        all_gaps_dir = os.path.join(output_dir, "all_gaps")
        os.makedirs(adjacent_dir, exist_ok=True)
        os.makedirs(all_gaps_dir, exist_ok=True)
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(param1_vals, param2_vals)
    
    # Plot adjacent minimum gaps as 2D heatmaps
    # First, identify which gaps exist in the data
    gap_indices = set()
    for i in range(len(param1_vals)):
        for j in range(len(param2_vals)):
            if i < len(param1_vals) and j < len(param2_vals):
                gap_indices.update(adjacent_min_gaps[i, j].keys())
    
    for gap_idx in gap_indices:
        plt.figure(figsize=(10, 8))
        
        # Create a 2D array for this gap
        Z = np.zeros((len(param2_vals), len(param1_vals)))
        
        # Fill in the values
        for i in range(len(param1_vals)):
            for j in range(len(param2_vals)):
                if i < len(param1_vals) and j < len(param2_vals) and gap_idx in adjacent_min_gaps[i, j]:
                    Z[j, i] = adjacent_min_gaps[i, j][gap_idx]['min_gap']
        
        # Get state information if available
        if 'states' in adjacent_min_gaps[0, 0][gap_idx]:
            state_i, state_j = adjacent_min_gaps[0, 0][gap_idx]['states']
            gap_label = f'Gap {state_i}-{state_j}'
        else:
            gap_label = f'Gap {gap_idx}-{gap_idx+1}'
        
        # Use log scale for better visualization
        if np.min(Z) <= 0:
            # Add small offset to avoid log(0)
            Z = Z + 1e-10
        
        plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto', norm='log')
        plt.colorbar(label=f'Minimum {gap_label}')
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Adjacent {gap_label} vs {param1} and {param2}')
        
        if output_dir:
            plt.savefig(os.path.join(adjacent_dir, f'2d_scan_adjacent_{gap_label.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Plot non-adjacent minimum gaps as 2D heatmaps
    # First, identify which non-adjacent gaps exist in the data
    non_adjacent_gap_indices = set()
    for i in range(len(param1_vals)):
        for j in range(len(param2_vals)):
            if i < len(param1_vals) and j < len(param2_vals):
                for gap_idx, info in all_min_gaps[i, j].items():
                    if 'states' in info:
                        state_i, state_j = info['states']
                        if state_j - state_i > 1:  # Non-adjacent gap
                            non_adjacent_gap_indices.add(gap_idx)
    
    for gap_idx in non_adjacent_gap_indices:
        plt.figure(figsize=(10, 8))
        
        # Create a 2D array for this gap
        Z = np.zeros((len(param2_vals), len(param1_vals)))
        
        # Fill in the values
        for i in range(len(param1_vals)):
            for j in range(len(param2_vals)):
                if i < len(param1_vals) and j < len(param2_vals) and gap_idx in all_min_gaps[i, j]:
                    Z[j, i] = all_min_gaps[i, j][gap_idx]['min_gap']
        
        # Get state information
        state_i, state_j = all_min_gaps[0, 0][gap_idx]['states']
        gap_label = f'Gap {state_i}-{state_j}'
        
        # Use log scale for better visualization
        if np.min(Z) <= 0:
            # Add small offset to avoid log(0)
            Z = Z + 1e-10
        
        plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto', norm='log')
        plt.colorbar(label=f'Minimum {gap_label}')
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Non-Adjacent {gap_label} vs {param1} and {param2}')
        
        if output_dir:
            plt.savefig(os.path.join(all_gaps_dir, f'2d_scan_non_adjacent_{gap_label.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Plot Berry phases as 2D heatmaps
    # First, determine how many states we have
    num_states = 0
    for i in range(len(param1_vals)):
        for j in range(len(param2_vals)):
            if i < len(param1_vals) and j < len(param2_vals):
                num_states = max(num_states, len(berry_phases[i, j]))
    
    for state_idx in range(num_states):
        plt.figure(figsize=(10, 8))
        
        # Create a 2D array for this state's Berry phase
        Z = np.zeros((len(param2_vals), len(param1_vals)))
        
        # Fill in the values
        for i in range(len(param1_vals)):
            for j in range(len(param2_vals)):
                if i < len(param1_vals) and j < len(param2_vals) and state_idx < len(berry_phases[i, j]):
                    Z[j, i] = berry_phases[i, j][state_idx]
        
        plt.pcolormesh(X, Y, Z, cmap='RdBu', shading='auto', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Berry Phase')
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Berry Phase for State {state_idx} vs {param1} and {param2}')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'2d_berry_phase_{state_idx}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Plot Berry phase differences between states
    # For states 1 and 2 (if they exist)
    if num_states > 2:
        plt.figure(figsize=(10, 8))
        
        # Create a 2D array for the Berry phase difference
        Z = np.zeros((len(param2_vals), len(param1_vals)))
        
        # Fill in the values
        for i in range(len(param1_vals)):
            for j in range(len(param2_vals)):
                if i < len(param1_vals) and j < len(param2_vals) and len(berry_phases[i, j]) > 2:
                    Z[j, i] = np.abs(berry_phases[i, j][1] - berry_phases[i, j][2])
        
        plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(label='|Berry Phase 1 - Berry Phase 2|')
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Berry Phase Difference |Φ₁ - Φ₂| vs {param1} and {param2}')
    
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'2d_berry_phase_diff_1_2.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # For non-adjacent states (if they exist)
    # Find a pair of non-adjacent states
    non_adjacent_states = None
    for gap_idx in non_adjacent_gap_indices:
        if len(all_min_gaps) > 0 and gap_idx in all_min_gaps[0, 0]:
            state_i, state_j = all_min_gaps[0, 0][gap_idx]['states']
            if state_j - state_i > 1 and state_j < num_states:
                non_adjacent_states = (state_i, state_j)
                break
    
    if non_adjacent_states:
        state_i, state_j = non_adjacent_states
        plt.figure(figsize=(10, 8))
        
        # Create a 2D array for the Berry phase difference
        Z = np.zeros((len(param2_vals), len(param1_vals)))
        
        # Fill in the values
        for i in range(len(param1_vals)):
            for j in range(len(param2_vals)):
                if i < len(param1_vals) and j < len(param2_vals) and len(berry_phases[i, j]) > state_j:
                    Z[j, i] = np.abs(berry_phases[i, j][state_i] - berry_phases[i, j][state_j])
        
        plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(label=f'|Berry Phase {state_i} - Berry Phase {state_j}|')
        
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Berry Phase Difference |Φ_{state_i} - Φ_{state_j}| vs {param1} and {param2}')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'2d_berry_phase_diff_{state_i}_{state_j}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def create_summary_report(results, output_dir=None):
    """
    Create a summary report of the parameter scan results.
    
    Parameters:
    results (dict): Results from the parameter scan
    output_dir (str, optional): Directory to save the report
    
    Returns:
    None
    """
    report = []
    report.append("Gap-Topology Analysis Summary Report")
    report.append("===================================\n")
    
    # Add scan type and parameters
    if results['type'] == '1D':
        param1 = results['param1']
        param1_range = (results['param1_vals'][0], results['param1_vals'][-1])
        
        report.append(f"1D Parameter Scan:")
        report.append(f"  {param1}: {param1_range[0]} to {param1_range[1]}")
        
        # Find parameter values where minimum adjacent gaps are smallest
        adjacent_min_gaps = results['adjacent_min_gaps']
        all_min_gaps = results['all_min_gaps']
        berry_phases = results['berry_phases']
        
        report.append(f"\nAdjacent Minimum Gap Analysis:")
        
        # Analyze each adjacent gap
        for gap_idx in adjacent_min_gaps[0].keys():
            # Get state information if available
            if 'states' in adjacent_min_gaps[0][gap_idx]:
                state_i, state_j = adjacent_min_gaps[0][gap_idx]['states']
                gap_label = f"Gap {state_i}-{state_j}"
            else:
                gap_label = f"Gap {gap_idx}"
            
            # Extract minimum gap values across parameter range
            min_gap_values = np.array([adjacent_min_gaps[i][gap_idx]['min_gap'] for i in range(len(results['param1_vals']))])
            min_gap_idx = np.argmin(min_gap_values)
            min_gap_value = min_gap_values[min_gap_idx]
            min_gap_param = results['param1_vals'][min_gap_idx]
            
            report.append(f"  Smallest {gap_label}: {min_gap_value:.6f} at {param1}={min_gap_param:.4f}")
        
        # Find non-adjacent gaps and analyze them
        report.append(f"\nNon-Adjacent Minimum Gap Analysis:")
        non_adjacent_gaps_found = False
        
        for gap_idx in all_min_gaps[0].keys():
            if 'states' in all_min_gaps[0][gap_idx]:
                state_i, state_j = all_min_gaps[0][gap_idx]['states']
                if state_j - state_i > 1:  # Non-adjacent gap
                    non_adjacent_gaps_found = True
                    gap_label = f"Gap {state_i}-{state_j}"
                    
                    # Extract minimum gap values across parameter range
                    min_gap_values = np.array([all_min_gaps[i][gap_idx]['min_gap'] for i in range(len(results['param1_vals']))])
                    min_gap_idx = np.argmin(min_gap_values)
                    min_gap_value = min_gap_values[min_gap_idx]
                    min_gap_param = results['param1_vals'][min_gap_idx]
                    
                    report.append(f"  Smallest {gap_label}: {min_gap_value:.6f} at {param1}={min_gap_param:.4f}")
        
        if not non_adjacent_gaps_found:
            report.append(f"  No non-adjacent gaps found in the analysis.")
        
        # Find parameter values where Berry phase differences are largest
        report.append(f"\nBerry Phase Analysis:")
        
        # For adjacent states (1 and 2)
        berry_diffs = []
        for i in range(len(results['param1_vals'])):
            phases = berry_phases[i]
            if len(phases) > 2:  # Make sure we have at least 3 states
                berry_diffs.append(np.abs(phases[1] - phases[2]))
            else:
                berry_diffs.append(0)
        
        berry_diffs = np.array(berry_diffs)
        max_berry_diff_idx = np.argmax(berry_diffs)
        max_berry_diff = berry_diffs[max_berry_diff_idx]
        max_berry_diff_param = results['param1_vals'][max_berry_diff_idx]
        
        report.append(f"  Largest Berry Phase Difference (1-2): {max_berry_diff:.6f} at {param1}={max_berry_diff_param:.4f}")
        
        # For non-adjacent states
        non_adjacent_states = None
        for gap_idx in all_min_gaps[0].keys():
            if 'states' in all_min_gaps[0][gap_idx]:
                state_i, state_j = all_min_gaps[0][gap_idx]['states']
                if state_j - state_i > 1:  # Non-adjacent gap
                    non_adjacent_states = (state_i, state_j)
                    break
        
        if non_adjacent_states:
            state_i, state_j = non_adjacent_states
            berry_diffs = []
            for i in range(len(results['param1_vals'])):
                phases = berry_phases[i]
                if len(phases) > state_j:  # Make sure we have enough states
                    berry_diffs.append(np.abs(phases[state_i] - phases[state_j]))
                else:
                    berry_diffs.append(0)
            
            berry_diffs = np.array(berry_diffs)
            max_berry_diff_idx = np.argmax(berry_diffs)
            max_berry_diff = berry_diffs[max_berry_diff_idx]
            max_berry_diff_param = results['param1_vals'][max_berry_diff_idx]
            
            report.append(f"  Largest Berry Phase Difference ({state_i}-{state_j}): {max_berry_diff:.6f} at {param1}={max_berry_diff_param:.4f}")
        
    else:  # 2D scan
        param1 = results['param1']
        param1_range = (results['param1_vals'][0], results['param1_vals'][-1])
        param2 = results['param2']
        param2_range = (results['param2_vals'][0], results['param2_vals'][-1])
        
        report.append(f"2D Parameter Scan:")
        report.append(f"  {param1}: {param1_range[0]} to {param1_range[1]}")
        report.append(f"  {param2}: {param2_range[0]} to {param2_range[1]}")
        
        # Find parameter values where minimum adjacent gaps are smallest
        adjacent_min_gaps = results['adjacent_min_gaps']
        all_min_gaps = results['all_min_gaps']
        berry_phases = results['berry_phases']
        
        report.append(f"\nAdjacent Minimum Gap Analysis:")
        
        # Analyze each adjacent gap
        for gap_idx in adjacent_min_gaps[0, 0].keys():
            # Get state information if available
            if 'states' in adjacent_min_gaps[0, 0][gap_idx]:
                state_i, state_j = adjacent_min_gaps[0, 0][gap_idx]['states']
                gap_label = f"Gap {state_i}-{state_j}"
            else:
                gap_label = f"Gap {gap_idx}"
            
            # Find minimum gap value across the 2D parameter space
            min_gap_value = float('inf')
            min_gap_params = (0, 0)
            
            for i in range(len(results['param1_vals'])):
                for j in range(len(results['param2_vals'])):
                    if i < len(results['param1_vals']) and j < len(results['param2_vals']) and gap_idx in adjacent_min_gaps[i, j]:
                        gap_value = adjacent_min_gaps[i, j][gap_idx]['min_gap']
                        if gap_value < min_gap_value:
                            min_gap_value = gap_value
                            min_gap_params = (results['param1_vals'][i], results['param2_vals'][j])
            
            report.append(f"  Smallest {gap_label}: {min_gap_value:.6f} at {param1}={min_gap_params[0]:.4f}, {param2}={min_gap_params[1]:.4f}")
        
        # Find non-adjacent gaps and analyze them
        report.append(f"\nNon-Adjacent Minimum Gap Analysis:")
        non_adjacent_gaps_found = False
        
        for gap_idx in all_min_gaps[0, 0].keys():
            if 'states' in all_min_gaps[0, 0][gap_idx]:
                state_i, state_j = all_min_gaps[0, 0][gap_idx]['states']
                if state_j - state_i > 1:  # Non-adjacent gap
                    non_adjacent_gaps_found = True
                    gap_label = f"Gap {state_i}-{state_j}"
                    
                    # Find minimum gap value across the 2D parameter space
                    min_gap_value = float('inf')
                    min_gap_params = (0, 0)
                    
                    for i in range(len(results['param1_vals'])):
                        for j in range(len(results['param2_vals'])):
                            if i < len(results['param1_vals']) and j < len(results['param2_vals']) and gap_idx in all_min_gaps[i, j]:
                                gap_value = all_min_gaps[i, j][gap_idx]['min_gap']
                                if gap_value < min_gap_value:
                                    min_gap_value = gap_value
                                    min_gap_params = (results['param1_vals'][i], results['param2_vals'][j])
                    
                    report.append(f"  Smallest {gap_label}: {min_gap_value:.6f} at {param1}={min_gap_params[0]:.4f}, {param2}={min_gap_params[1]:.4f}")
        
        if not non_adjacent_gaps_found:
            report.append(f"  No non-adjacent gaps found in the analysis.")
        
        # Find parameter values where Berry phase differences are largest
        report.append(f"\nBerry Phase Analysis:")
        
        # For adjacent states (1 and 2)
        # Find maximum Berry phase difference across the 2D parameter space
        max_berry_diff = 0
        max_berry_diff_params = (0, 0)
        
        for i in range(len(results['param1_vals'])):
            for j in range(len(results['param2_vals'])):
                if i < len(results['param1_vals']) and j < len(results['param2_vals']) and len(berry_phases[i, j]) > 2:
                    berry_diff = np.abs(berry_phases[i, j][1] - berry_phases[i, j][2])
                    if berry_diff > max_berry_diff:
                        max_berry_diff = berry_diff
                        max_berry_diff_params = (results['param1_vals'][i], results['param2_vals'][j])
        
        report.append(f"  Largest Berry Phase Difference (1-2): {max_berry_diff:.6f} at {param1}={max_berry_diff_params[0]:.4f}, {param2}={max_berry_diff_params[1]:.4f}")
        
        # For non-adjacent states
        non_adjacent_states = None
        for gap_idx in all_min_gaps[0, 0].keys():
            if 'states' in all_min_gaps[0, 0][gap_idx]:
                state_i, state_j = all_min_gaps[0, 0][gap_idx]['states']
                if state_j - state_i > 1:  # Non-adjacent gap
                    non_adjacent_states = (state_i, state_j)
                    break
        
        if non_adjacent_states:
            state_i, state_j = non_adjacent_states
            max_berry_diff = 0
            max_berry_diff_params = (0, 0)
            
            for i in range(len(results['param1_vals'])):
                for j in range(len(results['param2_vals'])):
                    if i < len(results['param1_vals']) and j < len(results['param2_vals']) and len(berry_phases[i, j]) > state_j:
                        berry_diff = np.abs(berry_phases[i, j][state_i] - berry_phases[i, j][state_j])
                        if berry_diff > max_berry_diff:
                            max_berry_diff = berry_diff
                            max_berry_diff_params = (results['param1_vals'][i], results['param2_vals'][j])
            
            report.append(f"  Largest Berry Phase Difference ({state_i}-{state_j}): {max_berry_diff:.6f} at {param1}={max_berry_diff_params[0]:.4f}, {param2}={max_berry_diff_params[1]:.4f}")
    
    # Add correlation analysis
    report.append(f"\nGap-Topology Correlation Analysis:")
    
    if results['type'] == '1D':
        # Calculate correlation between adjacent gaps and Berry phase differences
        adjacent_min_gaps = results['adjacent_min_gaps']
        all_min_gaps = results['all_min_gaps']
        berry_phases = results['berry_phases']
        
        # For adjacent gaps (e.g., 1-2)
        for gap_idx in adjacent_min_gaps[0].keys():
            if 'states' in adjacent_min_gaps[0][gap_idx]:
                state_i, state_j = adjacent_min_gaps[0][gap_idx]['states']
                if state_j - state_i == 1:  # Adjacent gap
                    gap_label = f"Gap {state_i}-{state_j}"
                    
                    # Extract gap values and Berry phase differences
                    gap_values = []
                    berry_diffs = []
                    
                    for i in range(len(results['param1_vals'])):
                        if gap_idx in adjacent_min_gaps[i] and len(berry_phases[i]) > state_j:
                            gap_values.append(adjacent_min_gaps[i][gap_idx]['min_gap'])
                            berry_diffs.append(np.abs(berry_phases[i][state_i] - berry_phases[i][state_j]))
                    
                    if len(gap_values) > 1:  # Need at least 2 points for correlation
                        corr = np.corrcoef(gap_values, berry_diffs)[0, 1]
                        report.append(f"  Correlation between {gap_label} and Berry Phase Difference: {corr:.4f}")
        
        # For non-adjacent gaps
        for gap_idx in all_min_gaps[0].keys():
            if 'states' in all_min_gaps[0][gap_idx]:
                state_i, state_j = all_min_gaps[0][gap_idx]['states']
                if state_j - state_i > 1:  # Non-adjacent gap
                    gap_label = f"Gap {state_i}-{state_j}"
                    
                    # Extract gap values and Berry phase differences
                    gap_values = []
                    berry_diffs = []
                    
                    for i in range(len(results['param1_vals'])):
                        if gap_idx in all_min_gaps[i] and len(berry_phases[i]) > state_j:
                            gap_values.append(all_min_gaps[i][gap_idx]['min_gap'])
                            berry_diffs.append(np.abs(berry_phases[i][state_i] - berry_phases[i][state_j]))
                    
                    if len(gap_values) > 1:  # Need at least 2 points for correlation
                        corr = np.corrcoef(gap_values, berry_diffs)[0, 1]
                        report.append(f"  Correlation between {gap_label} and Berry Phase Difference: {corr:.4f}")
    
    elif results['type'] == '2D':
        # For 2D scans, we can report the global min/max values but correlation calculation is more complex
        report.append(f"  Detailed correlation analysis for 2D scans is not implemented.")
        report.append(f"  Please refer to the 2D heatmaps for visual correlation analysis.")
    
    report.append(f"\nGeneral Observations:")
    report.append(f"  The relationship between energy gaps and Berry phases provides insight into the topological properties of the system.")
    report.append(f"  Small energy gaps often coincide with rapid changes in Berry phase, indicating potential topological transitions.")
    report.append(f"  Regions with both small gaps and large Berry phase differences are of particular interest for further study.")
    
    # Write report to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'gap_topology_report.txt'), 'w') as f:
            f.write('\n'.join(report))
    else:
        print('\n'.join(report))

def main():
    """Main function to run the gap-topology analysis."""
    parser = argparse.ArgumentParser(description='Gap-Topology Analysis for Berry Phase System')
    
    # Add scan type argument
    parser.add_argument('--scan_type', type=str, choices=['1d', '2d'], default='1d',
                        help='Type of parameter scan (1d or 2d)')
    
    # Add scan parameters
    parser.add_argument('--scan_param1', type=str, default='c',
                        help='First parameter to scan')
    parser.add_argument('--scan_param1_min', type=float, default=0.1,
                        help='Minimum value for first scan parameter')
    parser.add_argument('--scan_param1_max', type=float, default=0.3,
                        help='Maximum value for first scan parameter')
    
    parser.add_argument('--scan_param2', type=str, default='omega',
                        help='Second parameter to scan (only for 2d scan)')
    parser.add_argument('--scan_param2_min', type=float, default=0.01,
                        help='Minimum value for second scan parameter')
    parser.add_argument('--scan_param2_max', type=float, default=0.05,
                        help='Maximum value for second scan parameter')
    
    # Add fixed parameters
    parser.add_argument('--c', type=float, default=0.2,
                        help='Fixed coupling constant')
    parser.add_argument('--omega', type=float, default=0.025,
                        help='Frequency parameter')
    
    # Vx potential parameters
    parser.add_argument('--a_vx', type=float, default=0.018,
                        help='Coefficient for x^2 term in Vx potential')
    parser.add_argument('--b_vx', type=float, default=0.0,
                        help='Coefficient for x term in Vx potential')
    parser.add_argument('--c_vx', type=float, default=0.0,
                        help='Constant term in Vx potential')
    
    # Va potential parameters
    parser.add_argument('--a_va', type=float, default=0.42,
                        help='Coefficient for x^2 term in Va potential')
    parser.add_argument('--b_va', type=float, default=0.0,
                        help='Coefficient for x term in Va potential')
    parser.add_argument('--c_va', type=float, default=0.0,
                        help='Constant term in Va potential')
    
    # Shift parameters
    parser.add_argument('--x_shift', type=float, default=22.5,
                        help='Shift for Va potential on x-axis')
    parser.add_argument('--y_shift', type=float, default=547.7222222222222,
                        help='Shift for Va potential on y-axis')
    
    # R_theta parameter
    parser.add_argument('--d', type=float, default=0.005,
                        help='Parameter for R_theta')
    
    # Add scan resolution
    parser.add_argument('--num_points', type=int, default=10,
                        help='Number of points for each parameter dimension')
    
    # Add output directory
    parser.add_argument('--output_dir', type=str, default='gap_topology_analysis',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create parameter ranges dictionary
    param_ranges = {
        args.scan_param1: (args.scan_param1_min, args.scan_param1_max)
    }
    
    if args.scan_type == '2d':
        param_ranges[args.scan_param2] = (args.scan_param2_min, args.scan_param2_max)
    
    # Create fixed parameters dictionary
    fixed_params = {
        'c': args.c,
        'omega': args.omega,
        'a_vx': args.a_vx,
        'b_vx': args.b_vx,
        'c_vx': args.c_vx,
        'a_va': args.a_va,
        'b_va': args.b_va,
        'c_va': args.c_va,
        'x_shift': args.x_shift,
        'y_shift': args.y_shift,
        'd': args.d,
        'num_points': 100  # Reduced resolution for faster calculation
    }
    
    # Remove scan parameters from fixed parameters
    for param in param_ranges:
        if param in fixed_params:
            del fixed_params[param]
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running gap-topology analysis with {args.scan_type.upper()} parameter scan:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} to {max_val}")
    print("Fixed parameters:")
    for param, val in fixed_params.items():
        print(f"  {param}: {val}")
    
    # Run parameter scan
    results = scan_parameter_space(param_ranges, fixed_params, args.num_points)
    
    # Plot results
    print("\nGenerating plots...")
    if results['type'] == '1D':
        plot_1d_scan_results(results, output_dir)
    else:
        plot_2d_scan_results(results, output_dir)
    
    # Create summary report
    print("Creating summary report...")
    create_summary_report(results, output_dir)
    
    # Save raw data
    print("Saving raw data...")
    np.savez(os.path.join(output_dir, 'gap_topology_data.npz'), **results)
    
    print(f"\nGap-topology analysis completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
