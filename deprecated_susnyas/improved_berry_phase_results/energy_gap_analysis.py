#!/usr/bin/env python3
"""
Energy Gap Analysis for Berry Phase System

This script extends the improved_berry_phase.py functionality to analyze
the energy gaps between eigenstates as a function of theta. It helps identify
degeneracies, avoided crossings, and other interesting features in the energy spectrum.
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

def calculate_energy_spectrum(params):
    """
    Calculate the energy spectrum for a range of theta values using the given parameters.
    
    Parameters:
    params (dict): Dictionary containing the parameters for the calculation
    
    Returns:
    tuple: (theta_vals, eigenvalues, eigenstates, r_theta_vectors, Vx_values, Va_values)
    """
    # Extract parameters
    c = params.get('c', 0.2)  # Fixed coupling constant
    omega = params.get('omega', 1.0)  # Frequency parameter
    a = params.get('a', 1.0)  # First coefficient for potentials
    b = params.get('b', 0.5)  # Second coefficient for potentials
    c_const = params.get('c_const', 0.0)  # Constant term in potentials
    x_shift = params.get('x_shift', 0.2)  # Shift for Va potential on x-axis
    y_shift = params.get('y_shift', 0.2)  # Shift for Va potential on y-axis
    d = params.get('d', 1.0)  # Parameter for R_theta
    num_points = params.get('num_points', 1000)  # Number of points for theta
    
    # Create theta values
    theta_vals = np.linspace(0, 2*np.pi, num_points)
    
    # Initialize arrays for storing results
    eigenvalues = []
    eigenstates = []
    r_theta_vectors = []
    Vx_values = []
    Va_values = []
    
    # Loop over theta values to compute the eigenvalues and eigenstates
    for theta in theta_vals:
        # Calculate the Hamiltonian, R_theta, Vx, and Va for this theta
        H, r_theta_vector, Vx, Va = hamiltonian(
            theta, c, omega, a, b, c_const, x_shift, y_shift, d
        )
        
        r_theta_vectors.append(r_theta_vector)
        Vx_values.append(Vx)
        Va_values.append(Va)
        
        # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Apply phase convention: make the first component of each eigenvector real and positive
        for i in range(eigvecs.shape[1]):
            # Get the phase of the first component
            phase = np.angle(eigvecs[0, i])
            # Apply the phase correction
            eigvecs[:, i] = eigvecs[:, i] * np.exp(-1j * phase)
            # Ensure the first component is real and positive
            if eigvecs[0, i].real < 0:
                eigvecs[:, i] = -eigvecs[:, i]
        
        eigenvalues.append(eigvals)
        eigenstates.append(eigvecs)
    
    # Convert to numpy arrays for easier manipulation
    eigenvalues = np.array(eigenvalues)
    eigenstates = np.array(eigenstates)
    r_theta_vectors = np.array(r_theta_vectors)
    Vx_values = np.array(Vx_values)
    Va_values = np.array(Va_values)
    
    return theta_vals, eigenvalues, eigenstates, r_theta_vectors, Vx_values, Va_values

def calculate_energy_gaps(eigenvalues, adjacent_only=True):
    """
    Calculate the energy gaps between eigenstates.
    
    Parameters:
    eigenvalues (numpy.ndarray): Array of eigenvalues with shape (num_points, num_states)
    adjacent_only (bool): If True, only calculate gaps between adjacent states.
                          If False, calculate gaps between all pairs of states.
    
    Returns:
    tuple: (gaps, gap_indices) where:
           - gaps is a numpy.ndarray of energy gaps
           - gap_indices is a dictionary mapping gap indices to state pairs
    """
    num_points, num_states = eigenvalues.shape
    
    if adjacent_only:
        # Calculate gaps between adjacent states only
        gaps = np.abs(np.diff(eigenvalues, axis=1))
        gap_indices = {}
        
        for i in range(num_states-1):
            gap_indices[i] = (i, i+1)  # Map gap index to state pair
        
        return gaps, gap_indices
    else:
        # Calculate gaps between all pairs of states
        num_gaps = num_states * (num_states - 1) // 2  # Number of unique pairs
        gaps = np.zeros((num_points, num_gaps))
        gap_indices = {}
        
        gap_idx = 0
        for i in range(num_states):
            for j in range(i+1, num_states):
                gaps[:, gap_idx] = np.abs(eigenvalues[:, j] - eigenvalues[:, i])
                gap_indices[gap_idx] = (i, j)  # Map gap index to state pair
                gap_idx += 1
        
        return gaps, gap_indices


def calculate_adjacent_gaps(eigenvalues):
    """
    Calculate the energy gaps between adjacent eigenstates.
    
    Parameters:
    eigenvalues (numpy.ndarray): Array of eigenvalues with shape (num_points, num_states)
    
    Returns:
    tuple: (gaps, gap_indices) where:
           - gaps is a numpy.ndarray of energy gaps
           - gap_indices is a dictionary mapping gap indices to state pairs
    """
    return calculate_energy_gaps(eigenvalues, adjacent_only=True)


def calculate_all_gaps(eigenvalues):
    """
    Calculate the energy gaps between all pairs of eigenstates.
    
    Parameters:
    eigenvalues (numpy.ndarray): Array of eigenvalues with shape (num_points, num_states)
    
    Returns:
    tuple: (gaps, gap_indices) where:
           - gaps is a numpy.ndarray of energy gaps
           - gap_indices is a dictionary mapping gap indices to state pairs
    """
    return calculate_energy_gaps(eigenvalues, adjacent_only=False)

def analyze_minimum_gaps(theta_vals, gaps, gap_indices=None):
    """
    Analyze the minimum energy gaps and find where they occur.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values
    gaps (numpy.ndarray): Array of energy gaps
    gap_indices (dict, optional): Dictionary mapping gap indices to state pairs
    
    Returns:
    dict: Dictionary containing minimum gap information for each gap
    """
    num_gaps = gaps.shape[1]
    min_gap_info = {}
    
    for i in range(num_gaps):
        # Find the minimum gap
        min_gap = np.min(gaps[:, i])
        min_gap_idx = np.argmin(gaps[:, i])
        min_gap_theta = theta_vals[min_gap_idx]
        
        # Find all local minima
        # Invert the gaps to find peaks (which are minima in the original data)
        inverted_gaps = -gaps[:, i]
        peaks, properties = find_peaks(inverted_gaps, prominence=0.01)
        
        local_minima = []
        for peak_idx in peaks:
            local_minima.append({
                'theta': theta_vals[peak_idx],
                'gap': gaps[peak_idx, i],
                'idx': peak_idx
            })
        
        # Sort local minima by gap value (ascending)
        local_minima.sort(key=lambda x: x['gap'])
        
        # Create the gap info dictionary
        gap_info = {
            'min_gap': min_gap,
            'min_gap_theta': min_gap_theta,
            'min_gap_idx': min_gap_idx,
            'local_minima': local_minima
        }
        
        # Add state pair information if available
        if gap_indices is not None:
            gap_info['states'] = gap_indices[i]
        
        min_gap_info[i] = gap_info
    
    return min_gap_info

def calculate_gap_statistics(gaps):
    """
    Calculate statistics for the energy gaps.
    
    Parameters:
    gaps (numpy.ndarray): Array of energy gaps
    
    Returns:
    dict: Dictionary containing statistics for each gap
    """
    num_gaps = gaps.shape[1]
    gap_stats = {}
    
    for i in range(num_gaps):
        gap_stats[i] = {
            'mean': np.mean(gaps[:, i]),
            'std': np.std(gaps[:, i]),
            'min': np.min(gaps[:, i]),
            'max': np.max(gaps[:, i]),
            'range': np.max(gaps[:, i]) - np.min(gaps[:, i]),
            'median': np.median(gaps[:, i])
        }
    
    return gap_stats

def plot_energy_spectrum(theta_vals, eigenvalues, output_dir=None):
    """
    Plot the energy spectrum as a function of theta.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values
    eigenvalues (numpy.ndarray): Array of eigenvalues
    output_dir (str, optional): Directory to save the plot
    
    Returns:
    None
    """
    # Create a figure with two subplots - one for full spectrum and one zoomed in
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    num_states = eigenvalues.shape[1]
    
    # Full spectrum plot
    for i in range(num_states):
        ax1.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')
    
    ax1.set_xlabel('Theta (θ)')
    ax1.set_ylabel('Energy')
    ax1.set_title('Full Energy Spectrum vs Theta')
    ax1.grid(True)
    ax1.legend()
    
    # Zoomed in plot focusing on the lower energy states
    for i in range(min(3, num_states)):  # Only plot the first 3 states for clarity
        ax2.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')
    
    # Set y-limits to focus on the small gaps
    y_min = np.min(eigenvalues[:, :3]) - 0.001
    y_max = np.min(eigenvalues[:, :3]) + 0.005  # Small range to see the gaps
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_xlabel('Theta (θ)')
    ax2.set_ylabel('Energy')
    ax2.set_title('Zoomed View of Lower Energy States')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'energy_spectrum.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_energy_gaps(theta_vals, gaps, min_gap_info, output_dir=None):
    """
    Plot the energy gaps as a function of theta.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values
    gaps (numpy.ndarray): Array of energy gaps
    min_gap_info (dict): Dictionary containing minimum gap information
    output_dir (str, optional): Directory to save the plot
    
    Returns:
    None
    """
    # Create a figure with three subplots - full view and two zoomed views
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    num_gaps = gaps.shape[1]
    
    # Full view of all gaps
    for i in range(num_gaps):
        ax1.plot(theta_vals, gaps[:, i], label=f'Gap {i}-{i+1}')
        
        # Mark the minimum gap
        min_gap = min_gap_info[i]['min_gap']
        min_gap_theta = min_gap_info[i]['min_gap_theta']
        ax1.scatter(min_gap_theta, min_gap, color='red', s=50, zorder=5)
        ax1.annotate(f'Min: {min_gap:.4f}', 
                    (min_gap_theta, min_gap),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    ax1.set_xlabel('Theta (θ)')
    ax1.set_ylabel('Energy Gap')
    ax1.set_title('All Energy Gaps vs Theta')
    ax1.grid(True)
    ax1.legend()
    
    # Zoomed view of gap 0-1
    ax2.plot(theta_vals, gaps[:, 0], label=f'Gap 0-1', color='blue')
    min_gap = min_gap_info[0]['min_gap']
    min_gap_theta = min_gap_info[0]['min_gap_theta']
    ax2.scatter(min_gap_theta, min_gap, color='red', s=50, zorder=5)
    ax2.annotate(f'Min: {min_gap:.6f}', 
                (min_gap_theta, min_gap),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center')
    
    # Set y-limits for better visibility of small gaps
    max_gap_0_1 = np.max(gaps[:, 0])
    ax2.set_ylim(0, max_gap_0_1 * 1.2)
    
    ax2.set_xlabel('Theta (θ)')
    ax2.set_ylabel('Energy Gap')
    ax2.set_title('Zoomed View of Gap 0-1')
    ax2.grid(True)
    ax2.legend()
    
    # Zoomed view of gap 1-2
    ax3.plot(theta_vals, gaps[:, 1], label=f'Gap 1-2', color='green')
    min_gap = min_gap_info[1]['min_gap']
    min_gap_theta = min_gap_info[1]['min_gap_theta']
    ax3.scatter(min_gap_theta, min_gap, color='red', s=50, zorder=5)
    ax3.annotate(f'Min: {min_gap:.6f}', 
                (min_gap_theta, min_gap),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center')
    
    # Set y-limits for better visibility of small gaps
    max_gap_1_2 = np.max(gaps[:, 1])
    if max_gap_1_2 > 0:
        ax3.set_ylim(0, max_gap_1_2 * 1.2)
    else:
        # If all values are zero, set a small range
        ax3.set_ylim(0, 0.001)
    
    ax3.set_xlabel('Theta (θ)')
    ax3.set_ylabel('Energy Gap')
    ax3.set_title('Zoomed View of Gap 1-2')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'energy_gaps.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_comprehensive_analysis(theta_vals, eigenvalues, gaps, min_gap_info, berry_phases, output_dir=None):
    """
    Create a comprehensive plot with energy spectrum, gaps, and Berry phases.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values
    eigenvalues (numpy.ndarray): Array of eigenvalues
    gaps (numpy.ndarray): Array of energy gaps
    min_gap_info (dict): Dictionary containing minimum gap information
    berry_phases (numpy.ndarray): Array of Berry phases
    output_dir (str, optional): Directory to save the plot
    
    Returns:
    None
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Energy spectrum plot
    ax1 = fig.add_subplot(gs[0, :])
    num_states = eigenvalues.shape[1]
    for i in range(num_states):
        ax1.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')
    
    ax1.set_xlabel('Theta (θ)')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Spectrum vs Theta')
    ax1.grid(True)
    ax1.legend()
    
    # Energy gaps plot
    ax2 = fig.add_subplot(gs[1, 0])
    num_gaps = gaps.shape[1]
    for i in range(num_gaps):
        ax2.plot(theta_vals, gaps[:, i], label=f'Gap {i}-{i+1}')
        
        # Mark the minimum gap
        min_gap = min_gap_info[i]['min_gap']
        min_gap_theta = min_gap_info[i]['min_gap_theta']
        ax2.scatter(min_gap_theta, min_gap, color='red', s=50, zorder=5)
        ax2.annotate(f'Min: {min_gap:.4f}', 
                    (min_gap_theta, min_gap),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    ax2.set_xlabel('Theta (θ)')
    ax2.set_ylabel('Energy Gap')
    ax2.set_title('Energy Gaps vs Theta')
    ax2.grid(True)
    ax2.legend()
    
    # Berry phases plot
    ax3 = fig.add_subplot(gs[1, 1])
    num_berry_phases = len(berry_phases)
    for i in range(num_berry_phases):
        # Handle both 1D and 2D berry_phases arrays
        if berry_phases.ndim == 1:
            # Single berry phase value per state
            ax3.axhline(y=berry_phases[i], linestyle='--', label=f'State {i}')
        else:
            # Berry phases has shape (num_states, len(theta_vals))
            ax3.plot(theta_vals, np.cumsum(berry_phases[i, :]), label=f'State {i}')
    
    ax3.set_xlabel('Theta (θ)')
    ax3.set_ylabel('Cumulative Berry Phase')
    ax3.set_title('Berry Phase Accumulation vs Theta')
    ax3.grid(True)
    ax3.legend()
    
    # Gap statistics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Gap', 'Min', 'Max', 'Mean', 'Std Dev', 'Min Theta']
    
    for i in range(num_gaps):
        min_gap = min_gap_info[i]['min_gap']
        min_gap_theta = min_gap_info[i]['min_gap_theta']
        mean_gap = np.mean(gaps[:, i])
        std_gap = np.std(gaps[:, i])
        max_gap = np.max(gaps[:, i])
        
        table_data.append([
            f'{i}-{i+1}',
            f'{min_gap:.6f}',
            f'{max_gap:.6f}',
            f'{mean_gap:.6f}',
            f'{std_gap:.6f}',
            f'{min_gap_theta:.4f}'
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_gap_analysis_report(min_gap_info, gap_stats, output_dir=None):
    """
    Create a detailed report on energy gap analysis.
    
    Parameters:
    min_gap_info (dict): Dictionary containing minimum gap information
    gap_stats (dict): Dictionary containing gap statistics
    output_dir (str, optional): Directory to save the report
    
    Returns:
    None
    """
    report = []
    report.append("Energy Gap Analysis Report")
    report.append("==========================\n")
    
    # Add gap statistics
    report.append("Gap Statistics:")
    report.append("--------------")
    for gap_idx, stats in gap_stats.items():
        # Check if we have state information
        if 'states' in min_gap_info[gap_idx]:
            state_i, state_j = min_gap_info[gap_idx]['states']
            report.append(f"Gap {state_i}-{state_j}:")
        else:
            report.append(f"Gap {gap_idx}-{gap_idx+1}:")
            
        report.append(f"  Minimum: {stats['min']:.6f}")
        report.append(f"  Maximum: {stats['max']:.6f}")
        report.append(f"  Mean: {stats['mean']:.6f}")
        report.append(f"  Standard Deviation: {stats['std']:.6f}")
        report.append(f"  Range: {stats['range']:.6f}")
        report.append(f"  Median: {stats['median']:.6f}")
        report.append("")
    
    # Add minimum gap information
    report.append("Minimum Gap Analysis:")
    report.append("--------------------")
    for gap_idx, info in min_gap_info.items():
        # Check if we have state information
        if 'states' in info:
            state_i, state_j = info['states']
            report.append(f"Gap {state_i}-{state_j}:")
        else:
            report.append(f"Gap {gap_idx}-{gap_idx+1}:")
            
        report.append(f"  Global Minimum: {info['min_gap']:.6f} at θ = {info['min_gap_theta']:.4f}")
        
        if info['local_minima']:
            report.append("  Local Minima (sorted by gap value):")
            for i, lm in enumerate(info['local_minima'][:5]):  # Show top 5 local minima
                report.append(f"    {i+1}. Gap: {lm['gap']:.6f} at θ = {lm['theta']:.4f}")
        report.append("")
    
    # Write report to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'gap_analysis_report.txt'), 'w') as f:
            f.write('\n'.join(report))
    else:
        print('\n'.join(report))

def main():
    """Main function to run the energy gap analysis."""
    parser = argparse.ArgumentParser(description='Energy Gap Analysis for Berry Phase System')
    
    # Add parameters as arguments
    parser.add_argument('--c', type=float, default=0.2, help='Fixed coupling constant')
    parser.add_argument('--omega', type=float, default=1.0, help='Frequency parameter')
    parser.add_argument('--a', type=float, default=1.0, help='First coefficient for potentials')
    parser.add_argument('--b', type=float, default=0.5, help='Second coefficient for potentials')
    parser.add_argument('--c_const', type=float, default=0.0, help='Constant term in potentials')
    parser.add_argument('--x_shift', type=float, default=0.2, help='Shift for Va potential on x-axis')
    parser.add_argument('--y_shift', type=float, default=0.2, help='Shift for Va potential on y-axis')
    parser.add_argument('--d', type=float, default=1.0, help='Parameter for R_theta')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points for theta')
    parser.add_argument('--output_dir', type=str, default='energy_gap_analysis', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create parameter dictionary
    params = {
        'c': args.c,
        'omega': args.omega,
        'a': args.a,
        'b': args.b,
        'c_const': args.c_const,
        'x_shift': args.x_shift,
        'y_shift': args.y_shift,
        'd': args.d,
        'num_points': args.num_points
    }
    
    print("Running energy gap analysis with parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate energy spectrum
    print("\nCalculating energy spectrum...")
    theta_vals, eigenvalues, eigenstates, r_theta_vectors, Vx_values, Va_values = calculate_energy_spectrum(params)
    
    # Calculate energy gaps - both adjacent and all pairs
    print("Calculating energy gaps...")
    adjacent_gaps, adjacent_gap_indices = calculate_energy_gaps(eigenvalues, adjacent_only=True)
    all_gaps, all_gap_indices = calculate_energy_gaps(eigenvalues, adjacent_only=False)
    
    # Analyze minimum gaps
    print("Analyzing minimum gaps...")
    adjacent_min_gap_info = analyze_minimum_gaps(theta_vals, adjacent_gaps, adjacent_gap_indices)
    all_min_gap_info = analyze_minimum_gaps(theta_vals, all_gaps, all_gap_indices)
    
    # Calculate gap statistics
    print("Calculating gap statistics...")
    adjacent_gap_stats = calculate_gap_statistics(adjacent_gaps)
    all_gap_stats = calculate_gap_statistics(all_gaps)
    
    # Calculate Berry connection and phase
    print("Calculating Berry connection and phase...")
    A = berry_connection_analytical(theta_vals, params['c'])
    
    # Calculate the Berry phase using the existing function
    berry_phases = berry_phase_integration(A, theta_vals)
    
    # Create plots
    print("Creating plots...")
    plot_energy_spectrum(theta_vals, eigenvalues, output_dir)
    
    # Plot adjacent gaps
    plot_energy_gaps(theta_vals, adjacent_gaps, adjacent_min_gap_info, 
                     os.path.join(output_dir, 'adjacent_gaps'))
    
    # Plot comprehensive analysis with adjacent gaps
    plot_comprehensive_analysis(theta_vals, eigenvalues, adjacent_gaps, 
                               adjacent_min_gap_info, berry_phases, 
                               os.path.join(output_dir, 'adjacent_gaps'))
    
    # Plot all gaps (including non-adjacent)
    plot_energy_gaps(theta_vals, all_gaps, all_min_gap_info, 
                     os.path.join(output_dir, 'all_gaps'))
    
    # Create reports
    print("Creating gap analysis reports...")
    # Report for adjacent gaps
    create_gap_analysis_report(adjacent_min_gap_info, adjacent_gap_stats, 
                              os.path.join(output_dir, 'adjacent_gaps'))
    
    # Report for all gaps
    create_gap_analysis_report(all_min_gap_info, all_gap_stats, 
                              os.path.join(output_dir, 'all_gaps'))
    
    print(f"\nEnergy gap analysis completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
