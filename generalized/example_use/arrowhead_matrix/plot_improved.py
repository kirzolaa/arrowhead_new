#!/usr/bin/env python3
"""
Script to create improved plots for eigenvalues and eigenvectors:
1. 2D plots of eigenvalue vs. theta for each eigenvalue
2. 3D plots of eigenvectors without text labels
3. No connecting lines between points in any plots
4. Organizes output files into appropriate subdirectories
"""

import os
import sys
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def organize_results_directory(results_dir):
    """
    Organize the results directory by file type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    
    Returns:
    --------
    dict
        Dictionary mapping file extensions to subdirectories
    """
    # Create subdirectories if they don't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Move files to appropriate subdirectories
    for file_path in glob.glob(os.path.join(results_dir, '*')):
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip('.')
        
        # Skip if extension not in our list
        if ext not in subdirs:
            continue
        
        # Get destination directory
        dest_dir = subdirs[ext]
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Move file to destination directory
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"Moved {filename} to {os.path.relpath(dest_path, results_dir)}")
    
    return subdirs

def get_file_path(results_dir, filename, file_type=None):
    """
    Get the appropriate path for a file based on its type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    filename : str
        Name of the file
    file_type : str, optional
        Type of the file (extension without the dot)
        If None, will be determined from the filename
    
    Returns:
    --------
    str
        Path to the file
    """
    # Create the directory structure if it doesn't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Determine file type if not provided
    if file_type is None:
        _, ext = os.path.splitext(filename)
        file_type = ext.lstrip('.')
    
    # Get the appropriate directory
    if file_type in subdirs:
        return os.path.join(subdirs[file_type], filename)
    else:
        return os.path.join(results_dir, filename)

def plot_eigenvalues_2d(theta_values, all_eigenvalues, output_dir="./results"):
    """
    Create 2D plots of eigenvalue vs. theta for each eigenvalue.
    
    Parameters:
    -----------
    theta_values : list of float
        List of theta values in radians
    all_eigenvalues : list of numpy.ndarray
        List of eigenvalues for each theta
    output_dir : str
        Directory to save the plots
    """
    # Convert theta values to degrees for display
    theta_values_deg = np.degrees(theta_values)
    
    # Colors for different eigenvalues
    colors = ['r', 'g', 'b', 'purple']
    
    # Create a separate 2D plot for each eigenvalue
    for ev_idx in range(4):
        plt.figure(figsize=(10, 6))
        
        # Extract the eigenvalues for this index
        ev_values = [eigenvalues[ev_idx] for eigenvalues in all_eigenvalues]
        
        # Plot eigenvalue vs. theta
        plt.scatter(theta_values_deg, ev_values, c=colors[ev_idx], s=50)
        
        # Set labels and title
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Eigenvalue')
        plt.title(f'Eigenvalue {ev_idx} vs. Theta')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save the plot to the plots subdirectory
        plot_path = get_file_path(output_dir, f"eigenvalue_{ev_idx}_2d.png", "png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined 2D plot with all eigenvalues
    plt.figure(figsize=(12, 8))
    
    # Plot each eigenvalue series
    for ev_idx in range(4):
        ev_values = [eigenvalues[ev_idx] for eigenvalues in all_eigenvalues]
        plt.scatter(theta_values_deg, ev_values, c=colors[ev_idx], s=50, label=f'Eigenvalue {ev_idx}')
    
    # Set labels and title
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Eigenvalue')
    plt.title('All Eigenvalues vs. Theta')
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot to the plots subdirectory
    plot_path = get_file_path(output_dir, "all_eigenvalues_2d.png", "png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_eigenvectors_no_labels(theta_values, all_eigenvectors, output_dir="./results"):
    """
    Plot the eigenvector endpoints in 3D space without text labels.
    
    Parameters:
    -----------
    theta_values : list of float
        List of theta values in radians
    all_eigenvectors : list of numpy.ndarray
        List of eigenvectors for each theta
    output_dir : str
        Directory to save the plots
    """
    # Create a figure for all eigenvector endpoints
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert theta values to degrees for display
    theta_values_deg = np.degrees(theta_values)
    
    # Colors for different theta values
    cmap = plt.cm.viridis
    theta_colors = [cmap(i/len(theta_values)) for i in range(len(theta_values))]
    
    # Markers for different eigenvectors
    markers = ['o', 's', '^', 'd']  # circle, square, triangle, diamond
    
    # For each theta value
    for i, (theta, eigenvectors) in enumerate(zip(theta_values_deg, all_eigenvectors)):
        # For each eigenvector (0-3) of this matrix
        for ev_idx in range(4):
            # Get the eigenvector
            eigenvector = eigenvectors[:, ev_idx]
            
            # Plot the eigenvector endpoint (no text label)
            ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                      c=[theta_colors[i]], marker=markers[ev_idx], s=100, 
                      alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X Component')
    ax.set_ylabel('Y Component')
    ax.set_zlabel('Z Component')
    ax.set_title('All Eigenvector Endpoints for Different Theta Values')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add a colorbar to show the theta values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Theta (degrees)')
    
    # Add a legend for the eigenvector markers
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='gray', 
                  markersize=10, label=f'Eigenvector {i}') for i in range(4)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot to the plots subdirectory
    plot_path = get_file_path(output_dir, "eigenvectors_no_labels.png", "png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each eigenvector (without labels)
    for ev_idx in range(4):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # For each theta value, plot the corresponding eigenvector endpoint
        for i, (theta, eigenvectors) in enumerate(zip(theta_values_deg, all_eigenvectors)):
            # Get the eigenvector for this eigenvalue index
            eigenvector = eigenvectors[:, ev_idx]
            
            # Plot the eigenvector endpoint (no text label)
            ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                       c=[theta_colors[i]], marker='o', s=100)
        
        # Set labels and title
        ax.set_xlabel('X Component')
        ax.set_ylabel('Y Component')
        ax.set_zlabel('Z Component')
        ax.set_title(f'Eigenvector {ev_idx} Endpoints for Different Theta Values')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add a colorbar to show the theta values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
        cbar.set_label('Theta (degrees)')
        
        # Save the plot to the plots subdirectory
        plot_path = get_file_path(output_dir, f"eigenvector_{ev_idx}_no_labels.png", "png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to load saved eigenvalues and eigenvectors and create improved plots.
    """
    # Parameters
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # Load all eigenvalues and eigenvectors
    all_eigenvalues = []
    all_eigenvectors = []
    theta_values = []
    
    # Organize the results directory
    print("Organizing results directory...")
    organize_results_directory(results_dir)
    
    # Find all eigenvalue files
    i = 0
    numpy_dir = os.path.join(results_dir, 'numpy')
    while True:
        eigenvalues_file = os.path.join(numpy_dir, f"eigenvalues_theta_{i}.npy")
        eigenvectors_file = os.path.join(numpy_dir, f"eigenvectors_theta_{i}.npy")
        
        if not os.path.exists(eigenvalues_file) or not os.path.exists(eigenvectors_file):
            break
            
        # Load the data
        eigenvalues = np.load(eigenvalues_file)
        eigenvectors = np.load(eigenvectors_file)
        
        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)
        
        i += 1
    
    if not all_eigenvalues:
        print("No eigenvalue/eigenvector files found. Please run generate_plots_no_connections.py first.")
        return
    
    # Generate theta values (0-360 in 5-degree steps)
    theta_values = np.radians(np.arange(0, 360, 5)[:len(all_eigenvalues)])
    
    print(f"Found {len(all_eigenvalues)} sets of eigenvalues and eigenvectors.")
    print("Creating improved plots...")
    
    # Create 2D plots for eigenvalues
    print("Creating 2D eigenvalue plots...")
    plot_eigenvalues_2d(theta_values, all_eigenvalues, results_dir)
    
    # Create 3D plots for eigenvectors without labels
    print("Creating eigenvector plots without labels...")
    plot_eigenvectors_no_labels(theta_values, all_eigenvectors, results_dir)
    
    plots_dir = os.path.join(results_dir, 'plots')
    print("Plots created successfully:")
    print(f"  - {os.path.join(plots_dir, 'all_eigenvalues_2d.png')}")
    for i in range(4):
        print(f"  - {os.path.join(plots_dir, f'eigenvalue_{i}_2d.png')}")
    print(f"  - {os.path.join(plots_dir, 'eigenvectors_no_labels.png')}")
    for i in range(4):
        print(f"  - {os.path.join(plots_dir, f'eigenvector_{i}_no_labels.png')}")

if __name__ == "__main__":
    main()
