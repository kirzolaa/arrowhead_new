#!/usr/bin/env python3
"""
Improved script to plot eigenvector endpoints for arrowhead matrices.
This script loads the saved eigenvalues and eigenvectors and creates
improved visualizations showing 4 endpoints per matrix without connecting them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eigenvector_endpoints(theta_values, all_eigenvectors, output_dir="./results"):
    """
    Plot the eigenvector endpoints for all theta values in 3D space.
    Shows all 4 eigenvector endpoints for each matrix without connecting them.
    
    Parameters:
    -----------
    theta_values : list of float
        List of theta values in degrees
    all_eigenvectors : list of numpy.ndarray
        List of eigenvectors for each theta
    output_dir : str
        Directory to save the plots
    """
    # Create a figure for all eigenvector endpoints
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different theta values
    cmap = plt.cm.viridis
    theta_colors = [cmap(i/len(theta_values)) for i in range(len(theta_values))]
    
    # Markers for different eigenvectors
    markers = ['o', 's', '^', 'd']  # circle, square, triangle, diamond
    
    # For each theta value
    for i, (theta, eigenvectors) in enumerate(zip(theta_values, all_eigenvectors)):
        # For each eigenvector (0-3) of this matrix
        for ev_idx in range(4):
            # Get the eigenvector
            eigenvector = eigenvectors[:, ev_idx]
            
            # Plot the eigenvector endpoint
            ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                      c=[theta_colors[i]], marker=markers[ev_idx], s=100, 
                      alpha=0.8)
            
            # Add text label with theta value and eigenvector index
            ax.text(eigenvector[1], eigenvector[2], eigenvector[3], 
                   f'{theta:.1f}°,ev{ev_idx}', size=8)
    
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
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "eigenvector_endpoints.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to load saved eigenvalues and eigenvectors and create improved plots.
    """
    # Parameters
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # Generate theta values (0-360 in 5-degree steps)
    theta_values = np.arange(0, 360, 5)
    
    # Load eigenvectors for each theta
    all_eigenvectors = []
    for i, theta in enumerate(theta_values):
        # Load eigenvectors from the existing results
        # Note: We're assuming the files are named according to the original script's convention
        try:
            eigenvectors = np.load(os.path.join(results_dir, f"eigenvectors_theta_{i}.npy"))
            all_eigenvectors.append(eigenvectors)
        except FileNotFoundError:
            print(f"Warning: Could not find eigenvectors file for theta_{i}. Skipping.")
    
    # If we found any eigenvectors, create the plot
    if all_eigenvectors:
        # Use only the theta values for which we have eigenvectors
        theta_values = theta_values[:len(all_eigenvectors)]
        
        print(f"Creating improved eigenvector plot for {len(all_eigenvectors)} theta values...")
        plot_eigenvector_endpoints(theta_values, all_eigenvectors, results_dir)
        print(f"Plot saved to {os.path.join(results_dir, 'eigenvector_endpoints.png')}")
    else:
        print("No eigenvector files found. Please run the original script first.")

if __name__ == "__main__":
    main()
