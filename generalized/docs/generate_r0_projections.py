#!/usr/bin/env python3
"""
Script to generate R_0 plane projections for the combined effect figures.
This script creates 2x2 subplot figures showing the projections on the R_0 plane.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the vector generation and visualization functions
from vector_utils import create_orthogonal_vectors
from visualization import plot_vectors_2d_projection

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(output_dir, exist_ok=True)

def plot_r0_projection_subplot(ax, R_0, d, theta, title):
    """Plot R_0 plane projection in a subplot"""
    # Generate vectors
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, d, theta)
    
    # Clear the axis
    ax.clear()
    
    # Convert to numpy arrays if they aren't already
    R_0 = np.array(R_0)
    R_1 = np.array(R_1)
    R_2 = np.array(R_2)
    R_3 = np.array(R_3)
    
    # For R_0 plane projection
    if np.allclose(R_0, np.zeros(3)):
        # If R_0 is the origin, we can use any plane passing through the origin
        # Use the plane defined by R_1 and R_2
        basis1 = R_1 - R_0
        basis1 = basis1 / np.linalg.norm(basis1)
        
        basis2 = R_2 - R_0
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Make sure basis2 is orthogonal to basis1
        basis2 = basis2 - np.dot(basis2, basis1) * basis1
        basis2 = basis2 / np.linalg.norm(basis2)
    else:
        # Define the normal to the plane as the vector from origin to R_0
        normal = R_0 / np.linalg.norm(R_0)
        
        # Find two orthogonal vectors in the plane
        # First basis vector: cross product of normal with [1,0,0] or [0,1,0]
        if not np.allclose(normal, np.array([1, 0, 0])):
            basis1 = np.cross(normal, np.array([1, 0, 0]))
        else:
            basis1 = np.cross(normal, np.array([0, 1, 0]))
        basis1 = basis1 / np.linalg.norm(basis1)
        
        # Second basis vector: cross product of normal with basis1
        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
    
    # Project vectors onto the plane defined by basis1 and basis2
    R0_proj = np.array([0, 0])  # Origin in the plane
    
    # Project R_1, R_2, R_3 onto the plane
    v1 = R_1 - R_0
    v2 = R_2 - R_0
    v3 = R_3 - R_0
    
    R1_proj = np.array([np.dot(v1, basis1), np.dot(v1, basis2)])
    R2_proj = np.array([np.dot(v2, basis1), np.dot(v2, basis2)])
    R3_proj = np.array([np.dot(v3, basis1), np.dot(v3, basis2)])
    
    # Plot the origin
    ax.scatter(R0_proj[0], R0_proj[1], color='black', s=50, label='R_0')
    
    # Plot the vectors as arrows
    vectors_proj = [R1_proj, R2_proj, R3_proj]
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for vector, color, label in zip(vectors_proj, colors, labels):
        ax.arrow(R0_proj[0], R0_proj[1], 
                vector[0], vector[1], 
                head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Set labels and title
    ax.set_xlabel('Basis 1')
    ax.set_ylabel('Basis 2')
    ax.set_title(title)
    
    # Calculate appropriate axis limits
    all_coords = np.vstack([R1_proj, R2_proj, R3_proj])
    max_abs_coord = np.max(np.abs(all_coords)) * 1.2  # Add 20% margin
    
    # Set symmetric axis limits
    ax.set_xlim(-max_abs_coord, max_abs_coord)
    ax.set_ylim(-max_abs_coord, max_abs_coord)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True)
    
    # Add legend
    ax.legend(loc='best')

def generate_r0_projection_figure(R_0, d_values, theta_values, filename):
    """Generate a figure showing the R_0 plane projections for different parameter combinations"""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 grid of subplots
    gs = GridSpec(2, 2, figure=fig)
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Plot combinations of d and theta values
    for i, (d, theta) in enumerate(zip(d_values, theta_values)):
        if i < len(axes):
            plot_r0_projection_subplot(axes[i], R_0, d, theta, f'd = {d}, θ = {theta:.2f}')
    
    # Add a main title
    fig.suptitle(f'R_0 Plane Projections with R_0={R_0}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    """Main function to generate R_0 plane projection figures"""
    # Define parameter values
    R_0_values = [(0, 0, 0), (1, 1, 1), (0, 0, 2)]
    combined_d_values = [0.5, 1.0, 1.5, 2.0]
    combined_theta_values = [0, np.pi/6, np.pi/4, np.pi/3]
    
    # Generate figures for each R_0 value
    for R_0 in R_0_values:
        R_0_str = '_'.join(map(str, R_0)).replace('.', 'p')
        
        # Generate R_0 plane projection figure
        generate_r0_projection_figure(
            R_0,
            combined_d_values,
            combined_theta_values,
            os.path.join(output_dir, f'r0_projections_R0_{R_0_str}.png')
        )

if __name__ == "__main__":
    main()
