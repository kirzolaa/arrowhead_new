#!/usr/bin/env python3
"""
Script to generate combined 3D and R_0 plane projection figures for the documentation.
This script creates figures showing both 3D views and R_0 plane projections for each configuration.
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
from visualization import plot_vectors_3d, plot_vectors_2d_projection

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(output_dir, exist_ok=True)

def generate_combined_view_figure(R_0, d, theta, filename):
    """Generate a figure showing both 3D view and R_0 plane projection"""
    # Generate vectors
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, d, theta)
    
    # Convert to numpy arrays if they aren't already
    R_0 = np.array(R_0)
    R_1 = np.array(R_1)
    R_2 = np.array(R_2)
    R_3 = np.array(R_3)
    
    # Create figure with 1 row and 2 columns
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    # 3D plot on the left
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Plot the vectors in 3D
    ax_3d.scatter(*R_0, color='black', s=50, label='R_0')
    
    # Plot the vectors as arrows
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for i, R in enumerate([R_1, R_2, R_3]):
        ax_3d.quiver(*R_0, *(R - R_0), color=colors[i], label=labels[i])
    
    # Set labels and title
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(f'3D View (R_0={R_0}, d={d}, θ={theta:.2f})')
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max([R_1[i], R_2[i], R_3[i]]) - np.min([R_1[i], R_2[i], R_3[i]]) 
        for i in range(3)
    ])
    mid_x = np.mean([R_0[0], R_1[0], R_2[0], R_3[0]])
    mid_y = np.mean([R_0[1], R_1[1], R_2[1], R_3[1]])
    mid_z = np.mean([R_0[2], R_1[2], R_2[2], R_3[2]])
    ax_3d.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax_3d.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax_3d.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Add legend
    ax_3d.legend()
    
    # R_0 plane projection on the right
    ax_r0 = fig.add_subplot(gs[0, 1])
    
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
    ax_r0.scatter(R0_proj[0], R0_proj[1], color='black', s=50, label='R_0')
    
    # Plot the vectors as arrows
    vectors_proj = [R1_proj, R2_proj, R3_proj]
    
    for vector, color, label in zip(vectors_proj, colors, labels):
        ax_r0.arrow(R0_proj[0], R0_proj[1], 
                vector[0], vector[1], 
                head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Set labels and title
    ax_r0.set_xlabel('Basis 1')
    ax_r0.set_ylabel('Basis 2')
    ax_r0.set_title(f'R_0 Plane Projection (d={d}, θ={theta:.2f})')
    
    # Calculate appropriate axis limits
    all_coords = np.vstack([R1_proj, R2_proj, R3_proj])
    max_abs_coord = np.max(np.abs(all_coords)) * 1.2  # Add 20% margin
    
    # Set symmetric axis limits
    ax_r0.set_xlim(-max_abs_coord, max_abs_coord)
    ax_r0.set_ylim(-max_abs_coord, max_abs_coord)
    
    # Set equal aspect ratio
    ax_r0.set_aspect('equal')
    
    # Add grid
    ax_r0.grid(True)
    
    # Add legend
    ax_r0.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    """Main function to generate combined view figures"""
    # Define configurations
    configurations = [
        ((0, 0, 0), 0.5, 0),         # R_0 = (0,0,0), d = 0.5, θ = 0
        ((0, 0, 0), 1.0, np.pi/6),   # R_0 = (0,0,0), d = 1.0, θ = π/6
        ((0, 0, 0), 1.5, np.pi/4),   # R_0 = (0,0,0), d = 1.5, θ = π/4
        ((0, 0, 0), 2.0, np.pi/3),   # R_0 = (0,0,0), d = 2.0, θ = π/3
        
        ((1, 1, 1), 0.5, 0),         # R_0 = (1,1,1), d = 0.5, θ = 0
        ((1, 1, 1), 1.0, np.pi/6),   # R_0 = (1,1,1), d = 1.0, θ = π/6
        ((1, 1, 1), 1.5, np.pi/4),   # R_0 = (1,1,1), d = 1.5, θ = π/4
        ((1, 1, 1), 2.0, np.pi/3),   # R_0 = (1,1,1), d = 2.0, θ = π/3
        
        ((0, 0, 2), 0.5, 0),         # R_0 = (0,0,2), d = 0.5, θ = 0
        ((0, 0, 2), 1.0, np.pi/6),   # R_0 = (0,0,2), d = 1.0, θ = π/6
        ((0, 0, 2), 1.5, np.pi/4),   # R_0 = (0,0,2), d = 1.5, θ = π/4
        ((0, 0, 2), 2.0, np.pi/3),   # R_0 = (0,0,2), d = 2.0, θ = π/3
    ]
    
    # Generate figures for each configuration
    for R_0, d, theta in configurations:
        R_0_str = '_'.join(map(str, R_0)).replace('.', 'p')
        d_str = str(d).replace('.', 'p')
        theta_str = f"{theta:.2f}".replace('.', 'p')
        
        # Generate combined view figure
        generate_combined_view_figure(
            R_0, d, theta,
            os.path.join(output_dir, f'combined_view_R0_{R_0_str}_d_{d_str}_theta_{theta_str}.png')
        )

if __name__ == "__main__":
    main()
