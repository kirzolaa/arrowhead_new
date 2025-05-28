#!/usr/bin/env python3
"""
Simple visualization of a circle orthogonal to the x=y=z line,
matching the reference images with a clear red circle and proper projections.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
    """
    Create a single R vector that forms a perfect circle orthogonal to the x=y=z line.
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Define the basis vectors
    # basis1 is along the x=y=z line (normalized)
    basis1_raw = np.array([1, 1, 1])
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    
    # basis2 and basis3 are orthogonal to basis1 and to each other
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize basis2 and basis3
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Calculate the point on the circle
    circle_point = d * (np.cos(theta) * basis2 + np.sin(theta) * basis3)
    
    # Add the R_0 offset
    R = R_0 + circle_point
    
    return R

def create_perfect_orthogonal_circle(R_0=(0, 0, 0), d=1, num_points=36):
    """
    Create multiple vectors that form a perfect circle orthogonal to the x=y=z line.
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Generate equally spaced angles between 0 and 2π
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=True)
    
    # Initialize the array to store the vectors
    vectors = np.zeros((num_points, 3))
    
    # Generate vectors for each angle
    for i, theta in enumerate(thetas):
        vectors[i] = create_perfect_orthogonal_vectors(R_0, d, theta)
    
    return vectors

def visualize_circle(R_0=(0, 0, 0), d=0.001, output_dir="circle_visualization", scale_factor=60):
    """
    Create a visualization that matches the reference images with a clear red circle.
    
    Parameters:
    R_0 (tuple): Center point coordinates
    d (float): Actual radius of the circle (0.001 for CI points)
    output_dir (str): Directory to save output files
    scale_factor (float): Visual scaling factor to make the small circle visible
    """
    """
    Create a visualization that matches the reference images with a clear red circle.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the circle points with scaled radius for visibility
    # while keeping the actual radius value in the title
    num_points = 100
    display_d = d * scale_factor  # Scale for display purposes only
    circle_points = create_perfect_orthogonal_circle(R_0, display_d, num_points)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot R_0 as the origin
    ax.scatter([R_0[0]], [R_0[1]], [R_0[2]], color='black', s=100, label='Origin (R_0)')
    
    # Plot the circle with a thick red line
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9)
    
    # Plot the x=y=z line
    max_val = max(np.max(np.abs(circle_points)), np.max(np.abs(R_0))) * 1.5
    line_points = np.array([-max_val, max_val])
    ax.plot(line_points, line_points, line_points, 'k-', alpha=0.8, label='x=y=z line')
    
    # Plot the coordinate axes with colored lines
    ax.plot([-max_val, max_val], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.8)
    ax.plot([0, 0], [-max_val, max_val], [0, 0], 'g-', linewidth=2, alpha=0.8)
    ax.plot([0, 0], [0, 0], [-max_val, max_val], 'b-', linewidth=2, alpha=0.8)
    
    # Add axis labels
    ax.text(max_val, 0, 0, 'X', color='red', fontsize=12)
    ax.text(0, max_val, 0, 'Y', color='green', fontsize=12)
    ax.text(0, 0, max_val, 'Z', color='blue', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Use the actual d value in the title, not the scaled one
    R_0_rounded = tuple(round(float(val), 3) for val in R_0)
    title = f'Perfect Circle Orthogonal to x=y=z line (R_0 = {R_0_rounded}, d={d})'
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend at the top of the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    # Save the figure
    plt.savefig(f'{output_dir}/circle_3d.png', dpi=300, bbox_inches='tight')
    
    # Create projections onto the coordinate planes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY projection
    axs[0, 0].plot(circle_points[:, 0], circle_points[:, 1], 'r-', linewidth=4, alpha=0.9)
    axs[0, 0].scatter(R_0[0], R_0[1], color='black', s=100)
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].grid(True)
    
    # XZ projection
    axs[0, 1].plot(circle_points[:, 0], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9)
    axs[0, 1].scatter(R_0[0], R_0[2], color='black', s=100)
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].grid(True)
    
    # YZ projection
    axs[1, 0].plot(circle_points[:, 1], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9)
    axs[1, 0].scatter(R_0[1], R_0[2], color='black', s=100)
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('Z')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].grid(True)
    
    # Projection onto the plane orthogonal to x=y=z
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Project the circle points onto the basis2-basis3 plane
    circle_basis = np.zeros((len(circle_points), 2))
    for i, point in enumerate(circle_points):
        vec = point - R_0
        circle_basis[i, 0] = np.dot(vec, basis2)
        circle_basis[i, 1] = np.dot(vec, basis3)
    
    # Plot the projection
    axs[1, 1].plot(circle_basis[:, 0], circle_basis[:, 1], 'r-', linewidth=4, alpha=0.9)
    axs[1, 1].scatter(0, 0, color='black', s=100)
    axs[1, 1].set_xlabel('Basis Vector 1 Direction')
    axs[1, 1].set_ylabel('Basis Vector 2 Direction')
    axs[1, 1].set_title('Projection onto Plane Orthogonal to x=y=z')
    axs[1, 1].grid(True)
    
    # Make all plots square
    for ax in axs.flat:
        ax.set_aspect('equal')
    
    # Add individual legends to each subplot to avoid overlapping
    for i, ax in enumerate(axs.flat):
        if i == 3:  # For the orthogonal projection, add a more detailed legend
            ax.legend(['Circle Points', 'Origin (R_0)'], loc='upper right', fontsize=8)
    
    # Adjust layout without reserving space for a shared legend
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/circle_projections.png', dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")
    return circle_points

if __name__ == "__main__":
    # For R_0 at the origin (0,0,0)
    visualize_circle(R_0=(0, 0, 0), d=0.001, output_dir="berry_phase_corrected_run_n_minus_1/vectors/origin_circle")
    
    # For R_0 at (0.433, 0.433, 0.433)
    visualize_circle(R_0=(0.433, 0.433, 0.433), d=0.001, output_dir="berry_phase_corrected_run_n_minus_1/vectors/r0_circle")
