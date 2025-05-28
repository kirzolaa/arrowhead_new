#!/usr/bin/env python3
"""
Improved visualization for perfect circles in the plane orthogonal to the x=y=z line.
This script properly handles R0 at any position, including (0.433, 0.433, 0.433).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.gridspec import GridSpec

def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
    """
    Create a single R vector that forms a perfect circle orthogonal to the x=y=z line
    using normalized basis vectors.
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    theta (float): The angle parameter in radians, default is 0
    
    Returns:
    numpy.ndarray: The resulting R vector orthogonal to the x=y=z line
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

def create_perfect_orthogonal_circle(R_0=(0, 0, 0), d=1, num_points=36, start_theta=0, end_theta=2*np.pi):
    """
    Create multiple vectors that form a perfect circle orthogonal to the x=y=z line
    using normalized basis vectors.
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    num_points (int): The number of points to generate, default is 36
    start_theta (float): Starting angle in radians, default is 0
    end_theta (float): Ending angle in radians, default is 2*pi
    
    Returns:
    numpy.ndarray: Array of shape (num_points, 3) containing the generated vectors
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Generate equally spaced angles between start_theta and end_theta
    thetas = np.linspace(start_theta, end_theta, num_points, endpoint=True)
    
    # Initialize the array to store the vectors
    vectors = np.zeros((num_points, 3))
    
    # Generate vectors for each angle
    for i, theta in enumerate(thetas):
        vectors[i] = create_perfect_orthogonal_vectors(R_0, d, theta)
    
    return vectors

def improved_visualize_perfect_orthogonal_circle(points, R_0, d, save_dir, title_suffix=""):
    """
    Improved visualization of the perfect circle in the plane orthogonal to the x=y=z line.
    This function properly handles R0 at any position.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    R_0 (tuple or numpy.ndarray): The origin vector
    d (float): The distance parameter
    save_dir (str): Directory to save the visualizations
    title_suffix (str): Optional suffix for the plot titles
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the R_0 point - truncate coordinates to 3 decimal places
    R_0_truncated = tuple(round(val, 3) for val in R_0)
    ax.scatter([R_0[0]], [R_0[1]], [R_0[2]], color='black', s=100, label=f'Origin (R_0)')
    
    # Connect the points to show the circle with a thicker, more visible red line
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    
    # Plot the x=y=z line
    max_val = max(np.max(np.abs(points)), np.max(np.abs(R_0))) * 1.5
    line_points = np.array([-max_val, max_val])
    ax.plot(line_points, line_points, line_points, 'k-', label='x=y=z line', alpha=0.8)
    
    # Plot the coordinate axes with colored lines
    # X-axis - red
    ax.plot([-max_val, max_val], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.8)
    # Y-axis - green
    ax.plot([0, 0], [-max_val, max_val], [0, 0], 'g-', linewidth=2, alpha=0.8)
    # Z-axis - blue
    ax.plot([0, 0], [0, 0], [-max_val, max_val], 'b-', linewidth=2, alpha=0.8)
    
    # Add axis labels at the ends
    ax.text(max_val, 0, 0, 'X', color='red', fontsize=12)
    ax.text(0, max_val, 0, 'Y', color='green', fontsize=12)
    ax.text(0, 0, max_val, 'Z', color='blue', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Perfect Circle Orthogonal to x=y=z line (R_0 = {tuple(R_0)}, d={d}) {title_suffix}')
    
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
    
    # Add legend with better positioning
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.savefig(f'{save_dir}/perfect_circle_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create projections onto the coordinate planes and basis2-basis3 plane
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # XY projection
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.plot(points[:, 0], points[:, 1], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    ax_xy.scatter(R_0[0], R_0[1], color='black', s=100, label='Origin (R_0)')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title('XY Projection')
    ax_xy.grid(True)
    # Legend will be added as a single legend for all subplots
    
    # XZ projection
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.plot(points[:, 0], points[:, 2], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    ax_xz.scatter(R_0[0], R_0[2], color='black', s=100, label='Origin (R_0)')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('XZ Projection')
    ax_xz.grid(True)
    # Legend will be added as a single legend for all subplots
    
    # YZ projection
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_yz.plot(points[:, 1], points[:, 2], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    ax_yz.scatter(R_0[1], R_0[2], color='black', s=100, label='Origin (R_0)')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_title('YZ Projection')
    ax_yz.grid(True)
    # Legend will be added as a single legend for all subplots
    
    # Projection onto basis2-basis3 plane
    ax_basis = fig.add_subplot(gs[1, 1])
    
    # Project the points onto the basis2-basis3 plane
    points_basis = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        # Calculate the vector from R_0 to the point
        vec = point - R_0
        # Project onto basis2 and basis3
        points_basis[i, 0] = np.dot(vec, basis2)
        points_basis[i, 1] = np.dot(vec, basis3)
    
    # Plot the projected points
    ax_basis.scatter(points_basis[:, 0], points_basis[:, 1], color='red', s=20, alpha=0.7, label='Circle Points')
    ax_basis.plot(points_basis[:, 0], points_basis[:, 1], 'r-', alpha=0.5)
    
    # Plot the circle in the orthogonal plane
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = d * np.cos(theta)
    circle_y = d * np.sin(theta)
    ax_basis.plot(circle_x, circle_y, 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    
    # Plot the origin (which is R_0 in this projection)
    ax_basis.scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    
    ax_basis.set_xlabel('Basis Vector 1 Direction')
    ax_basis.set_ylabel('Basis Vector 2 Direction')
    ax_basis.set_title('Projection onto Plane Orthogonal to x=y=z')
    ax_basis.grid(True)
    ax_basis.legend()
    
    # Make the plot square
    ax_basis.set_aspect('equal')
    
    # Set equal aspect ratio for all projections
    for ax in [ax_xy, ax_xz, ax_yz]:
        ax.set_aspect('equal')
    
    # Add a single legend for all subplots
    handles, labels = ax_xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_dir}/perfect_circle_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot just for the basis2-basis3 projection
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the projected points
    ax.scatter(points_basis[:, 0], points_basis[:, 1], color='red', s=20, alpha=0.7, label='Circle Points')
    ax.plot(points_basis[:, 0], points_basis[:, 1], 'r-', alpha=0.5)
    
    # Plot the origin (which is R_0 in this projection)
    ax.scatter(0, 0, color='blue', s=100, label=f'Origin (R_0 {R_0_truncated})')
    
    # Plot the circle
    ax.plot(circle_x, circle_y, 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    
    ax.set_xlabel('Basis Vector 1 Direction')
    ax.set_ylabel('Basis Vector 2 Direction')
    ax.set_title('Perfect Circle in Plane Orthogonal to x=y=z')
    ax.grid(True)
    ax.legend()
    
    # Make the plot square
    ax.set_aspect('equal')
    
    # Add a single legend for all subplots
    handles, labels = ax_xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_dir}/perfect_circle_orthogonal_plane.png', dpi=300, bbox_inches='tight')
    plt.close()

def verify_circle_properties(d, num_points, points, R_0, save_dir):
    """
    Verify that the generated points form a perfect circle.
    
    Parameters:
    d (float): The distance parameter
    num_points (int): Number of points generated
    points (numpy.ndarray): Array of points forming the circle
    R_0 (tuple or numpy.ndarray): The origin vector
    save_dir (str): Directory to save the verification results
    
    Returns:
    dict: Dictionary containing verification results
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Calculate the distances from each point to R_0
    distances = np.linalg.norm(points - R_0, axis=1)
    
    # Calculate the dot product of each (point - R_0) with the basis1 vector
    dot_products = np.array([np.dot(point - R_0, basis1) for point in points])
    
    # Calculate the projections onto the basis2-basis3 plane
    projections = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        vec = point - R_0
        projections[i, 0] = np.dot(vec, basis2)
        projections[i, 1] = np.dot(vec, basis3)
    
    # Calculate the distances from each projection to the origin in the basis2-basis3 plane
    projection_distances = np.linalg.norm(projections, axis=1)
    
    # Calculate statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    mean_dot_product = np.mean(dot_products)
    std_dot_product = np.std(dot_products)
    mean_projection_distance = np.mean(projection_distances)
    std_projection_distance = np.std(projection_distances)
    
    # Create a dictionary with the verification results
    results = {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'mean_dot_product': mean_dot_product,
        'std_dot_product': std_dot_product,
        'mean_projection_distance': mean_projection_distance,
        'std_projection_distance': std_projection_distance,
        'expected_distance': d,
        'expected_dot_product': 0,
        'expected_projection_distance': d
    }
    
    # Save the verification results to a file
    with open(f'{save_dir}/verification_results.txt', 'w') as f:
        f.write(f"Verification Results for Perfect Circle (R_0 = {tuple(R_0)}, d = {d}):\n")
        f.write(f"Number of points: {num_points}\n\n")
        f.write(f"Distance from R_0 to points:\n")
        f.write(f"  Expected: {d}\n")
        f.write(f"  Mean: {mean_distance}\n")
        f.write(f"  Std Dev: {std_distance}\n\n")
        f.write(f"Dot product with basis1 (should be 0 for orthogonality):\n")
        f.write(f"  Expected: 0\n")
        f.write(f"  Mean: {mean_dot_product}\n")
        f.write(f"  Std Dev: {std_dot_product}\n\n")
        f.write(f"Distance in basis2-basis3 plane (should be equal to d):\n")
        f.write(f"  Expected: {d}\n")
        f.write(f"  Mean: {mean_projection_distance}\n")
        f.write(f"  Std Dev: {std_projection_distance}\n")
    
    return results

def visualize_vectorz(R_0, d, num_points, theta_min, theta_max, save_dir):
    """
    Improved version of the visualize_vectorz function that properly handles R0 at any position.
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector
    d (float): The distance parameter
    num_points (int): Number of points to generate
    theta_min (float): Minimum angle in radians
    theta_max (float): Maximum angle in radians
    save_dir (str): Directory to save the visualizations
    """
    # Generate the points
    points = create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max)
    
    # Visualize the points
    improved_visualize_perfect_orthogonal_circle(points, R_0, d, save_dir)
    
    # Verify the circle properties
    verify_circle_properties(d, num_points, points, R_0, save_dir)
    
    return points

if __name__ == "__main__":
    # Example usage
    R_0 = (0.433, 0.433, 0.433)  # The R_0 point from gabor_bph.py
    d = 0.001  # The distance parameter
    num_points = 36  # Number of points to generate
    theta_min = 0  # Minimum angle in radians
    theta_max = 2 * np.pi  # Maximum angle in radians
    save_dir = "output/improved_visualization"  # Directory to save the visualizations
    
    # Generate and visualize the points
    points = visualize_vectorz(R_0, d, num_points, theta_min, theta_max, save_dir)
    
    print(f"Visualization saved to {save_dir}")
