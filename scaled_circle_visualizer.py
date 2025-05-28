#!/usr/bin/env python3
"""
Scaled visualization for perfect circles in the plane orthogonal to the x=y=z line.
This script properly handles R0 at any position and scales the circle for better visibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_scaled_circle(points, R_0, d, save_dir, scale_factor=60):
    """
    Visualize a perfect circle in the plane orthogonal to the x=y=z line,
    with scaling for better visibility.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    R_0 (numpy.ndarray): The center point of the circle
    d (float): The actual radius of the circle
    save_dir (str): Directory to save the visualizations
    scale_factor (float): Factor to scale the circle for better visibility
    """
    # Create scaled_plots directory within save_dir
    scaled_dir = os.path.join(save_dir, 'scaled_plots')
    os.makedirs(scaled_dir, exist_ok=True)
    
    # Convert R_0 to numpy array
    R_0 = np.array(R_0)
    R_0_rounded = tuple(round(float(val), 3) for val in R_0)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the R_0 point
    ax.scatter([R_0[0]], [R_0[1]], [R_0[2]], color='black', s=100, label=f'R_0 {R_0_rounded}')
    
    # Plot the circle with a thick red line
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=4, alpha=0.9, label=f'Circle (d={d})')
    
    # Plot the x=y=z line
    # Calculate the maximum distance from R_0 to any point
    max_val = np.max(np.linalg.norm(points - R_0, axis=1)) * 1.5
    line_start = R_0 - max_val * np.array([1, 1, 1]) / np.sqrt(3)
    line_end = R_0 + max_val * np.array([1, 1, 1]) / np.sqrt(3)
    ax.plot([line_start[0], line_end[0]], 
            [line_start[1], line_end[1]], 
            [line_start[2], line_end[2]], 
            'k-', alpha=0.8, label='x=y=z line')
    
    # Plot the coordinate axes centered at R_0
    # X-axis - red
    ax.plot([R_0[0]-max_val, R_0[0]+max_val], [R_0[1], R_0[1]], [R_0[2], R_0[2]], 'r-', linewidth=2, alpha=0.8)
    # Y-axis - green
    ax.plot([R_0[0], R_0[0]], [R_0[1]-max_val, R_0[1]+max_val], [R_0[2], R_0[2]], 'g-', linewidth=2, alpha=0.8)
    # Z-axis - blue
    ax.plot([R_0[0], R_0[0]], [R_0[1], R_0[1]], [R_0[2]-max_val, R_0[2]+max_val], 'b-', linewidth=2, alpha=0.8)
    
    # Add axis labels at the ends
    ax.text(R_0[0]+max_val, R_0[1], R_0[2], 'X', color='red', fontsize=12)
    ax.text(R_0[0], R_0[1]+max_val, R_0[2], 'Y', color='green', fontsize=12)
    ax.text(R_0[0], R_0[1], R_0[2]+max_val, 'Z', color='blue', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{scaled_dir}/perfect_circle_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create projections onto the coordinate planes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Center all points around R_0
    centered_points = points - R_0
    
    # XY projection
    axs[0, 0].plot(centered_points[:, 0] + R_0[0], centered_points[:, 1] + R_0[1], 'r-', linewidth=4, alpha=0.9)
    axs[0, 0].scatter(R_0[0], R_0[1], color='black', s=100)
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].grid(True)
    
    # XZ projection
    axs[0, 1].plot(centered_points[:, 0] + R_0[0], centered_points[:, 2] + R_0[2], 'r-', linewidth=4, alpha=0.9)
    axs[0, 1].scatter(R_0[0], R_0[2], color='black', s=100)
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].grid(True)
    
    # YZ projection
    axs[1, 0].plot(centered_points[:, 1] + R_0[1], centered_points[:, 2] + R_0[2], 'r-', linewidth=4, alpha=0.9)
    axs[1, 0].scatter(R_0[1], R_0[2], color='black', s=100)
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('Z')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].grid(True)
    
    # Project the circle points onto the basis2-basis3 plane
    circle_basis = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        vec = point - R_0
        circle_basis[i, 0] = np.dot(vec, basis2)
        circle_basis[i, 1] = np.dot(vec, basis3)
    
    # Plot the projection onto the plane orthogonal to x=y=z
    axs[1, 1].plot(circle_basis[:, 0], circle_basis[:, 1], 'r-', linewidth=4, alpha=0.9)
    axs[1, 1].scatter(0, 0, color='black', s=100)
    axs[1, 1].set_xlabel('Basis Vector 1 Direction')
    axs[1, 1].set_ylabel('Basis Vector 2 Direction')
    axs[1, 1].set_title('Projection onto Plane Orthogonal to x=y=z')
    axs[1, 1].grid(True)
    
    # Add legend to the orthogonal projection only
    axs[1, 1].legend(['Circle Points', f'R_0 {R_0_rounded}'], loc='upper right', fontsize=8)
    
    # Make all plots square
    for ax in axs.flat:
        ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{scaled_dir}/perfect_circle_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scaled visualizations saved to {scaled_dir}")

def visualize_scaled_basis_projection(points, R_0, d, save_dir, scale_factor=60):
    """
    Create a separate visualization of the circle projected onto the basis vectors
    orthogonal to the x=y=z line.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    R_0 (numpy.ndarray): The center point of the circle
    d (float): The actual radius of the circle
    save_dir (str): Directory to save the visualizations
    scale_factor (float): Factor to scale the circle for better visibility
    """
    # Create scaled_plots directory within save_dir
    scaled_dir = os.path.join(save_dir, 'scaled_plots')
    os.makedirs(scaled_dir, exist_ok=True)
    
    # Convert R_0 to numpy array
    R_0 = np.array(R_0)
    R_0_rounded = tuple(round(float(val), 3) for val in R_0)
    
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Center all points around R_0
    centered_points = points - R_0
    
    # Project the circle points onto the basis2-basis3 plane
    circle_basis = np.zeros((len(points), 2))
    for i, point in enumerate(centered_points):
        circle_basis[i, 0] = np.dot(point, basis2)
        circle_basis[i, 1] = np.dot(point, basis3)
    
    # Create a figure for the basis projection
    plt.figure(figsize=(10, 10))
    
    # Plot the projection onto the plane orthogonal to x=y=z
    plt.plot(circle_basis[:, 0], circle_basis[:, 1], 'r-', linewidth=4, alpha=0.9, label=f'Circle (d={d})')
    plt.scatter(0, 0, color='black', s=100, label=f'R_0 {R_0_rounded}')
    
    # Add coordinate axes
    max_val = np.max(np.abs(circle_basis)) * 1.2
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Basis Vector 1 Direction')
    plt.ylabel('Basis Vector 2 Direction')
    plt.title('Projection onto Plane Orthogonal to x=y=z')
    plt.grid(True)
    plt.axis('equal')
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Set limits
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{scaled_dir}/scaled_basis_projection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scaled basis projection saved to {scaled_dir}/scaled_basis_projection.png")

def verify_scaled_circle_properties(points, R_0, d, save_dir, scale_factor=60):
    """
    Verify that the scaled circle has the expected properties and save the results.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    R_0 (numpy.ndarray): The center point of the circle
    d (float): The actual radius of the circle
    save_dir (str): Directory to save the verification results
    scale_factor (float): Factor used to scale the circle for better visibility
    """
    # Create scaled_plots directory within save_dir
    scaled_dir = os.path.join(save_dir, 'scaled_plots')
    os.makedirs(scaled_dir, exist_ok=True)
    
    # Convert R_0 to numpy array
    R_0 = np.array(R_0)
    
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    
    # Calculate the actual radius of the scaled circle
    actual_radius = np.mean([np.linalg.norm(p - R_0) for p in points])
    expected_radius = d * scale_factor
    radius_error = abs(actual_radius - expected_radius) / expected_radius * 100
    
    # Calculate the distance from each point to the x=y=z line
    distances_to_line = []
    for point in points:
        # Vector from R_0 to the point
        v = point - R_0
        # Project v onto the x=y=z line
        proj = np.dot(v, basis1) * basis1
        # Distance from point to its projection on the line
        distance = np.linalg.norm(v - proj)
        distances_to_line.append(distance)
    
    mean_distance = np.mean(distances_to_line)
    std_distance = np.std(distances_to_line)
    
    # Calculate the angle between each point-R_0 vector and the x=y=z line
    angles = []
    for point in points:
        v = point - R_0
        cos_angle = np.dot(v, basis1) / (np.linalg.norm(v) * np.linalg.norm(basis1))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        angles.append(angle)
    
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    # Write verification results to a file
    with open(f'{scaled_dir}/scaled_verification_results.txt', 'w') as f:
        f.write(f"Verification Results for Scaled Circle (scale_factor={scale_factor}):\n")
        f.write(f"R_0: {R_0}\n")
        f.write(f"Actual d: {d}\n")
        f.write(f"Scaled d: {d * scale_factor}\n\n")
        
        f.write(f"Expected radius: {expected_radius}\n")
        f.write(f"Actual radius: {actual_radius}\n")
        f.write(f"Radius error: {radius_error:.6f}%\n\n")
        
        f.write(f"Mean distance to x=y=z line: {mean_distance}\n")
        f.write(f"Standard deviation of distance: {std_distance}\n\n")
        
        f.write(f"Mean angle with x=y=z line: {mean_angle} degrees\n")
        f.write(f"Standard deviation of angle: {std_angle} degrees\n")
    
    print(f"Scaled verification results saved to {scaled_dir}/scaled_verification_results.txt")
    
    # Create a visualization of the verification results
    plt.figure(figsize=(12, 10))
    
    # Plot the distances to the x=y=z line
    plt.subplot(2, 1, 1)
    plt.plot(distances_to_line, 'b-', linewidth=2)
    plt.axhline(y=mean_distance, color='r', linestyle='--', label=f'Mean: {mean_distance:.6f}')
    plt.axhline(y=mean_distance + std_distance, color='g', linestyle=':', label=f'Std: {std_distance:.6f}')
    plt.axhline(y=mean_distance - std_distance, color='g', linestyle=':')
    plt.title('Distance from Points to x=y=z Line')
    plt.xlabel('Point Index')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.legend()
    
    # Plot the angles with the x=y=z line
    plt.subplot(2, 1, 2)
    plt.plot(angles, 'b-', linewidth=2)
    plt.axhline(y=mean_angle, color='r', linestyle='--', label=f'Mean: {mean_angle:.2f} degrees')
    plt.axhline(y=mean_angle + std_angle, color='g', linestyle=':', label=f'Std: {std_angle:.2f} degrees')
    plt.axhline(y=mean_angle - std_angle, color='g', linestyle=':')
    plt.title('Angle between Point-R_0 Vector and x=y=z Line')
    plt.xlabel('Point Index')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{scaled_dir}/scaled_verification_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scaled verification plot saved to {scaled_dir}/scaled_verification_plot.png")

def create_and_visualize_scaled_circle(R_0, d, num_points, theta_min, theta_max, save_dir, scale_factor=60):
    """
    Create a perfect circle and visualize it with scaling for better visibility.
    
    Parameters:
    R_0 (numpy.ndarray): The center point of the circle
    d (float): The actual radius of the circle
    num_points (int): Number of points to generate
    theta_min (float): Minimum angle in radians
    theta_max (float): Maximum angle in radians
    save_dir (str): Directory to save the visualizations
    scale_factor (float): Factor to scale the circle for better visibility
    """
    from generalized.vector_utils import create_perfect_orthogonal_circle
    
    # Create a scaled circle for better visualization
    display_d = d * scale_factor
    scaled_points = create_perfect_orthogonal_circle(R_0, display_d, num_points, theta_min, theta_max)
    
    # Visualize the scaled circle
    visualize_scaled_circle(scaled_points, R_0, d, save_dir, scale_factor)
    
    # Create a separate basis projection visualization
    visualize_scaled_basis_projection(scaled_points, R_0, d, save_dir, scale_factor)
    
    # Verify the scaled circle properties
    verify_scaled_circle_properties(scaled_points, R_0, d, save_dir, scale_factor)
    
    return scaled_points
