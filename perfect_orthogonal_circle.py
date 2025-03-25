#!/usr/bin/env python3
"""
Generate and visualize a perfect circle in the plane orthogonal to the x=y=z line.
This script uses R_0 = (0,0,0), d=1, and theta ranging from 0 to 360 degrees in 5-degree steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_perfect_orthogonal_circle(d=1.0, num_points=73):
    """
    Generate a perfect circle in the plane orthogonal to the x=y=z line.
    
    Parameters:
    d (float): Distance parameter, default is 1.0
    num_points (int): Number of points to generate, default is 73 (5-degree steps)
    
    Returns:
    numpy.ndarray: Array of points forming the circle
    """
    # Set R_0 to (0,0,0)
    R_0 = np.array([0, 0, 0])
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])
    basis2 = np.array([0, -1/2, 1/2])
    
    # Normalize the basis vectors
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Generate theta values from 0 to 2π
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=True)
    
    # Generate points directly using the basis vectors to ensure a perfect circle
    points = []
    for theta in thetas:
        # Create a point at distance d from the origin in the plane spanned by basis1 and basis2
        point = R_0 + d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
        points.append(point)
    
    return np.array(points)

def visualize_perfect_orthogonal_circle(points, save_dir):
    """
    Visualize the perfect circle in the plane orthogonal to the x=y=z line.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    """
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter([0], [0], [0], color='black', s=100, label='Origin (R_0)')
    
    # Plot the circle points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=20, alpha=0.7, label='Circle Points')
    
    # Connect the points to show the circle
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.5)
    
    # Add axis lines with higher visibility and labels
    # Adjust max_val to be closer to the actual data for better visualization
    max_val = np.max(np.abs(points)) * 1.5
    
    # X-axis - red with label and coordinate markers
    ax.plot([-max_val, max_val], [0, 0], [0, 0], 'r-', alpha=0.6, linewidth=1.0)
    ax.text(max_val*1.1, 0, 0, 'X', color='red', fontsize=12)
    
    # Add coordinate markers along X-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(i, 0, 0, f'{i}', color='red', fontsize=8, ha='center', va='bottom')
            # Add small tick marks
            ax.plot([i, i], [0, -0.05], [0, 0], 'r-', alpha=0.4, linewidth=0.5)
    
    # Y-axis - green with label and coordinate markers
    ax.plot([0, 0], [-max_val, max_val], [0, 0], 'g-', alpha=0.6, linewidth=1.0)
    ax.text(0, max_val*1.1, 0, 'Y', color='green', fontsize=12)
    
    # Add coordinate markers along Y-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(0, i, 0, f'{i}', color='green', fontsize=8, ha='right', va='center')
            # Add small tick marks
            ax.plot([0, -0.05], [i, i], [0, 0], 'g-', alpha=0.4, linewidth=0.5)
    
    # Z-axis - blue with label and coordinate markers
    ax.plot([0, 0], [0, 0], [-max_val, max_val], 'b-', alpha=0.6, linewidth=1.0)
    ax.text(0, 0, max_val*1.1, 'Z', color='blue', fontsize=12)
    
    # Add coordinate markers along Z-axis
    for i in range(-int(max_val), int(max_val)+1):
        if i != 0 and i % 1 == 0:  # Only show integer values, skip zero
            ax.text(0, 0, i, f'{i}', color='blue', fontsize=8, ha='right', va='center')
            # Add small tick marks
            ax.plot([0, -0.05], [0, 0], [i, i], 'b-', alpha=0.4, linewidth=0.5)
    
    # Plot the (1,1,1) direction
    max_val = np.max(np.abs(points)) * 1.5
    ax.plot([0, max_val], [0, max_val], [0, max_val], 'k-', label='x=y=z line')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Perfect Circle Orthogonal to x=y=z line (R_0 = (0,0,0), d=1)')
    
    # Set equal aspect ratio and adjust limits for better viewing
    buffer = max_val * 0.2  # Add a small buffer for better visibility
    
    # Calculate actual data bounds for better scaling
    data_max = np.max(np.abs(points)) * 1.2
    
    # Use data-driven limits instead of the larger max_val
    ax.set_xlim([-data_max-buffer, data_max+buffer])
    ax.set_ylim([-data_max-buffer, data_max+buffer])
    ax.set_zlim([-data_max-buffer, data_max+buffer])
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.savefig(f'{save_dir}/perfect_circle_3d.png', dpi=300, bbox_inches='tight')
    
    # Create projections onto the coordinate planes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY projection
    axs[0, 0].scatter(points[:, 0], points[:, 1], color='red', s=20, alpha=0.7)
    axs[0, 0].plot(points[:, 0], points[:, 1], 'r-', alpha=0.5)
    axs[0, 0].scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_aspect('equal')
    
    # XZ projection
    axs[0, 1].scatter(points[:, 0], points[:, 2], color='red', s=20, alpha=0.7)
    axs[0, 1].plot(points[:, 0], points[:, 2], 'r-', alpha=0.5)
    axs[0, 1].scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_aspect('equal')
    
    # YZ projection
    axs[1, 0].scatter(points[:, 1], points[:, 2], color='red', s=20, alpha=0.7)
    axs[1, 0].plot(points[:, 1], points[:, 2], 'r-', alpha=0.5)
    axs[1, 0].scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('Z')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_aspect('equal')
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])
    basis2 = np.array([0, -1/2, 1/2])
    
    # Normalize them
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Project all points onto the orthogonal plane
    points_proj = []
    for p in points:
        p_proj = np.array([np.dot(p, basis1), np.dot(p, basis2)])
        points_proj.append(p_proj)
    
    points_proj = np.array(points_proj)
    
    # Orthogonal plane projection
    axs[1, 1].scatter(points_proj[:, 0], points_proj[:, 1], color='red', s=20, alpha=0.7)
    axs[1, 1].plot(points_proj[:, 0], points_proj[:, 1], 'r-', alpha=0.5)
    axs[1, 1].scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    axs[1, 1].set_xlabel('Basis Vector 1 Direction')
    axs[1, 1].set_ylabel('Basis Vector 2 Direction')
    axs[1, 1].set_title('Projection onto Plane Orthogonal to x=y=z')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/perfect_circle_projections.png', dpi=300, bbox_inches='tight')
    
    # Create a separate figure for the projection onto the orthogonal plane
    plt.figure(figsize=(8, 8))
    plt.scatter(points_proj[:, 0], points_proj[:, 1], color='red', s=30, alpha=0.7)
    plt.plot(points_proj[:, 0], points_proj[:, 1], 'r-', alpha=0.5)
    plt.scatter(0, 0, color='black', s=100, label='Origin (R_0)')
    plt.xlabel('Basis Vector 1 Direction')
    plt.ylabel('Basis Vector 2 Direction')
    plt.title('Perfect Circle in Plane Orthogonal to x=y=z')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig(f'{save_dir}/perfect_circle_orthogonal_plane.png', dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {save_dir} directory.")

def verify_circle_properties(d, num_points, points, save_dir):
    """
    Verify that the generated points form a perfect circle.
    
    Parameters:
    points (numpy.ndarray): Array of points forming the circle
    
    Returns:
    dict: Dictionary containing verification results
    """
    # Calculate distances from origin
    distances = np.linalg.norm(points, axis=1)
    
    # Calculate statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    # Verify orthogonality to (1,1,1) direction
    unit_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized (1,1,1) vector
    
    # Calculate dot products for each point
    dot_products = []
    for p in points:
        dot_product = np.abs(np.dot(p, unit_111))
        dot_products.append(dot_product)
    
    max_dot_product = max(dot_products)
    avg_dot_product = sum(dot_products) / len(dot_products)
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])
    basis2 = np.array([0, -1/2, 1/2])
    
    # Normalize them
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Project all points onto the orthogonal plane
    points_proj = []
    for p in points:
        p_proj = np.array([np.dot(p, basis1), np.dot(p, basis2)])
        points_proj.append(p_proj)
    
    points_proj = np.array(points_proj)
    
    # Calculate distances from origin in the projected plane
    proj_distances = np.linalg.norm(points_proj, axis=1)
    
    # Calculate statistics for projected points
    proj_mean_distance = np.mean(proj_distances)
    proj_std_distance = np.std(proj_distances)
    proj_min_distance = np.min(proj_distances)
    proj_max_distance = np.max(proj_distances)
    
    verification = {
        "3D Circle Properties": {
            "Mean Distance from Origin": mean_distance,
            "Standard Deviation of Distances": std_distance,
            "Minimum Distance": min_distance,
            "Maximum Distance": max_distance,
            "Max/Min Ratio": max_distance / min_distance if min_distance > 0 else float('inf')
        },
        "Orthogonality Verification": {
            "Maximum Dot Product with (1,1,1)": max_dot_product,
            "Average Dot Product with (1,1,1)": avg_dot_product
        },
        "2D Projection Properties": {
            "Mean Distance from Origin": proj_mean_distance,
            "Standard Deviation of Distances": proj_std_distance,
            "Minimum Distance": proj_min_distance,
            "Maximum Distance": proj_max_distance,
            "Max/Min Ratio": proj_max_distance / proj_min_distance if proj_min_distance > 0 else float('inf')
        }
    }

    # Save verification results to file
    with open(f'{save_dir}/verification_results.txt', 'w') as f:
        f.write("Perfect Circle Properties Verification\n")
        f.write("======================================\n\n")
        f.write(f"Parameters: R_0=(0,0,0), d={d}, num_points={num_points}\n\n")
        
        for category, properties in verification.items():
            f.write(f"{category}:\n")
            for prop, value in properties.items():
                f.write(f"  {prop}: {value}\n")
            f.write("\n")
    
    print(f"\nVerification results saved to {save_dir}/verification_results.txt")

    return verification