#!/usr/bin/env python3
"""
Visualize the three CI points around R0=(0.433, 0.433, 0.433) using a perfect circle
orthogonal to the x=y=z line with d=0.001.
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

def calculate_ci_points(aVa=1.3, aVx=1.0, x_shift=0.1, d=0.001):
    """
    Calculate the CI points based on the parameters from gabor_bph.py.
    
    Parameters:
    aVa (float): Parameter for Va potential
    aVx (float): Parameter for Vx potential
    x_shift (float): Shift parameter
    d (float): Distance parameter for the circle
    
    Returns:
    tuple: (R_0, CI_points)
    """
    # Calculate x_prime and r0 using the formula from gabor_bph.py
    x_prime = (aVa/aVx) / (aVa/aVx-1) * x_shift
    r0 = x_prime * 1
    x = d  # Using the d value as the x parameter
    
    # Calculate R_0
    R_0 = np.array([r0, r0, r0])
    
    # Calculate the three CI points (120° apart)
    CI_points = []
    for n_CI in range(3):
        CI_point = np.array([r0+x+x if i == n_CI else r0-x for i in range(3)])
        CI_points.append(CI_point)
    
    return R_0, np.array(CI_points)

def visualize_ci_points(output_dir="berry_phase_corrected_run_n_minus_1/vectors", d=0.001, scale_factor=60):
    """
    Visualize the CI points around R0=(0.433, 0.433, 0.433).
    
    Parameters:
    output_dir (str): Directory to save the visualizations
    d (float): Actual radius of the circle (0.001 for CI points)
    scale_factor (float): Visual scaling factor to make the small circle visible
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the CI points
    aVa = 1.3
    aVx = 1.0
    x_shift = 0.1
    
    R_0, CI_points = calculate_ci_points(aVa, aVx, x_shift, d)
    R_0_rounded = tuple(round(float(val), 3) for val in R_0)
    
    # Print the values for verification with truncated coordinates
    print(f"R_0: {R_0_rounded}    r0: {round(float(R_0[0]), 3)}  sum(R_0)/3: {round(float(sum(R_0)/3), 3)}")
    for i, point in enumerate(CI_points):
        point_truncated = tuple(round(float(val), 3) for val in point)
        print(f"CI Point {i+1}: {point_truncated}")
    
    # Calculate distances between points
    distances_from_R0 = np.linalg.norm(CI_points - R_0, axis=1)
    print("\nDistances from R_0:")
    for i, dist in enumerate(distances_from_R0):
        print(f"CI Point {i+1} to R_0: {dist}")
    
    # Calculate distances between CI points
    print("\nDistances between CI points:")
    for i in range(len(CI_points)):
        for j in range(i+1, len(CI_points)):
            dist = np.linalg.norm(CI_points[i] - CI_points[j])
            print(f"CI Point {i+1} to CI Point {j+1}: {dist}")
    
    # Create a perfect circle for reference with scaled radius for visibility
    num_points = 100
    display_d = d * scale_factor  # Scale for display purposes only
    circle_points = create_perfect_orthogonal_circle(R_0, display_d, num_points)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot R_0 as the origin
    ax.scatter([R_0[0]], [R_0[1]], [R_0[2]], color='black', s=100, label='Origin (R_0)')
    
    # Plot the circle with a thick red line
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    
    # Plot the CI points with larger markers
    colors = ['green', 'purple', 'orange']
    for i, point in enumerate(CI_points):
        point_truncated = tuple(round(float(val), 3) for val in point)
        ax.scatter([point[0]], [point[1]], [point[2]], color=colors[i], s=150, label=f'CI Point {i+1}')
    
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
    title = f'CI Points Around R_0 = {R_0_rounded} with d={d}'
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
    
    # Add legend with better positioning
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ci_points_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create projections onto the coordinate planes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY projection
    axs[0, 0].plot(circle_points[:, 0], circle_points[:, 1], 'r-', linewidth=4, alpha=0.9)
    axs[0, 0].scatter(R_0[0], R_0[1], color='black', s=100)
    for i, point in enumerate(CI_points):
        axs[0, 0].scatter(point[0], point[1], color=colors[i], s=150)
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('XY Projection')
    axs[0, 0].grid(True)
    
    # XZ projection
    axs[0, 1].plot(circle_points[:, 0], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9)
    axs[0, 1].scatter(R_0[0], R_0[2], color='black', s=100)
    for i, point in enumerate(CI_points):
        axs[0, 1].scatter(point[0], point[2], color=colors[i], s=150)
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Z')
    axs[0, 1].set_title('XZ Projection')
    axs[0, 1].grid(True)
    
    # YZ projection
    axs[1, 0].plot(circle_points[:, 1], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9)
    axs[1, 0].scatter(R_0[1], R_0[2], color='black', s=100)
    for i, point in enumerate(CI_points):
        axs[1, 0].scatter(point[1], point[2], color=colors[i], s=150)
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('Z')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 0].grid(True)
    
    # Define the basis vectors for the orthogonal projection
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
    
    # Project the CI points onto the basis2-basis3 plane
    ci_points_basis = np.zeros((len(CI_points), 2))
    for i, point in enumerate(CI_points):
        vec = point - R_0
        ci_points_basis[i, 0] = np.dot(vec, basis2)
        ci_points_basis[i, 1] = np.dot(vec, basis3)
    
    # Plot the projection onto the plane orthogonal to x=y=z
    axs[1, 1].plot(circle_basis[:, 0], circle_basis[:, 1], 'r-', linewidth=4, alpha=0.9)
    axs[1, 1].scatter(0, 0, color='black', s=100)
    for i, point in enumerate(ci_points_basis):
        axs[1, 1].scatter(point[0], point[1], color=colors[i], s=150)
        
    # Connect the CI points to form a triangle
    ci_points_basis_closed = np.vstack([ci_points_basis, ci_points_basis[0]])
    axs[1, 1].plot(ci_points_basis_closed[:, 0], ci_points_basis_closed[:, 1], 'k-', alpha=0.7)
    
    axs[1, 1].set_xlabel('Basis Vector 1 Direction')
    axs[1, 1].set_ylabel('Basis Vector 2 Direction')
    axs[1, 1].set_title('Projection onto Plane Orthogonal to x=y=z')
    axs[1, 1].grid(True)
    
    # Add legend to the orthogonal projection only
    axs[1, 1].legend(['Circle Points', 'Origin (R_0)', 'CI Point 1', 'CI Point 2', 'CI Point 3'], 
                     loc='upper right', fontsize=8)
    
    # Make all plots square
    for ax in axs.flat:
        ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/ci_points_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Use the actual d value of 0.001 with a scale factor to make it visible
    visualize_ci_points(d=0.001, scale_factor=60)
