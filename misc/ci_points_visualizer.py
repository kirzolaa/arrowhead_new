#!/usr/bin/env python3
"""
Visualize the three CI points around R0=(0.433, 0.433, 0.433) using the improved visualization.
This script shows the CI points arranged 120° apart around the origin on a circle orthogonal
to the x=y=z line.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from improved_circle_visualization import visualize_vectorz, create_perfect_orthogonal_circle

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

def visualize_ci_points(output_dir="berry_phase_corrected_run_n_minus_1/vectors", d_scale=1.0):
    """
    Visualize the CI points around R0=(0.433, 0.433, 0.433).
    
    Parameters:
    output_dir (str): Directory to save the visualizations
    d_scale (float): Scale factor for the circle radius to make it more visible
    """
    """
    Visualize the CI points around R0=(0.433, 0.433, 0.433).
    
    Parameters:
    output_dir (str): Directory to save the visualizations
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the CI points
    aVa = 1.3
    aVx = 1.0
    x_shift = 0.1
    d = 0.001
    
    R_0, CI_points = calculate_ci_points(aVa, aVx, x_shift, d)
    
    # Print the values for verification with truncated coordinates
    R_0_truncated = tuple(round(float(val), 3) for val in R_0)
    print(f"R_0: {R_0_truncated}    r0: {round(float(R_0[0]), 3)}  sum(R_0)/3: {round(float(sum(R_0)/3), 3)}")
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
    
    # Create a perfect circle for reference with scaled radius for better visibility
    num_points = 36
    theta_min = 0
    theta_max = 2 * np.pi
    # Scale the radius to make the circle more visible
    circle_d = d * d_scale
    circle_points = create_perfect_orthogonal_circle(R_0, circle_d, num_points, theta_min, theta_max)
    
    # Visualize the points using our improved visualization
    visualize_vectorz(R_0, d, num_points, theta_min, theta_max, output_dir)
    
    # Create a 3D visualization with the CI points specifically highlighted
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the R_0 point with truncated coordinates
    R_0_truncated = tuple(round(float(val), 3) for val in R_0)
    ax.scatter([R_0[0]], [R_0[1]], [R_0[2]], color='black', s=100, label=f'Origin (R_0)')
    
    # Plot the circle with a thicker, more visible red line
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'r-', linewidth=4, alpha=0.9, label='Circle Points')
    
    # Plot the CI points with larger markers and truncated coordinates
    colors = ['green', 'purple', 'orange']
    for i, point in enumerate(CI_points):
        point_truncated = tuple(round(float(val), 3) for val in point)
        ax.scatter([point[0]], [point[1]], [point[2]], color=colors[i], s=150, label=f'CI Point {i+1}')
    
    # Plot the x=y=z line
    max_val = max(np.max(np.abs(circle_points)), np.max(np.abs(R_0))) * 1.5
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
    ax.set_title(f'CI Points Around R_0 = {tuple(R_0)}')
    
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
    
    # Add legend with better positioning in a single row
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
    
    # Add grid
    ax.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{output_dir}/ci_points_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 2D visualization of the CI points in the basis2-basis3 plane
    # Define the basis vectors
    basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
    basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
    basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)
    
    # Normalize the basis vectors
    basis1 = basis1_raw / np.linalg.norm(basis1_raw)
    basis2 = basis2_raw / np.linalg.norm(basis2_raw)
    basis3 = basis3_raw / np.linalg.norm(basis3_raw)
    
    # Create a figure with 2x2 subplots for the projections
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Project the CI points onto the basis2-basis3 plane
    ci_points_basis = np.zeros((len(CI_points), 2))
    for i, point in enumerate(CI_points):
        # Calculate the vector from R_0 to the point
        vec = point - R_0
        # Project onto basis2 and basis3
        ci_points_basis[i, 0] = np.dot(vec, basis2)
        ci_points_basis[i, 1] = np.dot(vec, basis3)
    
    # Create the 2D visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = d * np.cos(theta)
    circle_y = d * np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, alpha=0.5, label='Perfect Circle')
    
    # Plot the origin (which is R_0 in this projection)
    ax.scatter(0, 0, color='blue', s=100, label=f'Origin (R_0)')
    
    # Plot the CI points with truncated coordinates
    for i, point in enumerate(ci_points_basis):
        point_truncated = (round(float(point[0]), 3), round(float(point[1]), 3))
        ax.scatter(point[0], point[1], color=colors[i], s=150, label=f'CI Point {i+1}: {point_truncated}')
    
    # Connect the CI points to form a triangle
    ci_points_basis_closed = np.vstack([ci_points_basis, ci_points_basis[0]])
    ax.plot(ci_points_basis_closed[:, 0], ci_points_basis_closed[:, 1], 'k-', alpha=0.7)
    
    # Add angle markers to show the 120° spacing
    for i in range(len(CI_points)):
        # Calculate the angle to the next point
        j = (i + 1) % len(CI_points)
        v1 = ci_points_basis[i]
        v2 = ci_points_basis[j]
        
        # Calculate the angle between the two vectors
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)
        
        # Add a text label for the angle
        mid_point = (v1 + v2) / 2
        ax.text(mid_point[0] * 0.7, mid_point[1] * 0.7, f"{angle_deg:.1f}°", 
                fontsize=10, ha='center', va='center')
    
    ax.set_xlabel('Basis Vector 2 Direction')
    ax.set_ylabel('Basis Vector 3 Direction')
    ax.set_title('CI Points in Plane Orthogonal to x=y=z')
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Make the plot square
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{output_dir}/ci_points_orthogonal_plane.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Use a scale factor to make the circle more visible in the visualization
    # The actual d value is 0.001, but we scale it for better visualization
    visualize_ci_points(d_scale=60.0)
