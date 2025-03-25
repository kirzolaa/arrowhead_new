#!/usr/bin/env python3
"""
Generate perfect orthogonal circles with different distance values.
This script creates circles in the plane orthogonal to the (1,1,1) direction
with distances ranging from 0.5 to 3.0 in 0.5 increments.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add the generalized directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_circle

# Set up the figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define the origin
R_0 = np.array([0.0, 0.0, 0.0])

# Define the range of distances
distances = np.arange(0.5, 3.5, 0.5)

# Define the theta range (full circle)
theta_start = 0
theta_end = 2 * np.pi
num_points = 73  # 5-degree increments

# Generate and plot circles for each distance
colors = plt.cm.viridis(np.linspace(0, 1, len(distances)))

# Store all circle points for scaling calculations
all_circles_points = []

for i, d in enumerate(distances):
    # Generate the circle
    vectors = create_perfect_orthogonal_circle(
        R_0=R_0,
        d=d,
        start_theta=theta_start,
        end_theta=theta_end,
        num_points=num_points
    )
    
    # Store points for scaling calculations
    all_circles_points.append(vectors)
    
    # Plot the circle
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], 
               color=colors[i], label=f'd = {d}')
    
    # We'll keep the circles without the connecting lines to make the visualization cleaner

# Plot the origin
ax.scatter(R_0[0], R_0[1], R_0[2], color='red', s=100, marker='o', label='Origin R_0')

# Add axis lines with higher visibility and labels
# Adjust max_val to be closer to the actual data for better visualization
max_val = max(np.max(np.abs(distances)) * 1.5, 3.5)

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

# Plot the x=y=z line
line = np.array([[-1, -1, -1], [7, 7, 7]])
ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r--', label='x=y=z line')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Perfect Orthogonal Circles with Different Distances')

# Set equal aspect ratio and adjust limits for better viewing
buffer = max_val * 0.2  # Add a small buffer for better visibility

# Calculate actual data bounds for better scaling
# Use the stored points for scaling calculations
all_circle_points = np.vstack(all_circles_points)
data_max = np.max(np.abs(all_circle_points)) * 1.2

# Use data-driven limits instead of the larger max_val
ax.set_xlim([-data_max-buffer, data_max+buffer])
ax.set_ylim([-data_max-buffer, data_max+buffer])
ax.set_zlim([-data_max-buffer, data_max+buffer])

# Set equal aspect ratio for better 3D visualization
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend()

# Save the figure
plt.savefig('perfect_circle_distance_range.png', dpi=300, bbox_inches='tight')

# Create a second figure with 2D projections
fig2, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

# XY Projection
for i, d in enumerate(distances):
    vectors = create_perfect_orthogonal_circle(
        R_0=R_0,
        d=d,
        start_theta=theta_start,
        end_theta=theta_end,
        num_points=num_points
    )
    axs[0].scatter(vectors[:, 0], vectors[:, 1], color=colors[i], label=f'd = {d}')

axs[0].scatter(R_0[0], R_0[1], color='red', s=100, marker='o')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('XY Projection')
axs[0].grid(True)
axs[0].set_aspect('equal')

# XZ Projection
for i, d in enumerate(distances):
    vectors = create_perfect_orthogonal_circle(
        R_0=R_0,
        d=d,
        start_theta=theta_start,
        end_theta=theta_end,
        num_points=num_points
    )
    axs[1].scatter(vectors[:, 0], vectors[:, 2], color=colors[i], label=f'd = {d}')

axs[1].scatter(R_0[0], R_0[2], color='red', s=100, marker='o')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Z')
axs[1].set_title('XZ Projection')
axs[1].grid(True)
axs[1].set_aspect('equal')

# YZ Projection
for i, d in enumerate(distances):
    vectors = create_perfect_orthogonal_circle(
        R_0=R_0,
        d=d,
        start_theta=theta_start,
        end_theta=theta_end,
        num_points=num_points
    )
    axs[2].scatter(vectors[:, 1], vectors[:, 2], color=colors[i], label=f'd = {d}')

axs[2].scatter(R_0[1], R_0[2], color='red', s=100, marker='o')
axs[2].set_xlabel('Y')
axs[2].set_ylabel('Z')
axs[2].set_title('YZ Projection')
axs[2].grid(True)
axs[2].set_aspect('equal')

# Projection onto the plane orthogonal to the x=y=z line
# We'll create a custom projection by calculating coordinates in the orthogonal plane

# Define the normalized basis vectors for the orthogonal plane
basis1 = np.array([1, -1/2, -1/2])
basis1 = basis1 / np.linalg.norm(basis1)
basis2 = np.array([0, -1/2, 1/2])
basis2 = basis2 / np.linalg.norm(basis2)

# Calculate the projection coordinates for each circle
for i, d in enumerate(distances):
    vectors = create_perfect_orthogonal_circle(
        R_0=R_0,
        d=d,
        start_theta=theta_start,
        end_theta=theta_end,
        num_points=num_points
    )
    
    # Calculate displacement vectors from origin
    displacements = vectors - R_0
    
    # Project onto the basis vectors
    coord1 = np.array([np.dot(disp, basis1) for disp in displacements])
    coord2 = np.array([np.dot(disp, basis2) for disp in displacements])
    
    # Plot in the orthogonal plane
    axs[3].scatter(coord1, coord2, color=colors[i], label=f'd = {d}')

# Mark the origin in the orthogonal plane
axs[3].scatter(0, 0, color='red', s=100, marker='o')
axs[3].set_xlabel('Basis Vector 1 Direction')
axs[3].set_ylabel('Basis Vector 2 Direction')
axs[3].set_title('Projection onto Plane ⊥ to x=y=z')
axs[3].grid(True)
axs[3].set_aspect('equal')

# Add a legend
axs[3].legend(fontsize=10)

plt.tight_layout()
plt.savefig('perfect_circle_distance_range_projections.png', dpi=300, bbox_inches='tight')

print("Generated perfect orthogonal circles with different distances.")
print("Figures saved as 'perfect_circle_distance_range.png' and 'perfect_circle_distance_range_projections.png'")
