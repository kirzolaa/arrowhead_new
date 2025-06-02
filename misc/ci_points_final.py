#!/usr/bin/env python3
# Script to visualize CI points as specified in gabor_bph.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# Create output directories with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('output', f'ci_points_final_{timestamp}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_dir_2d = os.path.join(output_dir, 'plot_dir_2d')
plot_dir_3d = os.path.join(output_dir, 'plot_dir_3d')
if not os.path.exists(plot_dir_2d):
    os.makedirs(plot_dir_2d)
if not os.path.exists(plot_dir_3d):
    os.makedirs(plot_dir_3d)

print(f"Output directories created at:\n - {plot_dir_2d}\n - {plot_dir_3d}")

# Define the basis vectors
# basis1 is along the x=y=z line
basis1_raw = np.array([1, 1, 1])  # Along the x=y=z line
# basis2 and basis3 are orthogonal to basis1 and to each other
basis2_raw = np.array([2, -1, -1])  # Orthogonal to (1,1,1)
basis3_raw = np.array([0, 1, -1])   # Orthogonal to (1,1,1)

# Normalize the basis vectors
basis1 = basis1_raw / np.linalg.norm(basis1_raw)
basis2 = basis2_raw / np.linalg.norm(basis2_raw)
basis3 = basis3_raw / np.linalg.norm(basis3_raw)

# Verify orthogonality
print("Dot products to verify orthogonality:")
print(f"basis1 · basis2 = {np.dot(basis1, basis2):.10f}")
print(f"basis1 · basis3 = {np.dot(basis1, basis3):.10f}")
print(f"basis2 · basis3 = {np.dot(basis2, basis3):.10f}")

# Define parameters from gabor_bph.py
aVx = 1.0
aVa = 1.3
c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
x_shift = 0.1  # Shift in x direction

# Calculate x_prime using the formula from gabor_bph.py
x_prime = (aVa/aVx) / (aVa/aVx-1) * x_shift
print(f"x_prime = {x_prime}")

# Calculate r0 and x
r0 = x_prime * 1  # As in the original code
x = (2 * (x_prime - r0)) * 1  # As in the original code

# Since x would be 0 (because x_prime = r0), set a small value for x to visualize the points
x = 0.2

print(f"r0 = {r0}")
print(f"x = {x}")

# Define the origin point
origin = np.array([0, 0, 0])

# Define the point on the x=y=z line
r0_point = np.array([r0, r0, r0])

# Create three CI points 120° apart on a circle orthogonal to the x=y=z line
# These will be centered at the origin
theta1 = 0  # First point
theta2 = 2*np.pi/3  # Second point (120°)
theta3 = 4*np.pi/3  # Third point (240°)

# Calculate the CI points using the basis vectors and formulas from gabor_bph.py
# For n_CI = 0, 1, 2
ci_points = []
for n_CI in range(3):
    # Create R_0 triplet according to the formula
    R_0 = [r0+x+x if i == n_CI else r0-x for i in range(3)]
    ci_points.append(np.array(R_0))
    print(f"CI Point {n_CI+1}: R_0 = {R_0}, sum(R_0)/3 = {sum(R_0)/3}")

# Store all points for easier access
points = [origin, r0_point, ci_points[0], ci_points[1], ci_points[2]]
point_labels = ['Origin (0,0,0)', 'r0 Point', 'CI Point 1', 'CI Point 2', 'CI Point 3']
point_colors = ['black', 'purple', 'r', 'g', 'b']
point_markers = ['*', 'D', 'o', 'o', 'o']
point_sizes = [200, 150, 100, 100, 100]

# Calculate distances between CI points
print("\nDistances between CI points:")
for i in range(2, 5):
    for j in range(i+1, 5):
        dist = np.linalg.norm(points[i] - points[j])
        print(f"Distance between {point_labels[i]} and {point_labels[j]}: {dist}")

# Create 2D projections in a 2x2 grid
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig)
axes = []

# Create 2x2 grid of 2D subplots
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

# Plot the projections in each subplot
# XY projection
for i, point in enumerate(points):
    axes[0].scatter(point[0], point[1], color=point_colors[i], marker=point_markers[i], 
                   s=point_sizes[i], label=point_labels[i])

# Draw lines connecting r0_point to each CI point
for i in range(2, 5):
    axes[0].plot([r0_point[0], points[i][0]], [r0_point[1], points[i][1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to r0_point
axes[0].plot([origin[0], r0_point[0]], [origin[1], r0_point[1]], 
            color='purple', linestyle='--', alpha=0.7)

# Calculate radius for the circle around r0_point
radius = np.linalg.norm(np.array([ci_points[0][0] - r0, ci_points[0][1] - r0]))

# Plot the circle in the XY plane centered at r0_point
theta = np.linspace(0, 2*np.pi, 100)
circle_x = r0 + radius * np.cos(theta)
circle_y = r0 + radius * np.sin(theta)
axes[0].plot(circle_x, circle_y, 'k--', alpha=0.5, label='Circle')

axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('XY Projection')
axes[0].grid(True)
# Place legend outside the plot
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# XZ projection
for i, point in enumerate(points):
    axes[1].scatter(point[0], point[2], color=point_colors[i], marker=point_markers[i], 
                   s=point_sizes[i], label=point_labels[i])

# Draw lines connecting r0_point to each CI point
for i in range(2, 5):
    axes[1].plot([r0_point[0], points[i][0]], [r0_point[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to r0_point
axes[1].plot([origin[0], r0_point[0]], [origin[2], r0_point[2]], 
            color='purple', linestyle='--', alpha=0.7)

# Plot the circle in the XZ plane centered at r0_point
circle_x = r0 + radius * np.cos(theta)
circle_z = r0 + radius * np.sin(theta)
axes[1].plot(circle_x, circle_z, 'k--', alpha=0.5, label='Circle')

axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('XZ Projection')
axes[1].grid(True)
# Place legend outside the plot
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# YZ projection
for i, point in enumerate(points):
    axes[2].scatter(point[1], point[2], color=point_colors[i], marker=point_markers[i], 
                   s=point_sizes[i], label=point_labels[i])

# Draw lines connecting r0_point to each CI point
for i in range(2, 5):
    axes[2].plot([r0_point[1], points[i][1]], [r0_point[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to r0_point
axes[2].plot([origin[1], r0_point[1]], [origin[2], r0_point[2]], 
            color='purple', linestyle='--', alpha=0.7)

# Plot the circle in the YZ plane centered at r0_point
circle_y = r0 + radius * np.cos(theta)
circle_z = r0 + radius * np.sin(theta)
axes[2].plot(circle_y, circle_z, 'k--', alpha=0.5, label='Circle')

axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title('YZ Projection')
axes[2].grid(True)
# Place legend outside the plot
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Basis2 & Basis3 projection
# Project the points onto the basis2-basis3 plane
for i, point in enumerate(points):
    # Project point onto basis2-basis3 plane
    point_basis = np.array([np.dot(point, basis2), np.dot(point, basis3)])
    axes[3].scatter(point_basis[0], point_basis[1], color=point_colors[i], marker=point_markers[i], 
                   s=point_sizes[i], label=point_labels[i])

# Draw lines connecting r0_point to each CI point in basis space
r0_point_basis = np.array([np.dot(r0_point, basis2), np.dot(r0_point, basis3)])
for i in range(2, 5):
    point_basis = np.array([np.dot(points[i], basis2), np.dot(points[i], basis3)])
    axes[3].plot([r0_point_basis[0], point_basis[0]], [r0_point_basis[1], point_basis[1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to r0_point in basis space
origin_basis = np.array([np.dot(origin, basis2), np.dot(origin, basis3)])
axes[3].plot([origin_basis[0], r0_point_basis[0]], [origin_basis[1], r0_point_basis[1]], 
            color='purple', linestyle='--', alpha=0.7)

# Plot the circle in the basis2-basis3 plane centered at r0_point_basis
radius_basis = np.linalg.norm(np.array([np.dot(ci_points[0] - r0_point, basis2), np.dot(ci_points[0] - r0_point, basis3)]))
circle_basis_x = r0_point_basis[0] + radius_basis * np.cos(theta)
circle_basis_y = r0_point_basis[1] + radius_basis * np.sin(theta)
axes[3].plot(circle_basis_x, circle_basis_y, 'k--', alpha=0.5, label='Circle')

axes[3].set_xlabel('Basis2')
axes[3].set_ylabel('Basis3')
axes[3].set_title('Basis2-Basis3 Projection')
axes[3].grid(True)
# Place legend outside the plot
axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir_2d, 'ci_points_2d_projections.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot all points in 3D
for i, point in enumerate(points):
    ax.scatter(point[0], point[1], point[2], color=point_colors[i], marker=point_markers[i], 
              s=point_sizes[i], label=point_labels[i])

# Draw lines connecting r0_point to each CI point
for i in range(2, 5):
    ax.plot([r0_point[0], points[i][0]], [r0_point[1], points[i][1]], [r0_point[2], points[i][2]], 
           color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to r0_point
ax.plot([origin[0], r0_point[0]], [origin[1], r0_point[1]], [origin[2], r0_point[2]], 
        color='purple', linestyle='--', alpha=0.7)

# Plot the x=y=z line
line_points = np.array([-0.5, 1])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors with increased length for visibility
scale_factor = 0.3  # Scale factor to make vectors more visible
ax.quiver(origin[0], origin[1], origin[2], 
         scale_factor*basis1[0], scale_factor*basis1[1], scale_factor*basis1[2], 
         color='purple', linewidth=3, label='Basis1 (x=y=z)')
ax.quiver(origin[0], origin[1], origin[2], 
         scale_factor*basis2[0], scale_factor*basis2[1], scale_factor*basis2[2], 
         color='orange', linewidth=3, label='Basis2')
ax.quiver(origin[0], origin[1], origin[2], 
         scale_factor*basis3[0], scale_factor*basis3[1], scale_factor*basis3[2], 
         color='cyan', linewidth=3, label='Basis3')

# Add a circle in the plane perpendicular to the x=y=z line, centered at r0_point
# First, create points on a circle in the basis2-basis3 plane
circle_3d = np.zeros((3, len(theta)))
for i in range(len(theta)):
    # Create a point on the circle in the basis2-basis3 plane
    circle_point = radius * (np.cos(theta[i]) * basis2 + np.sin(theta[i]) * basis3)
    # Add the r0_point as the center
    circle_3d[:, i] = r0_point + circle_point

ax.plot(circle_3d[0, :], circle_3d[1, :], circle_3d[2, :], 
       'k--', alpha=0.5, linewidth=2, label='Circle')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of CI Points')
# Place legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

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

plt.tight_layout()
plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_xyz.png'), dpi=300, bbox_inches='tight')

# Create a second 3D plot with a different viewpoint to better show the arrangement
ax.view_init(elev=20, azim=45)  # Looking down the x=y=z line
plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_alternative_view.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to:\n - {plot_dir_2d}\n - {plot_dir_3d}")
