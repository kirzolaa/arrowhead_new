#!/usr/bin/env python3
# Script to visualize CI points as specified in gabor_bph.py with R0 = (0.433, 0.433, 0.433)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# Create output directories with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('output', f'ci_points_actual_{timestamp}')
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
# This matches what's done in gabor_bph.py where d = 0.001 is used
x = 0.001

print(f"r0 = {r0}")
print(f"x = {x}")

# Define the origin point (0,0,0)
origin = np.array([0, 0, 0])

# Define the R0 point on the x=y=z line
r0_point = np.array([r0, r0, r0])

# Create three CI points according to the formula in gabor_bph.py
ci_points = []
for n_CI in range(3):
    # Create R_0 triplet according to the formula
    R_0 = [r0+x+x if i == n_CI else r0-x for i in range(3)]
    ci_points.append(np.array(R_0))
    print(f"CI Point {n_CI+1}: R_0 = {R_0}, sum(R_0)/3 = {sum(R_0)/3}")

# Store all points for easier access
points = [origin, r0_point] + ci_points
point_labels = ['Origin (0,0,0)', 'R0 Point', 'CI Point 1', 'CI Point 2', 'CI Point 3']
point_colors = ['black', 'purple', 'r', 'g', 'b']
point_markers = ['*', 'D', 'o', 'o', 'o']
point_sizes = [200, 150, 100, 100, 100]

# Calculate distances between CI points
print("\nDistances between CI points:")
for i in range(2, 5):
    for j in range(i+1, 5):
        dist = np.linalg.norm(points[i] - points[j])
        print(f"Distance between {point_labels[i]} and {point_labels[j]}: {dist}")

# Calculate distances from R0 point to CI points
print("\nDistances from R0 point to CI points:")
for i in range(2, 5):
    dist = np.linalg.norm(points[i] - points[1])
    print(f"Distance from {point_labels[1]} to {point_labels[i]}: {dist}")

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

# Draw lines connecting R0 point to each CI point
for i in range(2, 5):
    axes[0].plot([r0_point[0], points[i][0]], [r0_point[1], points[i][1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to R0 point
axes[0].plot([origin[0], r0_point[0]], [origin[1], r0_point[1]], 
            color='purple', linestyle='--', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[0].plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'k-', alpha=0.5)
axes[0].plot([points[3][0], points[4][0]], [points[3][1], points[4][1]], 'k-', alpha=0.5)
axes[0].plot([points[4][0], points[2][0]], [points[4][1], points[2][1]], 'k-', alpha=0.5)

# Calculate radius for the circle around R0 point
radius = np.linalg.norm(np.array([ci_points[0][0] - r0, ci_points[0][1] - r0]))

# Plot the circle in the XY plane centered at R0 point
theta = np.linspace(0, 2*np.pi, 100)
# Create points on the circle in the basis2-basis3 plane
circle_points = []
for t in theta:
    # Create a point on the circle in the basis2-basis3 plane
    circle_point = radius * (np.cos(t) * basis2 + np.sin(t) * basis3)
    # Add the R0 point as the center
    circle_3d_point = r0_point + circle_point
    circle_points.append(circle_3d_point)
    
circle_points = np.array(circle_points)
axes[0].plot(circle_points[:, 0], circle_points[:, 1], 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting R0 point to each CI point
for i in range(2, 5):
    axes[1].plot([r0_point[0], points[i][0]], [r0_point[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to R0 point
axes[1].plot([origin[0], r0_point[0]], [origin[2], r0_point[2]], 
            color='purple', linestyle='--', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[1].plot([points[2][0], points[3][0]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
axes[1].plot([points[3][0], points[4][0]], [points[3][2], points[4][2]], 'k-', alpha=0.5)
axes[1].plot([points[4][0], points[2][0]], [points[4][2], points[2][2]], 'k-', alpha=0.5)

# Plot the circle in the XZ plane
axes[1].plot(circle_points[:, 0], circle_points[:, 2], 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting R0 point to each CI point
for i in range(2, 5):
    axes[2].plot([r0_point[1], points[i][1]], [r0_point[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to R0 point
axes[2].plot([origin[1], r0_point[1]], [origin[2], r0_point[2]], 
            color='purple', linestyle='--', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[2].plot([points[2][1], points[3][1]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
axes[2].plot([points[3][1], points[4][1]], [points[3][2], points[4][2]], 'k-', alpha=0.5)
axes[2].plot([points[4][1], points[2][1]], [points[4][2], points[2][2]], 'k-', alpha=0.5)

# Plot the circle in the YZ plane
axes[2].plot(circle_points[:, 1], circle_points[:, 2], 'k--', alpha=0.5, label='Circle')

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
    point_basis = np.array([np.dot(point - r0_point, basis2), np.dot(point - r0_point, basis3)])
    axes[3].scatter(point_basis[0], point_basis[1], color=point_colors[i], marker=point_markers[i], 
                   s=point_sizes[i], label=point_labels[i])

# Draw lines connecting R0 point to each CI point in basis space
r0_point_basis = np.array([0, 0])  # R0 point is the origin in this basis
for i in range(2, 5):
    point_basis = np.array([np.dot(points[i] - r0_point, basis2), np.dot(points[i] - r0_point, basis3)])
    axes[3].plot([r0_point_basis[0], point_basis[0]], [r0_point_basis[1], point_basis[1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle in basis space
ci_points_basis = []
for i in range(2, 5):
    point_basis = np.array([np.dot(points[i] - r0_point, basis2), np.dot(points[i] - r0_point, basis3)])
    ci_points_basis.append(point_basis)

axes[3].plot([ci_points_basis[0][0], ci_points_basis[1][0]], [ci_points_basis[0][1], ci_points_basis[1][1]], 'k-', alpha=0.5)
axes[3].plot([ci_points_basis[1][0], ci_points_basis[2][0]], [ci_points_basis[1][1], ci_points_basis[2][1]], 'k-', alpha=0.5)
axes[3].plot([ci_points_basis[2][0], ci_points_basis[0][0]], [ci_points_basis[2][1], ci_points_basis[0][1]], 'k-', alpha=0.5)

# Plot the circle in the basis2-basis3 plane
axes[3].plot(radius * np.cos(theta), radius * np.sin(theta), 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting R0 point to each CI point
for i in range(2, 5):
    ax.plot([r0_point[0], points[i][0]], [r0_point[1], points[i][1]], [r0_point[2], points[i][2]], 
           color=point_colors[i], linestyle='-', alpha=0.7)

# Draw line from origin to R0 point
ax.plot([origin[0], r0_point[0]], [origin[1], r0_point[1]], [origin[2], r0_point[2]], 
        color='purple', linestyle='--', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
ax.plot([points[3][0], points[4][0]], [points[3][1], points[4][1]], [points[3][2], points[4][2]], 'k-', alpha=0.5)
ax.plot([points[4][0], points[2][0]], [points[4][1], points[2][1]], [points[4][2], points[2][2]], 'k-', alpha=0.5)

# Plot the x=y=z line
line_points = np.array([-0.1, 0.8])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors with increased length for visibility
scale_factor = 0.2  # Scale factor to make vectors more visible
ax.quiver(r0_point[0], r0_point[1], r0_point[2], 
         scale_factor*basis1[0], scale_factor*basis1[1], scale_factor*basis1[2], 
         color='purple', linewidth=3, label='Basis1 (x=y=z)')
ax.quiver(r0_point[0], r0_point[1], r0_point[2], 
         scale_factor*basis2[0], scale_factor*basis2[1], scale_factor*basis2[2], 
         color='orange', linewidth=3, label='Basis2')
ax.quiver(r0_point[0], r0_point[1], r0_point[2], 
         scale_factor*basis3[0], scale_factor*basis3[1], scale_factor*basis3[2], 
         color='cyan', linewidth=3, label='Basis3')

# Add a circle in the plane perpendicular to the x=y=z line, centered at R0 point
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
       'k--', alpha=0.5, linewidth=2, label='Circle')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of CI Points around R0=({:.3f}, {:.3f}, {:.3f})'.format(r0, r0, r0))
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

# Create a second 3D plot with a different viewpoint looking down the x=y=z line
ax.view_init(elev=20, azim=45)  # Looking down the x=y=z line
plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_alternative_view.png'), dpi=300, bbox_inches='tight')

# Create a third 3D plot with a view directly along the x=y=z line
ax.view_init(elev=35.264, azim=45)  # Exact angle to look along x=y=z line
plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_along_xyz_line.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a special 2D plot showing just the basis2-basis3 projection with the CI points
# This will be similar to the plot in your berry_phase_corrected_run_n_minus_1/vectors directory
fig, ax = plt.subplots(figsize=(8, 8))

# Project the points onto the basis2-basis3 plane
for i, point in enumerate(points):
    # Project point onto basis2-basis3 plane
    point_basis = np.array([np.dot(point - r0_point, basis2), np.dot(point - r0_point, basis3)])
    ax.scatter(point_basis[0], point_basis[1], color=point_colors[i], marker=point_markers[i], 
              s=point_sizes[i], label=point_labels[i])

# Draw lines connecting R0 point to each CI point in basis space
r0_point_basis = np.array([0, 0])  # R0 point is the origin in this basis
for i in range(2, 5):
    point_basis = np.array([np.dot(points[i] - r0_point, basis2), np.dot(points[i] - r0_point, basis3)])
    ax.plot([r0_point_basis[0], point_basis[0]], [r0_point_basis[1], point_basis[1]], 
           color=point_colors[i], linestyle='-', alpha=0.7)

# Plot the circle in the basis2-basis3 plane
ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'r-', alpha=0.8, linewidth=2, label='Circle')

ax.set_xlabel('Basis Vector 1 Direction')
ax.set_ylabel('Basis Vector 2 Direction')
ax.set_title('Perfect Circle in Plane Orthogonal to x=y=z')
ax.grid(True)
ax.legend()

# Make the plot square
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir_2d, 'perfect_circle_orthogonal_plane.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to:\n - {plot_dir_2d}\n - {plot_dir_3d}")
