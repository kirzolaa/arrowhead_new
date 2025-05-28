#!/usr/bin/env python3
# Script to visualize three CI points arranged 120 degrees apart around the origin R0=(0,0,0)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# Create output directories with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('output', f'ci_points_origin_{timestamp}')
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

# Define the origin point
r0 = np.array([0, 0, 0])

# Define the parameter x for the CI points
x = 0.3  # This controls the size of the triangle formed by the CI points

# Create three CI points around the origin R0=(0,0,0)
ci_points = []
for n_CI in range(3):
    # Create R_0 triplet according to the formula: [r0+2x if i == n_CI else r0-x for i in range(3)]
    # Since r0 = 0, this simplifies to [2x if i == n_CI else -x for i in range(3)]
    R_0 = [2*x if i == n_CI else -x for i in range(3)]
    ci_points.append(np.array(R_0))
    print(f"CI Point {n_CI+1}: R_0 = {R_0}, sum(R_0)/3 = {sum(R_0)/3}")

# Store all points for easier access
points = [r0] + ci_points
point_labels = ['Origin R0=(0,0,0)', 'CI Point 1', 'CI Point 2', 'CI Point 3']
point_colors = ['black', 'r', 'g', 'b']
point_markers = ['*', 'o', 'o', 'o']
point_sizes = [200, 100, 100, 100]

# Calculate distances between CI points
print("\nDistances between CI points:")
for i in range(1, 4):
    for j in range(i+1, 4):
        dist = np.linalg.norm(points[i] - points[j])
        print(f"Distance between {point_labels[i]} and {point_labels[j]}: {dist}")

# Calculate distances from origin to CI points
print("\nDistances from origin to CI points:")
for i in range(1, 4):
    dist = np.linalg.norm(points[i] - points[0])
    print(f"Distance from {point_labels[0]} to {point_labels[i]}: {dist}")

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

# Draw lines connecting origin to each CI point
for i in range(1, 4):
    axes[0].plot([r0[0], points[i][0]], [r0[1], points[i][1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[0].plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], 'k-', alpha=0.5)
axes[0].plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'k-', alpha=0.5)
axes[0].plot([points[3][0], points[1][0]], [points[3][1], points[1][1]], 'k-', alpha=0.5)

# Calculate radius for the circle around the origin
radius = np.linalg.norm(np.array([ci_points[0][0], ci_points[0][1]]))

# Plot the circle in the XY plane centered at the origin
theta = np.linspace(0, 2*np.pi, 100)
circle_xy = np.zeros((2, len(theta)))
for i in range(len(theta)):
    point_3d = radius * (np.cos(theta[i]) * basis2 + np.sin(theta[i]) * basis3)
    circle_xy[0, i] = point_3d[0]
    circle_xy[1, i] = point_3d[1]

axes[0].plot(circle_xy[0, :], circle_xy[1, :], 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting origin to each CI point
for i in range(1, 4):
    axes[1].plot([r0[0], points[i][0]], [r0[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[1].plot([points[1][0], points[2][0]], [points[1][2], points[2][2]], 'k-', alpha=0.5)
axes[1].plot([points[2][0], points[3][0]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
axes[1].plot([points[3][0], points[1][0]], [points[3][2], points[1][2]], 'k-', alpha=0.5)

# Plot the circle in the XZ plane
circle_xz = np.zeros((2, len(theta)))
for i in range(len(theta)):
    point_3d = radius * (np.cos(theta[i]) * basis2 + np.sin(theta[i]) * basis3)
    circle_xz[0, i] = point_3d[0]
    circle_xz[1, i] = point_3d[2]

axes[1].plot(circle_xz[0, :], circle_xz[1, :], 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting origin to each CI point
for i in range(1, 4):
    axes[2].plot([r0[1], points[i][1]], [r0[2], points[i][2]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
axes[2].plot([points[1][1], points[2][1]], [points[1][2], points[2][2]], 'k-', alpha=0.5)
axes[2].plot([points[2][1], points[3][1]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
axes[2].plot([points[3][1], points[1][1]], [points[3][2], points[1][2]], 'k-', alpha=0.5)

# Plot the circle in the YZ plane
circle_yz = np.zeros((2, len(theta)))
for i in range(len(theta)):
    point_3d = radius * (np.cos(theta[i]) * basis2 + np.sin(theta[i]) * basis3)
    circle_yz[0, i] = point_3d[1]
    circle_yz[1, i] = point_3d[2]

axes[2].plot(circle_yz[0, :], circle_yz[1, :], 'k--', alpha=0.5, label='Circle')

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

# Draw lines connecting origin to each CI point in basis space
origin_basis = np.array([np.dot(r0, basis2), np.dot(r0, basis3)])
for i in range(1, 4):
    point_basis = np.array([np.dot(points[i], basis2), np.dot(points[i], basis3)])
    axes[3].plot([origin_basis[0], point_basis[0]], [origin_basis[1], point_basis[1]], 
                color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle in basis space
ci_points_basis = []
for i in range(1, 4):
    point_basis = np.array([np.dot(points[i], basis2), np.dot(points[i], basis3)])
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

# Draw lines connecting origin to each CI point
for i in range(1, 4):
    ax.plot([r0[0], points[i][0]], [r0[1], points[i][1]], [r0[2], points[i][2]], 
           color=point_colors[i], linestyle='-', alpha=0.7)

# Draw lines connecting the CI points to form a triangle
ax.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], [points[1][2], points[2][2]], 'k-', alpha=0.5)
ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], 'k-', alpha=0.5)
ax.plot([points[3][0], points[1][0]], [points[3][1], points[1][1]], [points[3][2], points[1][2]], 'k-', alpha=0.5)

# Plot the x=y=z line
line_points = np.array([-0.5, 0.5])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors with increased length for visibility
scale_factor = 0.3  # Scale factor to make vectors more visible
ax.quiver(r0[0], r0[1], r0[2], 
         scale_factor*basis1[0], scale_factor*basis1[1], scale_factor*basis1[2], 
         color='purple', linewidth=3, label='Basis1 (x=y=z)')
ax.quiver(r0[0], r0[1], r0[2], 
         scale_factor*basis2[0], scale_factor*basis2[1], scale_factor*basis2[2], 
         color='orange', linewidth=3, label='Basis2')
ax.quiver(r0[0], r0[1], r0[2], 
         scale_factor*basis3[0], scale_factor*basis3[1], scale_factor*basis3[2], 
         color='cyan', linewidth=3, label='Basis3')

# Add a circle in the plane perpendicular to the x=y=z line
circle_3d = np.zeros((3, len(theta)))
for i in range(len(theta)):
    # The circle is in the basis2-basis3 plane
    circle_3d[:, i] = radius * (np.cos(theta[i]) * basis2 + np.sin(theta[i]) * basis3)

ax.plot(circle_3d[0, :], circle_3d[1, :], circle_3d[2, :], 
       'k--', alpha=0.5, linewidth=2, label='Circle')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of CI Points around Origin R0=(0,0,0)')
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

print(f"\nPlots saved to:\n - {plot_dir_2d}\n - {plot_dir_3d}")
