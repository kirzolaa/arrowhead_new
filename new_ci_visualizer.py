#!/usr/bin/env python3
# Script to visualize three CI points arranged 120 degrees apart around a circle
# orthogonal to the x=y=z line, based on the mathematical formulas provided

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# Create output directories with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('output', f'three_ci_points_{timestamp}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_dir_2d = os.path.join(output_dir, 'plot_dir_2d')
plot_dir_3d = os.path.join(output_dir, 'plot_dir_3d')
if not os.path.exists(plot_dir_2d):
    os.makedirs(plot_dir_2d)
if not os.path.exists(plot_dir_3d):
    os.makedirs(plot_dir_3d)

print(f"Output directories created at:\n - {plot_dir_2d}\n - {plot_dir_3d}")

# Define the basis vectors for visualization
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

# Define parameters for the CI points
aVx = 1.0
aVa = 1.3
c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
x_shift = 0.1  # Shift in x direction

# Calculate x_prime using the formula from gabor_bph.py
x_prime = (aVa/aVx) / (aVa/aVx-1) * x_shift
print(f"x_prime = {x_prime}")

# For visualization purposes, we'll use two sets of CI points:
# 1. One set with r0 = x_prime (as in the original code)
# 2. Another set with r0 = (0,0,0) to show the origin-centered configuration

# Set 1: Original calculation
r0_original = x_prime * 1  # As in the original code
x_original = 0.2  # Set a non-zero value to ensure distinct CI points

# Set 2: Origin-centered
r0_origin = 0.0  # Origin point
x_origin = 0.2  # Same x value for comparison

print(f"r0_original = {r0_original}, x_original = {x_original}")
print(f"r0_origin = {r0_origin}, x_origin = {x_origin}")

# Define the two sets of CI points
ci_points_original = []
ci_points_origin = []

# Generate both sets of CI points
for n_CI in range(3):
    # Set 1: Original calculation
    R_0_original = [r0_original+x_original+x_original if i == n_CI else r0_original-x_original for i in range(3)]
    ci_points_original.append(R_0_original)
    print(f"Original CI Point {n_CI+1}: R_0 = {R_0_original}, sum(R_0)/3 = {sum(R_0_original)/3}")
    
    # Set 2: Origin-centered
    R_0_origin = [r0_origin+x_origin+x_origin if i == n_CI else r0_origin-x_origin for i in range(3)]
    ci_points_origin.append(R_0_origin)
    print(f"Origin CI Point {n_CI+1}: R_0 = {R_0_origin}, sum(R_0)/3 = {sum(R_0_origin)/3}")

# Convert the CI points to 3D coordinates for visualization
ci_points_original_3d = []
ci_points_origin_3d = []

# Convert original CI points
for R_0 in ci_points_original:
    # Each R_0 is a triplet [r0, r1, r2] that defines a point in 3D space
    point_3d = np.array([R_0[0], R_0[1], R_0[2]])
    ci_points_original_3d.append(point_3d)

# Convert origin-centered CI points
for R_0 in ci_points_origin:
    # Each R_0 is a triplet [r0, r1, r2] that defines a point in 3D space
    point_3d = np.array([R_0[0], R_0[1], R_0[2]])
    ci_points_origin_3d.append(point_3d)

# Calculate the distance between original CI points
print("\nDistances between original CI points:")
for i in range(len(ci_points_original_3d)):
    for j in range(i+1, len(ci_points_original_3d)):
        dist = np.linalg.norm(ci_points_original_3d[i] - ci_points_original_3d[j])
        print(f"Distance between Original CI Point {i+1} and {j+1}: {dist}")

# Calculate the distance between origin CI points
print("\nDistances between origin-centered CI points:")
for i in range(len(ci_points_origin_3d)):
    for j in range(i+1, len(ci_points_origin_3d)):
        dist = np.linalg.norm(ci_points_origin_3d[i] - ci_points_origin_3d[j])
        print(f"Distance between Origin CI Point {i+1} and {j+1}: {dist}")

# Create 2D projections in a 2x2 grid
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig)
axes = []

# Create 2x2 grid of 2D subplots
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

# Define colors for each CI point
colors = ['r', 'g', 'b']
labels = ['CI Point 1', 'CI Point 2', 'CI Point 3']

# Plot the projections in each subplot
# XY projection
# Plot original CI points
for i, point in enumerate(ci_points_original_3d):
    axes[0].scatter(point[0], point[1], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Plot origin-centered CI points
for i, point in enumerate(ci_points_origin_3d):
    axes[0].scatter(point[0], point[1], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Plot origin point
axes[0].scatter(0, 0, color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the circle in the XY plane for origin-centered points
radius_origin = np.linalg.norm(ci_points_origin_3d[0][0:2])
theta = np.linspace(0, 2*np.pi, 100)
axes[0].plot(radius_origin * np.cos(theta), radius_origin * np.sin(theta), 'k--', alpha=0.3, label='Circle (Origin CI)')

# Plot the circle in the XY plane for original points
radius_original = np.linalg.norm(ci_points_original_3d[0][0:2] - np.array([r0_original, r0_original]))
center_original = np.array([r0_original, r0_original])
axes[0].plot(center_original[0] + radius_original * np.cos(theta), 
             center_original[1] + radius_original * np.sin(theta), 
             'k:', alpha=0.3, label='Circle (Original CI)')

axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('XY Projection')
axes[0].grid(True)
axes[0].legend(fontsize='small')

# XZ projection
# Plot original CI points
for i, point in enumerate(ci_points_original_3d):
    axes[1].scatter(point[0], point[2], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Plot origin-centered CI points
for i, point in enumerate(ci_points_origin_3d):
    axes[1].scatter(point[0], point[2], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Plot origin point
axes[1].scatter(0, 0, color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the circle in the XZ plane for origin-centered points
radius_origin = np.linalg.norm(np.array([ci_points_origin_3d[0][0], ci_points_origin_3d[0][2]]))
axes[1].plot(radius_origin * np.cos(theta), radius_origin * np.sin(theta), 'k--', alpha=0.3)

# Plot the circle in the XZ plane for original points
radius_original = np.linalg.norm(np.array([ci_points_original_3d[0][0] - r0_original, ci_points_original_3d[0][2] - r0_original]))
center_original = np.array([r0_original, r0_original])
axes[1].plot(center_original[0] + radius_original * np.cos(theta), 
             center_original[1] + radius_original * np.sin(theta), 
             'k:', alpha=0.3)

axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('XZ Projection')
axes[1].grid(True)
axes[1].legend(fontsize='small')

# YZ projection
# Plot original CI points
for i, point in enumerate(ci_points_original_3d):
    axes[2].scatter(point[1], point[2], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Plot origin-centered CI points
for i, point in enumerate(ci_points_origin_3d):
    axes[2].scatter(point[1], point[2], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Plot origin point
axes[2].scatter(0, 0, color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the circle in the YZ plane for origin-centered points
radius_origin = np.linalg.norm(np.array([ci_points_origin_3d[0][1], ci_points_origin_3d[0][2]]))
axes[2].plot(radius_origin * np.cos(theta), radius_origin * np.sin(theta), 'k--', alpha=0.3)

# Plot the circle in the YZ plane for original points
radius_original = np.linalg.norm(np.array([ci_points_original_3d[0][1] - r0_original, ci_points_original_3d[0][2] - r0_original]))
center_original = np.array([r0_original, r0_original])
axes[2].plot(center_original[0] + radius_original * np.cos(theta), 
             center_original[1] + radius_original * np.sin(theta), 
             'k:', alpha=0.3)

axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title('YZ Projection')
axes[2].grid(True)
axes[2].legend(fontsize='small')

# Basis2 & Basis3 projection
# Project the CI points onto the basis2-basis3 plane
# Project original CI points
for i, point in enumerate(ci_points_original_3d):
    # Project point onto basis2-basis3 plane
    point_basis = np.array([np.dot(point, basis2), np.dot(point, basis3)])
    axes[3].scatter(point_basis[0], point_basis[1], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Project origin-centered CI points
for i, point in enumerate(ci_points_origin_3d):
    # Project point onto basis2-basis3 plane
    point_basis = np.array([np.dot(point, basis2), np.dot(point, basis3)])
    axes[3].scatter(point_basis[0], point_basis[1], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Project origin point
origin_basis = np.array([np.dot(np.zeros(3), basis2), np.dot(np.zeros(3), basis3)])
axes[3].scatter(origin_basis[0], origin_basis[1], color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the circle in the basis2-basis3 plane for origin-centered points
radius_origin = np.linalg.norm(np.array([np.dot(ci_points_origin_3d[0], basis2), np.dot(ci_points_origin_3d[0], basis3)]))
axes[3].plot(radius_origin * np.cos(theta), radius_origin * np.sin(theta), 'k--', alpha=0.3, label='Circle (Origin CI)')

# Plot the circle in the basis2-basis3 plane for original points
radius_original = np.linalg.norm(np.array([np.dot(ci_points_original_3d[0] - r0_original*np.ones(3), basis2), 
                                           np.dot(ci_points_original_3d[0] - r0_original*np.ones(3), basis3)]))
center_original = np.array([np.dot(r0_original*np.ones(3), basis2), np.dot(r0_original*np.ones(3), basis3)])
axes[3].plot(center_original[0] + radius_original * np.cos(theta), 
             center_original[1] + radius_original * np.sin(theta), 
             'k:', alpha=0.3, label='Circle (Original CI)')

axes[3].set_xlabel('Basis2')
axes[3].set_ylabel('Basis3')
axes[3].set_title('Basis2-Basis3 Projection')
axes[3].grid(True)
axes[3].legend(fontsize='small')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir_2d, 'ci_points_2d_projections.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create 3D plot for xyz coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original CI points in 3D
for i, point in enumerate(ci_points_original_3d):
    ax.scatter(point[0], point[1], point[2], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Plot the origin-centered CI points in 3D
for i, point in enumerate(ci_points_origin_3d):
    ax.scatter(point[0], point[1], point[2], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Plot the origin point
ax.scatter(0, 0, 0, color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the x=y=z line
line_points = np.array([-1, 1])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors with increased length for visibility
origin = np.zeros(3)
scale_factor = 0.5  # Scale factor to make vectors more visible
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis1[0], scale_factor*basis1[1], scale_factor*basis1[2], 
          color='purple', linewidth=3, label='Basis1 (x=y=z)')
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis2[0], scale_factor*basis2[1], scale_factor*basis2[2], 
          color='orange', linewidth=3, label='Basis2')
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis3[0], scale_factor*basis3[1], scale_factor*basis3[2], 
          color='cyan', linewidth=3, label='Basis3')

# Add circles in the plane perpendicular to the x=y=z line
# For origin-centered CI points
radius_origin = np.sqrt(np.dot(ci_points_origin_3d[0], basis2)**2 + np.dot(ci_points_origin_3d[0], basis3)**2)
theta = np.linspace(0, 2*np.pi, 100)
circle_origin_x = radius_origin * np.cos(theta)
circle_origin_y = radius_origin * np.sin(theta)

# Transform the origin-centered circle to 3D space
circle_origin_3d = np.zeros((3, len(theta)))
for i in range(len(theta)):
    # The circle is in the basis2-basis3 plane
    circle_origin_3d[:, i] = circle_origin_x[i] * basis2 + circle_origin_y[i] * basis3

ax.plot(circle_origin_3d[0, :], circle_origin_3d[1, :], circle_origin_3d[2, :], 
        'k--', alpha=0.5, linewidth=2, label='Circle for Origin CI Points')

# For original CI points
# First, find the center of the circle in 3D space (should be on the x=y=z line)
center_original_3d = np.array([r0_original, r0_original, r0_original])

# Calculate radius as the distance from any original CI point to this center,
# projected onto the basis2-basis3 plane
point_rel = ci_points_original_3d[0] - center_original_3d
radius_original = np.sqrt(np.dot(point_rel, basis2)**2 + np.dot(point_rel, basis3)**2)

# Create the circle
circle_original_x = radius_original * np.cos(theta)
circle_original_y = radius_original * np.sin(theta)

# Transform to 3D space
circle_original_3d = np.zeros((3, len(theta)))
for i in range(len(theta)):
    # The circle is in the basis2-basis3 plane, centered at center_original_3d
    circle_original_3d[:, i] = center_original_3d + circle_original_x[i] * basis2 + circle_original_y[i] * basis3

ax.plot(circle_original_3d[0, :], circle_original_3d[1, :], circle_original_3d[2, :], 
        'k:', alpha=0.5, linewidth=2, label='Circle for Original CI Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of CI Points')
ax.legend()

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

plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_xyz.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a second 3D plot with a different viewpoint to better show the arrangement
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original CI points in 3D
for i, point in enumerate(ci_points_original_3d):
    ax.scatter(point[0], point[1], point[2], color=colors[i], marker='o', s=100, label=f'Original {labels[i]}')

# Plot the origin-centered CI points in 3D
for i, point in enumerate(ci_points_origin_3d):
    ax.scatter(point[0], point[1], point[2], color=colors[i], marker='s', s=100, label=f'Origin {labels[i]}')

# Plot the origin point
ax.scatter(0, 0, 0, color='black', marker='*', s=200, label='Origin (0,0,0)')

# Plot the x=y=z line
line_points = np.array([-1, 1])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors with increased length for visibility
origin = np.zeros(3)
scale_factor = 0.5  # Scale factor to make vectors more visible
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis1[0], scale_factor*basis1[1], scale_factor*basis1[2], 
          color='purple', linewidth=3, label='Basis1 (x=y=z)')
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis2[0], scale_factor*basis2[1], scale_factor*basis2[2], 
          color='orange', linewidth=3, label='Basis2')
ax.quiver(origin[0], origin[1], origin[2], 
          scale_factor*basis3[0], scale_factor*basis3[1], scale_factor*basis3[2], 
          color='cyan', linewidth=3, label='Basis3')

# Add the circles in the plane perpendicular to the x=y=z line
ax.plot(circle_origin_3d[0, :], circle_origin_3d[1, :], circle_origin_3d[2, :], 
        'k--', alpha=0.5, linewidth=2, label='Circle for Origin CI Points')
ax.plot(circle_original_3d[0, :], circle_original_3d[1, :], circle_original_3d[2, :], 
        'k:', alpha=0.5, linewidth=2, label='Circle for Original CI Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of CI Points (Alternative View)')
ax.legend()

# Set a different viewpoint
ax.view_init(elev=20, azim=45)  # Looking down the x=y=z line

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

plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_alternative_view.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to:\n - {plot_dir_2d}\n - {plot_dir_3d}")