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

# Define the basis vectors orthogonal to the x=y=z line
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
r0 = np.array([0, 0, 0])  # Starting point at origin
a = 5  # Parameter from the equations
a_per_a_minus1 = a / (a - 1)
x_shift = np.array([1, 1, 1])  # Shift along the x=y=z line
x_prime = (a / (a - 1)) * x_shift  # From equation #05

# Calculate delta based on equation #11
# delta = 3 * (x_prime - r0)
delta_magnitude = np.linalg.norm(x_prime - r0)

print(f"x_prime = {x_prime}")
print(f"delta_magnitude = {delta_magnitude}")

# Instead of rotating r1 and r2, we'll create three different CI points directly
# Each will be 120° apart in the plane perpendicular to the x=y=z line

# For the first CI point, we'll use the original equations
# The delta will be along basis2 direction
delta1 = 3 * delta_magnitude * basis2
r1_point1 = r0 - 2 * delta1  # Using equation #12 form: r1 = r0 - 2 * (x_prime - r0)
r2_point1 = r0 + 4 * delta1  # Using equation #13 form: r2 = r0 + 4 * (x_prime - r0)

# For the second CI point, rotate 120° around the x=y=z line
# This means the delta will be in a direction 120° from basis2 in the basis2-basis3 plane
theta_120 = 2*np.pi/3  # 120 degrees in radians
delta2 = 3 * delta_magnitude * (np.cos(theta_120) * basis2 + np.sin(theta_120) * basis3)
r1_point2 = r0 - 2 * delta2
r2_point2 = r0 + 4 * delta2

# For the third CI point, rotate 240° around the x=y=z line
theta_240 = 4*np.pi/3  # 240 degrees in radians
delta3 = 3 * delta_magnitude * (np.cos(theta_240) * basis2 + np.sin(theta_240) * basis3)
r1_point3 = r0 - 2 * delta3
r2_point3 = r0 + 4 * delta3

# We don't need the rotation matrix function anymore since we're directly calculating the points

# Store all CI points for easier access
ci_points = [
    (r0, r1_point1, r2_point1),  # First CI point
    (r0, r1_point2, r2_point2),  # Second CI point (120° rotation)
    (r0, r1_point3, r2_point3)   # Third CI point (240° rotation)
]

# Print the CI points
print("\nCI Points:")
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    print(f"CI Point {i+1}:")
    print(f"  r0 = {r0_i}")
    print(f"  r1 = {r1_i}")
    print(f"  r2 = {r2_i}")
    # Calculate distance using equation #15: d_CI = 2 * sqrt(6) * (x_prime-r0)
    d_ci = 2 * np.sqrt(6) * np.linalg.norm(x_prime - r0_i)
    print(f"  d_CI = {d_ci}")

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
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    axes[0].scatter(r0_i[0], r0_i[1], color=colors[i], marker='o', s=100, label=f'{labels[i]} (r0)')
    axes[0].scatter(r1_i[0], r1_i[1], color=colors[i], marker='^', s=100, label=f'{labels[i]} (r1)')
    axes[0].scatter(r2_i[0], r2_i[1], color=colors[i], marker='s', s=100, label=f'{labels[i]} (r2)')
    
    # Draw lines connecting the points
    axes[0].plot([r0_i[0], r1_i[0]], [r0_i[1], r1_i[1]], color=colors[i], linestyle='-', alpha=0.7)
    axes[0].plot([r0_i[0], r2_i[0]], [r0_i[1], r2_i[1]], color=colors[i], linestyle='-', alpha=0.7)
    axes[0].plot([r1_i[0], r2_i[0]], [r1_i[1], r2_i[1]], color=colors[i], linestyle='-', alpha=0.7)

axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('XY Projection')
axes[0].grid(True)

# XZ projection
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    axes[1].scatter(r0_i[0], r0_i[2], color=colors[i], marker='o', s=100)
    axes[1].scatter(r1_i[0], r1_i[2], color=colors[i], marker='^', s=100)
    axes[1].scatter(r2_i[0], r2_i[2], color=colors[i], marker='s', s=100)
    
    # Draw lines connecting the points
    axes[1].plot([r0_i[0], r1_i[0]], [r0_i[2], r1_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    axes[1].plot([r0_i[0], r2_i[0]], [r0_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    axes[1].plot([r1_i[0], r2_i[0]], [r1_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)

axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('XZ Projection')
axes[1].grid(True)

# YZ projection
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    axes[2].scatter(r0_i[1], r0_i[2], color=colors[i], marker='o', s=100)
    axes[2].scatter(r1_i[1], r1_i[2], color=colors[i], marker='^', s=100)
    axes[2].scatter(r2_i[1], r2_i[2], color=colors[i], marker='s', s=100)
    
    # Draw lines connecting the points
    axes[2].plot([r0_i[1], r1_i[1]], [r0_i[2], r1_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    axes[2].plot([r0_i[1], r2_i[1]], [r0_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    axes[2].plot([r1_i[1], r2_i[1]], [r1_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)

axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title('YZ Projection')
axes[2].grid(True)

# Basis2 & Basis3 projection
# Project the CI points onto the basis2-basis3 plane
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    # Project points onto basis2-basis3 plane
    r0_basis = np.array([np.dot(r0_i, basis2), np.dot(r0_i, basis3)])
    r1_basis = np.array([np.dot(r1_i, basis2), np.dot(r1_i, basis3)])
    r2_basis = np.array([np.dot(r2_i, basis2), np.dot(r2_i, basis3)])
    
    axes[3].scatter(r0_basis[0], r0_basis[1], color=colors[i], marker='o', s=100)
    axes[3].scatter(r1_basis[0], r1_basis[1], color=colors[i], marker='^', s=100)
    axes[3].scatter(r2_basis[0], r2_basis[1], color=colors[i], marker='s', s=100)
    
    # Draw lines connecting the points
    axes[3].plot([r0_basis[0], r1_basis[0]], [r0_basis[1], r1_basis[1]], color=colors[i], linestyle='-', alpha=0.7)
    axes[3].plot([r0_basis[0], r2_basis[0]], [r0_basis[1], r2_basis[1]], color=colors[i], linestyle='-', alpha=0.7)
    axes[3].plot([r1_basis[0], r2_basis[0]], [r1_basis[1], r2_basis[1]], color=colors[i], linestyle='-', alpha=0.7)

axes[3].set_xlabel('Basis2')
axes[3].set_ylabel('Basis3')
axes[3].set_title('Basis2-Basis3 Projection')
axes[3].grid(True)

# Add a single legend for all subplots
handles, labels = [], []
for i in range(3):
    handles.append(plt.Line2D([0], [0], marker='o', color=colors[i], linestyle='None', markersize=10))
    labels.append(f'CI Point {i+1} (r0)')
    handles.append(plt.Line2D([0], [0], marker='^', color=colors[i], linestyle='None', markersize=10))
    labels.append(f'CI Point {i+1} (r1)')
    handles.append(plt.Line2D([0], [0], marker='s', color=colors[i], linestyle='None', markersize=10))
    labels.append(f'CI Point {i+1} (r2)')

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig(os.path.join(plot_dir_2d, 'ci_points_2d_projections.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create 3D plot for xyz coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the CI points in 3D
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    ax.scatter(r0_i[0], r0_i[1], r0_i[2], color=colors[i], marker='o', s=100, label=f'CI Point {i+1} (r0)')
    ax.scatter(r1_i[0], r1_i[1], r1_i[2], color=colors[i], marker='^', s=100, label=f'CI Point {i+1} (r1)')
    ax.scatter(r2_i[0], r2_i[1], r2_i[2], color=colors[i], marker='s', s=100, label=f'CI Point {i+1} (r2)')
    
    # Draw lines connecting the points
    ax.plot([r0_i[0], r1_i[0]], [r0_i[1], r1_i[1]], [r0_i[2], r1_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    ax.plot([r0_i[0], r2_i[0]], [r0_i[1], r2_i[1]], [r0_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)
    ax.plot([r1_i[0], r2_i[0]], [r1_i[1], r2_i[1]], [r1_i[2], r2_i[2]], color=colors[i], linestyle='-', alpha=0.7)

# Plot the x=y=z line
line_points = np.array([-5, 5])
ax.plot(line_points, line_points, line_points, 'k--', label='x=y=z line', alpha=0.5)

# Plot the basis vectors
origin = np.zeros(3)
ax.quiver(origin[0], origin[1], origin[2], basis1[0], basis1[1], basis1[2], color='purple', label='Basis1 (x=y=z)')
ax.quiver(origin[0], origin[1], origin[2], basis2[0], basis2[1], basis2[2], color='orange', label='Basis2')
ax.quiver(origin[0], origin[1], origin[2], basis3[0], basis3[1], basis3[2], color='cyan', label='Basis3')

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

# Create 3D plot for basis coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert points to basis coordinates
for i, (r0_i, r1_i, r2_i) in enumerate(ci_points):
    # Convert to basis coordinates
    r0_basis_coords = np.array([np.dot(r0_i, basis1), np.dot(r0_i, basis2), np.dot(r0_i, basis3)])
    r1_basis_coords = np.array([np.dot(r1_i, basis1), np.dot(r1_i, basis2), np.dot(r1_i, basis3)])
    r2_basis_coords = np.array([np.dot(r2_i, basis1), np.dot(r2_i, basis2), np.dot(r2_i, basis3)])
    
    ax.scatter(r0_basis_coords[0], r0_basis_coords[1], r0_basis_coords[2], color=colors[i], marker='o', s=100, label=f'CI Point {i+1} (r0)')
    ax.scatter(r1_basis_coords[0], r1_basis_coords[1], r1_basis_coords[2], color=colors[i], marker='^', s=100, label=f'CI Point {i+1} (r1)')
    ax.scatter(r2_basis_coords[0], r2_basis_coords[1], r2_basis_coords[2], color=colors[i], marker='s', s=100, label=f'CI Point {i+1} (r2)')
    
    # Draw lines connecting the points
    ax.plot([r0_basis_coords[0], r1_basis_coords[0]], 
            [r0_basis_coords[1], r1_basis_coords[1]], 
            [r0_basis_coords[2], r1_basis_coords[2]], color=colors[i], linestyle='-', alpha=0.7)
    ax.plot([r0_basis_coords[0], r2_basis_coords[0]], 
            [r0_basis_coords[1], r2_basis_coords[1]], 
            [r0_basis_coords[2], r2_basis_coords[2]], color=colors[i], linestyle='-', alpha=0.7)
    ax.plot([r1_basis_coords[0], r2_basis_coords[0]], 
            [r1_basis_coords[1], r2_basis_coords[1]], 
            [r1_basis_coords[2], r2_basis_coords[2]], color=colors[i], linestyle='-', alpha=0.7)

# Plot the basis vectors
origin = np.zeros(3)
ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='purple', label='Basis1 direction')
ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='orange', label='Basis2 direction')
ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='cyan', label='Basis3 direction')

ax.set_xlabel('Basis1')
ax.set_ylabel('Basis2')
ax.set_zlabel('Basis3')
ax.set_title('3D Visualization of CI Points in Basis Coordinates')
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

plt.savefig(os.path.join(plot_dir_3d, 'ci_points_3d_basis.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to:\n - {plot_dir_2d}\n - {plot_dir_3d}")
