# script to visualize x_0 plus/minus delta, in 2d projections to xy, yz, xz
# add a basis1, basis2 basis3 vectors as well, plot the 2d projections in a 2x2 grid
# the basis vectors should be normalized
# we need also a 3d plot for both the xyz, and the three basises as well
# x_0' = a/(a-1)*x_0
# x_1 = x_0' +- delta
# x_2 = x_0' -+ delta
# thus we would have 3 cases:
#   - (0,0,0) --> trivial (we expect the lines in the projections as well as the xyz 3d plots, that we have an intersection @ (0,0,0)
#   - (x_1, x_2, x_2)
#   - (x_2, x_1, x_1)
# we could step delta from -1...+1, and plot the 4 projections and the 2 3d plots
# example basis2 = (2,-1,1), basis3 = (0,1,-1), since we are orthogonal to the (1,1,1) line
# basis1 could be calcuélated and set as an orthogonal (orthonormal) basis1 to basis2 and basis3
# basis1 = basis2 x basis3
# basis1 = basis1 / ||basis1||
# basis2 = basis2 / ||basis2||
# basis3 = basis3 / ||basis3||
#thus we have such lines in the xyz and basis1, basis2, basis3 3d plots, as
#   - (0,0,0) --> trivial (we expect the lines in the projections as well as the xyz 3d plots, that we have an intersection @ (0,0,0)
#   - (x_1, x_2, x_2) --> (x_0' +/- delta, x_0' -/+ delta, x_0' -/+ delta) (3 lines)
#   - (x_2, x_1, x_1) --> (x_0' -/+ delta, x_0' +/- delta, x_0' +/- delta) (3 lines)
# line1 = (x_0' + delta, x_0' - delta, x_0' - delta)
# line2 = (x_0' - delta, x_0' + delta, x_0' + delta)
# we expect the lines in the projections as well as the xyz 3d plots, that we have an intersection @ (0,0,0)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os

# define the basis vectors
basis2 = np.array([2, 1, 1])
basis3 = np.array([0, 1, -1])
basis1 = np.cross(basis2, basis3)
basis1 = basis1 / np.linalg.norm(basis1)
basis2 = basis2 / np.linalg.norm(basis2)
basis3 = basis3 / np.linalg.norm(basis3)

# define the x_0 vector
x_0 = np.array([1, 1, 1]) # non-origin point to demonstrate different line behavior

a = 5 # we assume we have a = 2 and aVa and aVx are the same, x_shift = 0

a_per_a_minus1 = a / (a - 1)
x_0_ = a_per_a_minus1 * x_0

delta = np.linspace(-1, 1, 100, endpoint=True)

# x_line1 = (x_0' + delta, x_0' - delta, x_0' - delta)
# x_line2 = (x_0' - delta, x_0' + delta, x_0' + delta)
# we dont need to do more, since we are using delta as an array, so we have the 2nd and the 3rd cases wrapped together

x_line1 = np.array([x_0_[0] + delta, x_0_[1] - delta, x_0_[2] - delta])
x_line2 = np.array([x_0_[0] - delta, x_0_[1] + delta, x_0_[2] + delta])

# create an output directory with timestamp to keep track of different runs
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('output', f'run_{timestamp}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create a plot_dir_2d and 3d as well
plot_dir_2d = os.path.join(output_dir, 'plot_dir_2d')
plot_dir_3d = os.path.join(output_dir, 'plot_dir_3d')
if not os.path.exists(plot_dir_2d):
    os.makedirs(plot_dir_2d)
if not os.path.exists(plot_dir_3d):
    os.makedirs(plot_dir_3d)

print(f"Output directories created at:\n - {plot_dir_2d}\n - {plot_dir_3d}")

# Find indices for delta = -1, 0, 1 for highlighting specific cases
delta_minus1_idx = np.argmin(np.abs(delta - (-1)))
delta_zero_idx = np.argmin(np.abs(delta))
delta_plus1_idx = np.argmin(np.abs(delta - 1))

# Extract the specific points for key delta values
# Delta = 0 (intersection point)
case_delta0_line1 = np.array([x_line1[0][delta_zero_idx], x_line1[1][delta_zero_idx], x_line1[2][delta_zero_idx]])
case_delta0_line2 = np.array([x_line2[0][delta_zero_idx], x_line2[1][delta_zero_idx], x_line2[2][delta_zero_idx]])

# Delta = 1 (upper bound)
case_delta1_line1 = np.array([x_line1[0][delta_plus1_idx], x_line1[1][delta_plus1_idx], x_line1[2][delta_plus1_idx]])
case_delta1_line2 = np.array([x_line2[0][delta_plus1_idx], x_line2[1][delta_plus1_idx], x_line2[2][delta_plus1_idx]])

# Delta = -1 (lower bound)
case_deltaN1_line1 = np.array([x_line1[0][delta_minus1_idx], x_line1[1][delta_minus1_idx], x_line1[2][delta_minus1_idx]])
case_deltaN1_line2 = np.array([x_line2[0][delta_minus1_idx], x_line2[1][delta_minus1_idx], x_line2[2][delta_minus1_idx]])

# Print the values for reference
print(f"\nLine equations:")
print(f"Line 1: (x_0' + delta, x_0' - delta, x_0' - delta)")
print(f"Line 2: (x_0' - delta, x_0' + delta, x_0' + delta)")
print(f"\nKey points on the lines:")
print(f"Delta = -1: Line 1 = {case_deltaN1_line1}, Line 2 = {case_deltaN1_line2}")
print(f"Delta = 0: Line 1 = {case_delta0_line1}, Line 2 = {case_delta0_line2}")
print(f"Delta = 1: Line 1 = {case_delta1_line1}, Line 2 = {case_delta1_line2}")
print(f"\nDelta varies continuously from -1 to 1, creating the full lines")

# plot the lines in 2d projections in 2x2 grid subplots, as xy, xz, yz, basis2&basis3
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig)
axes = []

# Create 2x2 grid of 2D subplots
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

# Plot the projections in each subplot
# XY projection
axes[0].plot(x_line1[0], x_line1[1], 'r', label='Line 1', alpha=0.7)
axes[0].plot(x_line2[0], x_line2[1], 'g', label='Line 2', alpha=0.7)

# Highlight key delta values
# Delta = -1
axes[0].scatter(case_deltaN1_line1[0], case_deltaN1_line1[1], color='blue', s=100, marker='<', label='δ=-1 (Line 1)')
axes[0].scatter(case_deltaN1_line2[0], case_deltaN1_line2[1], color='blue', s=100, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
axes[0].scatter(case_delta0_line1[0], case_delta0_line1[1], color='black', s=100, marker='o', label='δ=0 (intersection)')

# Delta = 1
axes[0].scatter(case_delta1_line1[0], case_delta1_line1[1], color='purple', s=100, marker='>', label='δ=1 (Line 1)')
axes[0].scatter(case_delta1_line2[0], case_delta1_line2[1], color='purple', s=100, marker='>', label='δ=1 (Line 2)')

axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('XY Projection')
axes[0].legend()
axes[0].grid(True)

# XZ projection
axes[1].plot(x_line1[0], x_line1[2], 'r', label='Line 1', alpha=0.7)
axes[1].plot(x_line2[0], x_line2[2], 'g', label='Line 2', alpha=0.7)

# Highlight key delta values
# Delta = -1
axes[1].scatter(case_deltaN1_line1[0], case_deltaN1_line1[2], color='blue', s=100, marker='<', label='δ=-1 (Line 1)')
axes[1].scatter(case_deltaN1_line2[0], case_deltaN1_line2[2], color='blue', s=100, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
axes[1].scatter(case_delta0_line1[0], case_delta0_line1[2], color='black', s=100, marker='o', label='δ=0 (intersection)')

# Delta = 1
axes[1].scatter(case_delta1_line1[0], case_delta1_line1[2], color='purple', s=100, marker='>', label='δ=1 (Line 1)')
axes[1].scatter(case_delta1_line2[0], case_delta1_line2[2], color='purple', s=100, marker='>', label='δ=1 (Line 2)')

axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('XZ Projection')
axes[1].legend()
axes[1].grid(True)

# YZ projection
axes[2].plot(x_line1[1], x_line1[2], 'r', label='Line 1', alpha=0.7)
axes[2].plot(x_line2[1], x_line2[2], 'g', label='Line 2', alpha=0.7)

# Highlight key delta values
# Delta = -1
axes[2].scatter(case_deltaN1_line1[1], case_deltaN1_line1[2], color='blue', s=100, marker='<', label='δ=-1 (Line 1)')
axes[2].scatter(case_deltaN1_line2[1], case_deltaN1_line2[2], color='blue', s=100, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
axes[2].scatter(case_delta0_line1[1], case_delta0_line1[2], color='black', s=100, marker='o', label='δ=0 (intersection)')

# Delta = 1
axes[2].scatter(case_delta1_line1[1], case_delta1_line1[2], color='purple', s=100, marker='>', label='δ=1 (Line 1)')
axes[2].scatter(case_delta1_line2[1], case_delta1_line2[2], color='purple', s=100, marker='>', label='δ=1 (Line 2)')

axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title('YZ Projection')
axes[2].legend()
axes[2].grid(True)

# Basis2 & Basis3 projection
# Project the lines onto the basis2-basis3 plane
line1_basis2 = np.dot(x_line1.T, basis2)
line1_basis3 = np.dot(x_line1.T, basis3)
line2_basis2 = np.dot(x_line2.T, basis2)
line2_basis3 = np.dot(x_line2.T, basis3)

# Project key points onto basis2-basis3 plane
# Delta = -1
case_deltaN1_line1_basis = np.array([np.dot(case_deltaN1_line1, basis2), np.dot(case_deltaN1_line1, basis3)])
case_deltaN1_line2_basis = np.array([np.dot(case_deltaN1_line2, basis2), np.dot(case_deltaN1_line2, basis3)])

# Delta = 0
case_delta0_line1_basis = np.array([np.dot(case_delta0_line1, basis2), np.dot(case_delta0_line1, basis3)])

# Delta = 1
case_delta1_line1_basis = np.array([np.dot(case_delta1_line1, basis2), np.dot(case_delta1_line1, basis3)])
case_delta1_line2_basis = np.array([np.dot(case_delta1_line2, basis2), np.dot(case_delta1_line2, basis3)])

axes[3].plot(line1_basis2, line1_basis3, 'r', label='Line 1', alpha=0.7)
axes[3].plot(line2_basis2, line2_basis3, 'g', label='Line 2', alpha=0.7)

# Highlight key delta values
# Delta = -1
axes[3].scatter(case_deltaN1_line1_basis[0], case_deltaN1_line1_basis[1], color='blue', s=100, marker='<', label='δ=-1 (Line 1)')
axes[3].scatter(case_deltaN1_line2_basis[0], case_deltaN1_line2_basis[1], color='blue', s=100, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
axes[3].scatter(case_delta0_line1_basis[0], case_delta0_line1_basis[1], color='black', s=100, marker='o', label='δ=0 (intersection)')

# Delta = 1
axes[3].scatter(case_delta1_line1_basis[0], case_delta1_line1_basis[1], color='purple', s=100, marker='>', label='δ=1 (Line 1)')
axes[3].scatter(case_delta1_line2_basis[0], case_delta1_line2_basis[1], color='purple', s=100, marker='>', label='δ=1 (Line 2)')

axes[3].set_xlabel('Basis2')
axes[3].set_ylabel('Basis3')
axes[3].set_title('Basis2-Basis3 Projection')
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir_2d, 'x_0_plusminus_delta_projections_2d.png'))
plt.close()

# Create a 3D plot for the lines in xyz coordinate system (without basis vectors)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the lines
ax.plot(x_line1[0], x_line1[1], x_line1[2], 'r', label='Line 1', alpha=0.7, linewidth=2)
ax.plot(x_line2[0], x_line2[1], x_line2[2], 'g', label='Line 2', alpha=0.7, linewidth=2)

# Highlight key delta values in 3D
# Delta = -1
ax.scatter(case_deltaN1_line1[0], case_deltaN1_line1[1], case_deltaN1_line1[2], color='blue', s=150, marker='<', label='δ=-1 (Line 1)')
ax.scatter(case_deltaN1_line2[0], case_deltaN1_line2[1], case_deltaN1_line2[2], color='blue', s=150, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
ax.scatter(case_delta0_line1[0], case_delta0_line1[1], case_delta0_line1[2], color='black', s=150, marker='o', label='δ=0 (intersection)')

# Delta = 1
ax.scatter(case_delta1_line1[0], case_delta1_line1[1], case_delta1_line1[2], color='purple', s=150, marker='>', label='δ=1 (Line 1)')
ax.scatter(case_delta1_line2[0], case_delta1_line2[1], case_delta1_line2[2], color='purple', s=150, marker='>', label='δ=1 (Line 2)')

# Add text labels for key delta values
ax.text(case_deltaN1_line1[0], case_deltaN1_line1[1], case_deltaN1_line1[2], '  δ=-1 (Line 1)', color='blue')
ax.text(case_deltaN1_line2[0], case_deltaN1_line2[1], case_deltaN1_line2[2], '  δ=-1 (Line 2)', color='blue')
ax.text(case_delta0_line1[0], case_delta0_line1[1], case_delta0_line1[2], '  δ=0 (intersection)', color='black')
ax.text(case_delta1_line1[0], case_delta1_line1[1], case_delta1_line1[2], '  δ=1 (Line 1)', color='purple')
ax.text(case_delta1_line2[0], case_delta1_line2[1], case_delta1_line2[2], '  δ=1 (Line 2)', color='purple')

# Plot the coordinate system axes
origin = np.zeros(3)
max_range = max(np.max(np.abs(x_line1)), np.max(np.abs(x_line2))) * 1.2

# Plot only the standard xyz coordinate system (no basis vectors)
ax.quiver(origin[0], origin[1], origin[2], max_range, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis')
ax.quiver(origin[0], origin[1], origin[2], 0, max_range, 0, color='g', arrow_length_ratio=0.1, label='Y axis')
ax.quiver(origin[0], origin[1], origin[2], 0, 0, max_range, color='b', arrow_length_ratio=0.1, label='Z axis')

# Add a grid for better orientation
ax.grid(True)

# Improve axis labels
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
ax.set_title('3D Plot of Lines in X-Y-Z Coordinate System', fontsize=16)
ax.legend(loc='best')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Save the XYZ plot
plt.savefig(os.path.join(plot_dir_3d, 'x_y_z.png'), dpi=150)
plt.close()

# Create a separate plot showing both coordinate systems for reference
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the lines
ax.plot(x_line1[0], x_line1[1], x_line1[2], 'r', label='Line 1', alpha=0.5, linewidth=1.5)
ax.plot(x_line2[0], x_line2[1], x_line2[2], 'g', label='Line 2', alpha=0.5, linewidth=1.5)

# Plot both coordinate systems for reference
origin = np.zeros(3)
max_range = max(np.max(np.abs(x_line1)), np.max(np.abs(x_line2))) * 1.2

# Plot the standard xyz coordinate system
ax.quiver(origin[0], origin[1], origin[2], max_range, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis')
ax.quiver(origin[0], origin[1], origin[2], 0, max_range, 0, color='g', arrow_length_ratio=0.1, label='Y axis')
ax.quiver(origin[0], origin[1], origin[2], 0, 0, max_range, color='b', arrow_length_ratio=0.1, label='Z axis')

# Plot the basis vectors
scale_factor = max_range * 0.8
ax.quiver(origin[0], origin[1], origin[2], basis1[0]*scale_factor, basis1[1]*scale_factor, basis1[2]*scale_factor, 
          color='m', arrow_length_ratio=0.1, label='Basis1')
ax.quiver(origin[0], origin[1], origin[2], basis2[0]*scale_factor, basis2[1]*scale_factor, basis2[2]*scale_factor, 
          color='c', arrow_length_ratio=0.1, label='Basis2')
ax.quiver(origin[0], origin[1], origin[2], basis3[0]*scale_factor, basis3[1]*scale_factor, basis3[2]*scale_factor, 
          color='y', arrow_length_ratio=0.1, label='Basis3')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reference: Both Coordinate Systems')
ax.legend()

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.savefig(os.path.join(plot_dir_3d, 'both_coordinate_systems.png'), dpi=150)
plt.close()

# Create a 3D plot for the lines in basis1, basis2, basis3 coordinate system
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Transform the lines to basis coordinates
line1_basis = np.zeros((3, len(delta)))
line2_basis = np.zeros((3, len(delta)))

for i in range(len(delta)):
    # For each point, compute the coordinates in the basis
    point1 = np.array([x_line1[0][i], x_line1[1][i], x_line1[2][i]])
    point2 = np.array([x_line2[0][i], x_line2[1][i], x_line2[2][i]])
    
    # Project onto basis vectors
    line1_basis[0][i] = np.dot(point1, basis1)
    line1_basis[1][i] = np.dot(point1, basis2)
    line1_basis[2][i] = np.dot(point1, basis3)
    
    line2_basis[0][i] = np.dot(point2, basis1)
    line2_basis[1][i] = np.dot(point2, basis2)
    line2_basis[2][i] = np.dot(point2, basis3)

# Transform key points to basis coordinates
# Delta = -1
case_deltaN1_line1_basis3d = np.array([np.dot(case_deltaN1_line1, basis1), np.dot(case_deltaN1_line1, basis2), np.dot(case_deltaN1_line1, basis3)])
case_deltaN1_line2_basis3d = np.array([np.dot(case_deltaN1_line2, basis1), np.dot(case_deltaN1_line2, basis2), np.dot(case_deltaN1_line2, basis3)])

# Delta = 0
case_delta0_line1_basis3d = np.array([np.dot(case_delta0_line1, basis1), np.dot(case_delta0_line1, basis2), np.dot(case_delta0_line1, basis3)])

# Delta = 1
case_delta1_line1_basis3d = np.array([np.dot(case_delta1_line1, basis1), np.dot(case_delta1_line1, basis2), np.dot(case_delta1_line1, basis3)])
case_delta1_line2_basis3d = np.array([np.dot(case_delta1_line2, basis1), np.dot(case_delta1_line2, basis2), np.dot(case_delta1_line2, basis3)])

# Set the view angle to match the x_y_z plot
ax.view_init(elev=20, azim=-60)

# Plot the lines in basis coordinates - use the same colors and styles as in x_y_z plot
ax.plot(line1_basis[0], line1_basis[1], line1_basis[2], 'r', label='Line 1', alpha=0.7, linewidth=2)
ax.plot(line2_basis[0], line2_basis[1], line2_basis[2], 'g', label='Line 2', alpha=0.7, linewidth=2)

# Highlight key delta values in 3D basis coordinates
# Delta = -1
ax.scatter(case_deltaN1_line1_basis3d[0], case_deltaN1_line1_basis3d[1], case_deltaN1_line1_basis3d[2], color='blue', s=150, marker='<', label='δ=-1 (Line 1)')
ax.scatter(case_deltaN1_line2_basis3d[0], case_deltaN1_line2_basis3d[1], case_deltaN1_line2_basis3d[2], color='blue', s=150, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
ax.scatter(case_delta0_line1_basis3d[0], case_delta0_line1_basis3d[1], case_delta0_line1_basis3d[2], color='black', s=150, marker='o', label='δ=0 (intersection)')

# Delta = 1
ax.scatter(case_delta1_line1_basis3d[0], case_delta1_line1_basis3d[1], case_delta1_line1_basis3d[2], color='purple', s=150, marker='>', label='δ=1 (Line 1)')
ax.scatter(case_delta1_line2_basis3d[0], case_delta1_line2_basis3d[1], case_delta1_line2_basis3d[2], color='purple', s=150, marker='>', label='δ=1 (Line 2)')

# Add text labels for key delta values - position them better
ax.text(case_deltaN1_line1_basis3d[0], case_deltaN1_line1_basis3d[1], case_deltaN1_line1_basis3d[2], '  δ=-1 (Line 1)', color='blue', fontsize=10)
ax.text(case_deltaN1_line2_basis3d[0], case_deltaN1_line2_basis3d[1], case_deltaN1_line2_basis3d[2], '  δ=-1 (Line 2)', color='blue', fontsize=10)
ax.text(case_delta0_line1_basis3d[0], case_delta0_line1_basis3d[1], case_delta0_line1_basis3d[2], '  δ=0 (intersection)', color='black', fontsize=10)
ax.text(case_delta1_line1_basis3d[0], case_delta1_line1_basis3d[1], case_delta1_line1_basis3d[2], '  δ=1 (Line 1)', color='purple', fontsize=10)
ax.text(case_delta1_line2_basis3d[0], case_delta1_line2_basis3d[1], case_delta1_line2_basis3d[2], '  δ=1 (Line 2)', color='purple', fontsize=10)

# Plot the coordinate system using X, Y, Z labels for the basis vectors
origin = np.zeros(3)
max_range = max(np.max(np.abs(line1_basis)), np.max(np.abs(line2_basis))) * 1.2

ax.quiver(origin[0], origin[1], origin[2], max_range, 0, 0, color='m', arrow_length_ratio=0.1, label='X (Basis1)')
ax.quiver(origin[0], origin[1], origin[2], 0, max_range, 0, color='c', arrow_length_ratio=0.1, label='Y (Basis2)')
ax.quiver(origin[0], origin[1], origin[2], 0, 0, max_range, color='y', arrow_length_ratio=0.1, label='Z (Basis3)')

# Add a grid for better orientation
ax.grid(True)

# Set limits to ensure proper display of the full lines
max_coord = max(np.max(np.abs(line1_basis)), np.max(np.abs(line2_basis))) * 1.2
ax.set_xlim(-max_coord, max_coord)
ax.set_ylim(-max_coord, max_coord)
ax.set_zlim(-max_coord, max_coord)

# Improve axis labels
ax.set_xlabel('X (Basis1)', fontsize=14)
ax.set_ylabel('Y (Basis2)', fontsize=14)
ax.set_zlabel('Z (Basis3)', fontsize=14)
ax.set_title('3D Plot of Lines in Basis1-Basis2-Basis3 Coordinate System', fontsize=16)
ax.legend(loc='best')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Save with a clearer filename
plt.savefig(os.path.join(plot_dir_3d, 'basis1_basis2_basis3.png'), dpi=150)
plt.close()

# Create a separate plot for basis2 & basis3 projection with improved visibility
fig = plt.figure(figsize=(10, 10))

# Plot the lines with improved visibility
plt.plot(line1_basis2, line1_basis3, 'r', label='Line 1', linewidth=2, alpha=0.7)
plt.plot(line2_basis2, line2_basis3, 'g', label='Line 2', linewidth=2, alpha=0.7)

# Highlight key delta values with the same markers as in other plots
# Delta = -1
plt.scatter(case_deltaN1_line1_basis[0], case_deltaN1_line1_basis[1], color='blue', s=150, marker='<', label='δ=-1 (Line 1)')
plt.scatter(case_deltaN1_line2_basis[0], case_deltaN1_line2_basis[1], color='blue', s=150, marker='<', label='δ=-1 (Line 2)')

# Delta = 0 (intersection)
plt.scatter(case_delta0_line1_basis[0], case_delta0_line1_basis[1], color='black', s=150, marker='o', label='δ=0 (intersection)')

# Delta = 1
plt.scatter(case_delta1_line1_basis[0], case_delta1_line1_basis[1], color='purple', s=150, marker='>', label='δ=1 (Line 1)')
plt.scatter(case_delta1_line2_basis[0], case_delta1_line2_basis[1], color='purple', s=150, marker='>', label='δ=1 (Line 2)')

# Add text labels for key points
plt.annotate('δ=-1 (Line 1)', xy=(case_deltaN1_line1_basis[0], case_deltaN1_line1_basis[1]), 
             xytext=(10, 10), textcoords='offset points', color='blue', fontsize=10)
plt.annotate('δ=-1 (Line 2)', xy=(case_deltaN1_line2_basis[0], case_deltaN1_line2_basis[1]), 
             xytext=(10, 10), textcoords='offset points', color='blue', fontsize=10)
plt.annotate('δ=0', xy=(case_delta0_line1_basis[0], case_delta0_line1_basis[1]), 
             xytext=(10, 10), textcoords='offset points', color='black', fontsize=10)
plt.annotate('δ=1 (Line 1)', xy=(case_delta1_line1_basis[0], case_delta1_line1_basis[1]), 
             xytext=(10, 10), textcoords='offset points', color='purple', fontsize=10)
plt.annotate('δ=1 (Line 2)', xy=(case_delta1_line2_basis[0], case_delta1_line2_basis[1]), 
             xytext=(10, 10), textcoords='offset points', color='purple', fontsize=10)

# Add axis lines for better orientation
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Improve appearance
plt.xlabel('Basis2', fontsize=14)
plt.ylabel('Basis3', fontsize=14)
plt.title('Basis2-Basis3 Projection', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

# Set equal aspect ratio for better visualization
plt.axis('equal')

# Calculate the plot limits to center the origin
xlim = plt.xlim()
ylim = plt.ylim()
max_range = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
margin = max_range * 0.1  # Add a 10% margin

# Set symmetric limits around the origin
plt.xlim(-max_range-margin, max_range+margin)
plt.ylim(-max_range-margin, max_range+margin)

# Add a tight layout and save with higher DPI for better quality
plt.tight_layout()
plt.savefig(os.path.join(plot_dir_2d, 'basis2_basis3_projections_2d.png'), dpi=150)
plt.close()


