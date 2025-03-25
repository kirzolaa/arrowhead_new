#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

def generate_circle_points_xy():
    """
    Generate 72 points in a circle in the XY plane by varying theta from 0 to 360 degrees
    with a fixed radius of 0.1 from origin (0,0,0)
    
    Returns:
    tuple: (R_0, vectors) where R_0 is the origin and vectors is a list of (d, theta, R) tuples
    """
    # Set parameters
    R_0 = np.array([0, 0, 0])  # Origin
    radius = 0.1               # Circle radius
    
    # Generate theta values from 0 to 360 degrees in steps of 5 degrees
    # Convert to radians for calculations
    theta_values = np.radians(np.arange(0, 361, 5))
    
    # Generate vectors for each theta value (traditional circle in XY plane)
    vectors = []
    for theta in theta_values:
        # Create a point on the circle in the XY plane
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0  # Set z=0 for a flat circle in XY plane
        
        R = np.array([x, y, z])
        vectors.append((radius, theta, R))
        print(f"Generated point for θ={math.degrees(theta):.1f}°: {R}")
    
    return R_0, vectors

def plot_multiple_vectors_3d(R_0, vectors, figsize=(12, 10), show_legend=True, endpoints_only=True):
    """
    Plot multiple vectors in 3D
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    vectors (list): List of tuples (d, theta, R) containing the parameters and vectors
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    endpoints_only (bool): If True, only plot the endpoints of vectors, not the arrows
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter(R_0[0], R_0[1], R_0[2], color='black', s=100, label='R_0')
    
    # Get a colormap for the vectors
    cmap = plt.cm.get_cmap('viridis')
    num_vectors = len(vectors)
    
    # Extract all R vectors for axis scaling
    all_Rs = [R for _, _, R in vectors]
    
    # Plot the vectors
    for i, (d, theta, R) in enumerate(vectors):
        color = cmap(i / max(1, num_vectors - 1))
        label = f'R (θ={math.degrees(theta):.1f}°)' if i % 10 == 0 else None
        
        # Plot only the endpoint
        ax.scatter(R[0], R[1], R[2], color=color, s=50, label=label)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Circle Points in 3D (XY Plane)')
    
    # Set equal aspect ratio
    all_points = [R_0] + all_Rs
    max_range = np.array([
        np.max([p[0] for p in all_points]) - np.min([p[0] for p in all_points]),
        np.max([p[1] for p in all_points]) - np.min([p[1] for p in all_points]),
        np.max([p[2] for p in all_points]) - np.min([p[2] for p in all_points])
    ]).max() / 2.0
    
    mid_x = (np.max([p[0] for p in all_points]) + np.min([p[0] for p in all_points])) / 2
    mid_y = (np.max([p[1] for p in all_points]) + np.min([p[1] for p in all_points])) / 2
    mid_z = (np.max([p[2] for p in all_points]) + np.min([p[2] for p in all_points])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def plot_multiple_vectors_2d(R_0, vectors, plane='xy', figsize=(10, 10), show_legend=True, show_grid=True):
    """
    Plot the 2D projection of multiple vectors
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    vectors (list): List of tuples (d, theta, R) containing the parameters and vectors
    plane (str): The plane to project onto ('xy', 'xz', 'yz')
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    show_grid (bool): Whether to show the grid
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define indices based on the projection plane
    if plane == 'xy':
        i, j = 0, 1
        plane_name = 'XY'
    elif plane == 'xz':
        i, j = 0, 2
        plane_name = 'XZ'
    elif plane == 'yz':
        i, j = 1, 2
        plane_name = 'YZ'
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")
    
    # Plot the origin
    ax.scatter(R_0[i], R_0[j], color='black', s=100, label='R_0')
    
    # Get a colormap for the vectors
    cmap = plt.cm.get_cmap('viridis')
    num_vectors = len(vectors)
    
    # Plot the vectors
    for i_vec, (d, theta, R) in enumerate(vectors):
        color = cmap(i_vec / max(1, num_vectors - 1))
        label = f'R (θ={math.degrees(theta):.1f}°)' if i_vec % 10 == 0 else None
        
        # Plot only the endpoint
        ax.scatter(R[i], R[j], color=color, s=50, label=label)
    
    # Set labels and title
    ax.set_xlabel(f'{plane_name[0]} axis')
    ax.set_ylabel(f'{plane_name[1]} axis')
    ax.set_title(f'Circle Points on {plane_name} Plane')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    if show_grid:
        ax.grid(True)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def main():
    """
    Main function to generate and visualize circle points
    """
    print("Generating circle points in XY plane...")
    R_0, vectors = generate_circle_points_xy()
    
    print(f"\nGenerated {len(vectors)} points.")
    
    # Create plots directory if it doesn't exist
    output_dir = 'circle_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the points in 3D
    print("Creating plots...")
    fig_3d, ax_3d = plot_multiple_vectors_3d(R_0, vectors)
    
    # Plot the points in 2D (XY plane)
    fig_xy, ax_xy = plot_multiple_vectors_2d(R_0, vectors, plane='xy')
    
    # Save the plots
    plots = {
        '3d_xy_circle': (fig_3d, ax_3d),
        'xy_circle': (fig_xy, ax_xy)
    }
    
    for name, (fig, _) in plots.items():
        filename = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filename)
        print(f"Saved plot to {filename}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
