#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap

def plot_vectors_3d(R_0, R, figsize=(10, 8), show_legend=True):
    """
    Plot the vector in 3D
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R (numpy.ndarray): The R vector
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter(R_0[0], R_0[1], R_0[2], color='black', s=100, label='R_0')
    
    # Plot the R vector as an arrow from the origin
    ax.quiver(R_0[0], R_0[1], R_0[2], 
             R[0]-R_0[0], R[1]-R_0[1], R[2]-R_0[2], 
             color='r', label='R', arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of R Vector')
    
    # Set equal aspect ratio
    max_range = np.array([
        np.max([R_0[0], R[0]]) - np.min([R_0[0], R[0]]),
        np.max([R_0[1], R[1]]) - np.min([R_0[1], R[1]]),
        np.max([R_0[2], R[2]]) - np.min([R_0[2], R[2]])
    ]).max() / 2.0
    
    mid_x = (np.max([R_0[0], R[0]]) + np.min([R_0[0], R[0]])) / 2
    mid_y = (np.max([R_0[1], R[1]]) + np.min([R_0[1], R[1]])) / 2
    mid_z = (np.max([R_0[2], R[2]]) + np.min([R_0[2], R[2]])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def plot_vectors_2d_projection(R_0, R, plane='xy', figsize=(8, 8), show_legend=True, show_grid=True):
    """
    Plot the 2D projection of the vectors
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R_1, R_2, R_3 (numpy.ndarray): The three orthogonal vectors
    plane (str): The plane to project onto ('xy', 'xz', 'yz', or 'r0')
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
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_proj = np.array([R[i], R[j]])
        
    elif plane == 'xz':
        i, j = 0, 2
        plane_name = 'XZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_proj = np.array([R[i], R[j]])
        
    elif plane == 'yz':
        i, j = 1, 2
        plane_name = 'YZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_proj = np.array([R[i], R[j]])
        
    elif plane == 'r0':
        # For R_0 plane, we need to find two orthogonal vectors in the plane
        # perpendicular to the vector from origin to R_0
        
        # If R_0 is the origin, we can use any plane passing through the origin
        if np.allclose(R_0, np.zeros(3)):
            # Create a basis for the plane
            if not np.allclose(R - R_0, np.zeros(3)):
                # Use R as one basis vector
                basis1 = R - R_0
                basis1 = basis1 / np.linalg.norm(basis1)
                
                # Find a vector orthogonal to basis1
                if not np.allclose(basis1, np.array([1, 0, 0])):
                    basis2 = np.cross(basis1, np.array([1, 0, 0]))
                else:
                    basis2 = np.cross(basis1, np.array([0, 1, 0]))
                basis2 = basis2 / np.linalg.norm(basis2)
            else:
                # If R is at origin, use standard basis
                basis1 = np.array([1, 0, 0])
                basis2 = np.array([0, 1, 0])
        else:
            # Define the normal to the plane as the vector from origin to R_0
            normal = R_0 / np.linalg.norm(R_0)
            
            # Find two orthogonal vectors in the plane
            # First basis vector: cross product of normal with [1,0,0] or [0,1,0]
            if not np.allclose(normal, np.array([1, 0, 0])):
                basis1 = np.cross(normal, np.array([1, 0, 0]))
            else:
                basis1 = np.cross(normal, np.array([0, 1, 0]))
            basis1 = basis1 / np.linalg.norm(basis1)
            
            # Second basis vector: cross product of normal with basis1
            basis2 = np.cross(normal, basis1)
            basis2 = basis2 / np.linalg.norm(basis2)
        
        plane_name = 'R_0'
        
        # Project vectors onto the plane defined by basis1 and basis2
        R0_proj = np.array([0, 0])  # Origin in the plane
        
        # Project R onto the plane
        v = R - R_0
        R_proj = np.array([np.dot(v, basis1), np.dot(v, basis2)])
    else:
        raise ValueError("Plane must be 'xy', 'xz', 'yz', or 'r0'")
    
    # Plot the origin
    ax.scatter(R0_proj[0], R0_proj[1], color='black', s=100, label='R_0')
    
    # Plot the R vector as an arrow from the origin
    ax.arrow(R0_proj[0], R0_proj[1], 
            R_proj[0]-R0_proj[0], R_proj[1]-R0_proj[1], 
            head_width=0.05, head_length=0.1, fc='r', ec='r', label='R')
    
    # Set labels and title
    if plane == 'r0':
        ax.set_xlabel('Basis 1')
        ax.set_ylabel('Basis 2')
        ax.set_title(f'2D Projection on the {plane_name} Plane')
    else:
        ax.set_xlabel(f'{plane_name[0]} axis')
        ax.set_ylabel(f'{plane_name[1]} axis')
        ax.set_title(f'2D Projection on {plane_name} Plane')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and legend
    if show_grid:
        ax.grid(True)
    
    if show_legend:
        ax.legend()
    
    return fig, ax

def plot_all_projections(R_0, R, show_r0_plane=True, figsize_3d=(10, 8), figsize_2d=(8, 8)):
    """
    Plot the 3D vector and all 2D projections
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R (numpy.ndarray): The R vector
    show_r0_plane (bool): Whether to show the R_0 plane projection
    figsize_3d (tuple): Figure size for 3D plot
    figsize_2d (tuple): Figure size for 2D plots
    
    Returns:
    dict: Dictionary containing all figure and axis objects
    """
    results = {}
    
    # Plot in 3D
    fig_3d, ax_3d = plot_vectors_3d(R_0, R, figsize=figsize_3d)
    results['3d'] = (fig_3d, ax_3d)
    
    # Plot 2D projections
    fig_xy, ax_xy = plot_vectors_2d_projection(R_0, R, plane='xy', figsize=figsize_2d)
    results['xy'] = (fig_xy, ax_xy)
    
    fig_xz, ax_xz = plot_vectors_2d_projection(R_0, R, plane='xz', figsize=figsize_2d)
    results['xz'] = (fig_xz, ax_xz)
    
    fig_yz, ax_yz = plot_vectors_2d_projection(R_0, R, plane='yz', figsize=figsize_2d)
    results['yz'] = (fig_yz, ax_yz)
    
    # Plot on the R_0 plane if requested
    if show_r0_plane:
        fig_r0, ax_r0 = plot_vectors_2d_projection(R_0, R, plane='r0', figsize=figsize_2d)
        results['r0'] = (fig_r0, ax_r0)
    
    return results

def plot_multiple_vectors_3d(R_0, vectors, figsize=(12, 10), show_legend=True, endpoints_only=False):
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
    
    # Add axis lines with higher visibility and labels
    # Extract all R vectors for axis scaling
    all_Rs = [R for _, _, R in vectors]
    # Adjust max_val to be closer to the actual data for better visualization
    max_val = max(np.max(np.abs(all_Rs)), np.max(np.abs(R_0))) * 1.5 if len(all_Rs) > 0 else 3.5
    
    # Add axis lines with colors and labels
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
    
    # Get a colormap for the vectors
    cmap = get_cmap('viridis')
    num_vectors = len(vectors)
    
    # Extract all R vectors for axis scaling
    all_Rs = [R for _, _, R in vectors]
    
    # Plot the vectors
    for i, (d, theta, R) in enumerate(vectors):
        color = cmap(i / max(1, num_vectors - 1))
        label = f'R (d={d:.2f}, θ={theta:.2f})'
        
        if endpoints_only:
            # Plot only the endpoint
            ax.scatter(R[0], R[1], R[2], color=color, s=50, label=label)
        else:
            # Plot the vector as an arrow
            ax.quiver(R_0[0], R_0[1], R_0[2], 
                     R[0]-R_0[0], R[1]-R_0[1], R[2]-R_0[2], 
                     color=color, label=label, arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Multiple R Vectors')
    
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
    
    # Add a buffer for better visibility
    buffer_factor = getattr(config, 'buffer_factor', 0.2)  # Default to 0.2 if not specified
    buffer = max_range * buffer_factor
    
    # Calculate a tighter range based on actual data points
    data_range = max([
        np.max([p[0] for p in all_points]) - np.min([p[0] for p in all_points]),
        np.max([p[1] for p in all_points]) - np.min([p[1] for p in all_points]),
        np.max([p[2] for p in all_points]) - np.min([p[2] for p in all_points])
    ]) / 2.0 * 0.7  # Scale down to 70% of original range for tighter view
    
    # Use data-driven limits for better scaling
    ax.set_xlim(mid_x - data_range - buffer, mid_x + data_range + buffer)
    ax.set_ylim(mid_y - data_range - buffer, mid_y + data_range + buffer)
    ax.set_zlim(mid_z - data_range - buffer, mid_z + data_range + buffer)
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([1, 1, 1])
    
    if show_legend and num_vectors <= 10:  # Only show legend if not too many vectors
        ax.legend()
    
    return fig, ax

def plot_multiple_vectors_2d(R_0, vectors, plane='xy', figsize=(10, 10), show_legend=True, show_grid=True, endpoints_only=False):
    """
    Plot the 2D projection of multiple vectors
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    vectors (list): List of tuples (d, theta, R) containing the parameters and vectors
    plane (str): The plane to project onto ('xy', 'xz', 'yz', or 'r0')
    figsize (tuple): Figure size (width, height) in inches
    show_legend (bool): Whether to show the legend
    show_grid (bool): Whether to show the grid
    endpoints_only (bool): If True, only plot the endpoints of vectors, not the arrows
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get a colormap for the vectors
    cmap = get_cmap('viridis')
    num_vectors = len(vectors)
    
    # Extract all R vectors for projections
    all_Rs = [R for _, _, R in vectors]
    
    # Define indices based on the projection plane
    if plane == 'xy':
        i, j = 0, 1
        plane_name = 'XY'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_projs = [np.array([R[i], R[j]]) for R in all_Rs]
        
    elif plane == 'xz':
        i, j = 0, 2
        plane_name = 'XZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_projs = [np.array([R[i], R[j]]) for R in all_Rs]
        
    elif plane == 'yz':
        i, j = 1, 2
        plane_name = 'YZ'
        
        # Project vectors onto the plane
        R0_proj = np.array([R_0[i], R_0[j]])
        R_projs = [np.array([R[i], R[j]]) for R in all_Rs]
        
    elif plane == 'r0':
        # For R_0 plane, we need to find two orthogonal vectors in the plane
        # perpendicular to the vector from origin to R_0
        
        # If R_0 is the origin, we can use any plane passing through the origin
        if np.allclose(R_0, np.zeros(3)):
            # Use the first non-zero vector as a basis if available
            non_zero_vectors = [R for R in all_Rs if not np.allclose(R, np.zeros(3))]
            if non_zero_vectors:
                # Use the first non-zero vector as one basis vector
                basis1 = non_zero_vectors[0] - R_0
                basis1 = basis1 / np.linalg.norm(basis1)
                
                # Find a vector orthogonal to basis1
                if not np.allclose(basis1, np.array([1, 0, 0])):
                    basis2 = np.cross(basis1, np.array([1, 0, 0]))
                else:
                    basis2 = np.cross(basis1, np.array([0, 1, 0]))
                basis2 = basis2 / np.linalg.norm(basis2)
            else:
                # If all vectors are at origin, use standard basis
                basis1 = np.array([1, 0, 0])
                basis2 = np.array([0, 1, 0])
        else:
            # Define the normal to the plane as the vector from origin to R_0
            normal = R_0 / np.linalg.norm(R_0)
            
            # Find two orthogonal vectors in the plane
            # First basis vector: cross product of normal with [1,0,0] or [0,1,0]
            if not np.allclose(normal, np.array([1, 0, 0])):
                basis1 = np.cross(normal, np.array([1, 0, 0]))
            else:
                basis1 = np.cross(normal, np.array([0, 1, 0]))
            basis1 = basis1 / np.linalg.norm(basis1)
            
            # Second basis vector: cross product of normal with basis1
            basis2 = np.cross(normal, basis1)
            basis2 = basis2 / np.linalg.norm(basis2)
        
        plane_name = 'R_0'
        
        # Project vectors onto the plane defined by basis1 and basis2
        R0_proj = np.array([0, 0])  # Origin in the plane
        
        # Project all R vectors onto the plane
        R_projs = []
        for R in all_Rs:
            v = R - R_0
            R_projs.append(np.array([np.dot(v, basis1), np.dot(v, basis2)]))
    else:
        raise ValueError("Plane must be 'xy', 'xz', 'yz', or 'r0'")
    
    # Plot the origin
    ax.scatter(R0_proj[0], R0_proj[1], color='black', s=100, label='R_0')
    
    # Plot the vectors
    for i, ((d, theta, _), R_proj) in enumerate(zip(vectors, R_projs)):
        color = cmap(i / max(1, num_vectors - 1))
        label = f'R (d={d:.2f}, θ={theta:.2f})'
        
        if endpoints_only:
            # Plot only the endpoint
            ax.scatter(R_proj[0], R_proj[1], color=color, s=50, label=label)
        else:
            # Plot the vector as an arrow
            ax.arrow(R0_proj[0], R0_proj[1], 
                    R_proj[0]-R0_proj[0], R_proj[1]-R0_proj[1], 
                    head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Set labels and title
    if plane == 'r0':
        ax.set_xlabel('Basis 1')
        ax.set_ylabel('Basis 2')
        ax.set_title(f'2D Projection on the {plane_name} Plane')
    else:
        ax.set_xlabel(f'{plane_name[0]} axis')
        ax.set_ylabel(f'{plane_name[1]} axis')
        ax.set_title(f'2D Projection on {plane_name} Plane')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and legend
    if show_grid:
        ax.grid(True)
    
    if show_legend and num_vectors <= 10:  # Only show legend if not too many vectors
        ax.legend()
    
    return fig, ax

def plot_multiple_vectors(R_0, vectors, show_r0_plane=True, figsize_3d=(12, 10), figsize_2d=(10, 10), endpoints_only=False):
    """
    Plot multiple vectors in 3D and 2D projections
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    vectors (list): List of tuples (d, theta, R) containing the parameters and vectors
    show_r0_plane (bool): Whether to show the R_0 plane projection
    figsize_3d (tuple): Figure size for 3D plot
    figsize_2d (tuple): Figure size for 2D plots
    endpoints_only (bool): If True, only plot the endpoints of vectors, not the arrows
    
    Returns:
    dict: Dictionary containing all figure and axis objects
    """
    results = {}
    
    # Plot in 3D
    fig_3d, ax_3d = plot_multiple_vectors_3d(R_0, vectors, figsize=figsize_3d, endpoints_only=endpoints_only)
    results['3d'] = (fig_3d, ax_3d)
    
    # Plot 2D projections
    fig_xy, ax_xy = plot_multiple_vectors_2d(R_0, vectors, plane='xy', figsize=figsize_2d, endpoints_only=endpoints_only)
    results['xy'] = (fig_xy, ax_xy)
    
    fig_xz, ax_xz = plot_multiple_vectors_2d(R_0, vectors, plane='xz', figsize=figsize_2d, endpoints_only=endpoints_only)
    results['xz'] = (fig_xz, ax_xz)
    
    fig_yz, ax_yz = plot_multiple_vectors_2d(R_0, vectors, plane='yz', figsize=figsize_2d, endpoints_only=endpoints_only)
    results['yz'] = (fig_yz, ax_yz)
    
    # Plot on the R_0 plane if requested
    if show_r0_plane:
        fig_r0, ax_r0 = plot_multiple_vectors_2d(R_0, vectors, plane='r0', figsize=figsize_2d, endpoints_only=endpoints_only)
        results['r0'] = (fig_r0, ax_r0)
    
    return results
