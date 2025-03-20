#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the Arrowhead/generalized directory to the path
sys.path.append('/home/zoli/arrowhead/Arrowhead/generalized')

# Try to import the perfect orthogonal circle generation function
try:
    from vector_utils import create_perfect_orthogonal_vectors
except ImportError:
    print("Warning: Could not import create_perfect_orthogonal_vectors from Arrowhead/generalized package.")
    print("Falling back to simple circle implementation.")
    # Define a fallback function if the import fails
    def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([1, -1/2, -1/2])  # First basis vector
        basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
        
        # Normalize the basis vectors
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Create a point at distance d from the origin in the plane spanned by basis1 and basis2
        R = np.array(R_0) + d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
        
        return R

# Function to create a simple circle in the x-y plane
def simple_circle(d, theta):
    """
    Create a vector that traces a circle in the x-y plane with radius d.
    
    Parameters:
    d (float): The radius of the circle
    theta (float): The angle parameter
    
    Returns:
    numpy.ndarray: A 3D vector [x, y, z] where x = d*cos(theta), y = d*sin(theta), z = 0
    """
    # Simple parametric equation for a circle in the x-y plane
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    z = 0.0
    
    return np.array([x, y, z])

# Function to create a perfect orthogonal circle
def perfect_circle(d, theta):
    """
    Create a vector that traces a perfect circle orthogonal to the x=y=z line.
    
    Parameters:
    d (float): The radius of the circle
    theta (float): The angle parameter
    
    Returns:
    numpy.ndarray: A 3D vector orthogonal to the x=y=z line
    """
    # Origin vector
    R_0 = np.array([0, 0, 0])
    
    # Generate the perfect orthogonal vector
    return create_perfect_orthogonal_vectors(R_0, d, theta)

# Function to visualize the circles
def visualize_circles(d, N=100):
    """
    Visualize the simple circle and perfect orthogonal circle
    
    Parameters:
    d (float): The radius of the circle
    N (int): Number of points to plot
    """
    # Generate theta values
    theta_values = np.linspace(0, 2*np.pi, N+1)[:-1]  # Exclude the last point to avoid duplication
    
    # Generate points for simple circle
    simple_circle_points = np.array([simple_circle(d, theta) for theta in theta_values])
    
    # Generate points for perfect orthogonal circle
    perfect_circle_points = np.array([perfect_circle(d, theta) for theta in theta_values])
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the simple circle
    ax.plot(simple_circle_points[:, 0], simple_circle_points[:, 1], simple_circle_points[:, 2], 'b-', label='Simple Circle (x-y plane)')
    
    # Plot the perfect orthogonal circle
    ax.plot(perfect_circle_points[:, 0], perfect_circle_points[:, 1], perfect_circle_points[:, 2], 'r-', label='Perfect Orthogonal Circle')
    
    # Plot the origin
    ax.scatter([0], [0], [0], color='k', s=100, label='Origin')
    
    # Plot the (1,1,1) direction
    unit_vector = np.array([1, 1, 1]) / np.sqrt(3)
    ax.quiver(0, 0, 0, unit_vector[0], unit_vector[1], unit_vector[2], color='g', label='(1,1,1) Direction')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Comparison of Circle Types (d={d})')
    
    # Set equal aspect ratio
    max_range = np.array([simple_circle_points[:, 0].max() - simple_circle_points[:, 0].min(),
                          simple_circle_points[:, 1].max() - simple_circle_points[:, 1].min(),
                          simple_circle_points[:, 2].max() - simple_circle_points[:, 2].min(),
                          perfect_circle_points[:, 0].max() - perfect_circle_points[:, 0].min(),
                          perfect_circle_points[:, 1].max() - perfect_circle_points[:, 1].min(),
                          perfect_circle_points[:, 2].max() - perfect_circle_points[:, 2].min()]).max() / 2.0
    mid_x = (simple_circle_points[:, 0].max() + simple_circle_points[:, 0].min()) * 0.5
    mid_y = (simple_circle_points[:, 1].max() + simple_circle_points[:, 1].min()) * 0.5
    mid_z = (simple_circle_points[:, 2].max() + simple_circle_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add a legend
    ax.legend()
    
    # Add a grid
    ax.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('circle_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Visualize and compare simple circle and perfect orthogonal circle')
    parser.add_argument('--distance', '-d', type=float, default=0.1, help='Distance parameter d (default: 0.1)')
    parser.add_argument('--points', '-n', type=int, default=100, help='Number of points to plot (default: 100)')
    args = parser.parse_args()
    
    # Visualize the circles
    visualize_circles(args.distance, args.points)
