#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

# Import from the generalized module
from vector_utils import create_orthogonal_vectors
from visualization import plot_multiple_vectors_3d, plot_multiple_vectors_2d, plot_multiple_vectors

def generate_circle_points():
    """
    Generate 72 points in a circle by varying theta from 0 to 360 degrees
    with a fixed distance d=0.1 from origin R_0=(0,0,0)
    
    Returns:
    list: List of tuples (d, theta, R) containing the parameters and vectors
    """
    # Set parameters
    R_0 = np.array([0, 0, 0])  # Origin
    d = 0.1                    # Fixed distance
    
    # Generate theta values from 0 to 360 degrees in steps of 5 degrees
    # Convert to radians for calculations
    theta_values = np.radians(np.arange(0, 361, 5))
    
    # Generate vectors for each theta value
    vectors = []
    for theta in theta_values:
        R = create_orthogonal_vectors(R_0, d, theta)
        vectors.append((d, theta, R))
        print(f"Generated point for θ={math.degrees(theta):.1f}°: {R}")
    
    return R_0, vectors

def main():
    """
    Main function to generate and visualize circle points
    """
    print("Generating circle points...")
    R_0, vectors = generate_circle_points()
    
    print(f"\nGenerated {len(vectors)} points.")
    
    # Create plots directory if it doesn't exist
    output_dir = 'circle_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the points
    print("Creating plots...")
    plots = plot_multiple_vectors(
        R_0, 
        vectors,
        show_r0_plane=True,
        figsize_3d=(12, 10),
        figsize_2d=(10, 10),
        endpoints_only=True  # Only plot the endpoints
    )
    
    # Save the plots
    for name, (fig, _) in plots.items():
        filename = os.path.join(output_dir, f"circle_{name}.png")
        fig.savefig(filename)
        print(f"Saved plot to {filename}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
