#!/usr/bin/env python3
"""
Example demonstrating how to use the generalized implementation as a Python package.

This script shows how to import and use the various components of the package.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Import from the generalized package
from generalized import (
    create_orthogonal_vectors,
    check_orthogonality,
    plot_vectors_3d,
    plot_vectors_2d_projection,
    plot_all_projections,
    VectorConfig
)

def example_direct_usage():
    """
    Example demonstrating direct usage of the functions
    """
    print("Example of direct usage:")
    
    # Define parameters
    R_0 = np.array([0, 0, 0])  # Origin
    d = 1                      # Distance parameter
    theta = math.pi/4          # 45 degrees in radians
    
    # Create the orthogonal vectors
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, d, theta)
    
    # Print vector information
    print("R_0:", R_0)
    print("R_1:", R_1)
    print("R_2:", R_2)
    print("R_3:", R_3)
    
    # Check orthogonality
    orthogonality = check_orthogonality(R_0, R_1, R_2, R_3)
    print("\nChecking orthogonality (dot products should be close to zero):")
    for key, value in orthogonality.items():
        print(f"{key}: {value}")
    
    # Create individual plots
    fig_3d, ax_3d = plot_vectors_3d(R_0, R_1, R_2, R_3)
    fig_xy, ax_xy = plot_vectors_2d_projection(R_0, R_1, R_2, R_3, plane='xy')
    
    plt.show()

def example_config_usage():
    """
    Example demonstrating usage with the VectorConfig class
    """
    print("Example of usage with VectorConfig:")
    
    # Create a configuration
    config = VectorConfig(
        R_0=(1, 1, 1),  # Non-origin starting point
        d=2,            # Larger distance
        theta=math.pi/3 # 60 degrees
    )
    
    # Create the orthogonal vectors
    R_0 = config.R_0
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, config.d, config.theta)
    
    # Print vector information
    print("R_0:", R_0)
    print("R_1:", R_1)
    print("R_2:", R_2)
    print("R_3:", R_3)
    
    # Check orthogonality
    orthogonality = check_orthogonality(R_0, R_1, R_2, R_3)
    print("\nChecking orthogonality (dot products should be close to zero):")
    for key, value in orthogonality.items():
        print(f"{key}: {value}")
    
    # Create all plots at once
    plots = plot_all_projections(
        R_0, R_1, R_2, R_3,
        show_r0_plane=config.show_r0_plane,
        figsize_3d=config.figsize_3d,
        figsize_2d=config.figsize_2d
    )
    
    plt.show()

if __name__ == "__main__":
    # Choose which example to run
    example_direct_usage()
    # example_config_usage()
