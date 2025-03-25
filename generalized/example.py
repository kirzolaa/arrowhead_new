#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

from vector_utils import create_orthogonal_vectors, check_orthogonality
from visualization import plot_all_projections
from config import VectorConfig

def example_default():
    """
    Example using default parameters
    """
    print("Example with default parameters:")
    
    # Create a default configuration
    config = VectorConfig()
    
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
    
    # Plot the vectors
    plots = plot_all_projections(
        R_0, R_1, R_2, R_3,
        show_r0_plane=config.show_r0_plane,
        figsize_3d=config.figsize_3d,
        figsize_2d=config.figsize_2d
    )
    
    plt.show()

def example_custom():
    """
    Example using custom parameters
    """
    print("Example with custom parameters:")
    
    # Create a custom configuration
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
    
    # Plot the vectors
    plots = plot_all_projections(
        R_0, R_1, R_2, R_3,
        show_r0_plane=config.show_r0_plane,
        figsize_3d=config.figsize_3d,
        figsize_2d=config.figsize_2d
    )
    
    plt.show()

def example_save_load_config():
    """
    Example demonstrating saving and loading configurations
    """
    print("Example demonstrating saving and loading configurations:")
    
    # Create a custom configuration
    original_config = VectorConfig(
        R_0=(2, 0, 1),
        d=1.5,
        theta=math.pi/6
    )
    
    # Save the configuration to a file
    config_file = "example_config.json"
    original_config.save_to_file(config_file)
    print(f"Saved configuration to {config_file}")
    
    # Load the configuration from the file
    loaded_config = VectorConfig.load_from_file(config_file)
    print("Loaded configuration:")
    print(loaded_config.to_dict())
    
    # Create vectors using the loaded configuration
    R_0 = loaded_config.R_0
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, loaded_config.d, loaded_config.theta)
    
    # Print vector information
    print("R_0:", R_0)
    print("R_1:", R_1)
    print("R_2:", R_2)
    print("R_3:", R_3)

if __name__ == "__main__":
    # Uncomment the example you want to run
    example_default()
    # example_custom()
    # example_save_load_config()
