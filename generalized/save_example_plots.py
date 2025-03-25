#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from vector_utils import create_orthogonal_vectors, check_orthogonality
from visualization import plot_all_projections
from config import VectorConfig

def save_example_plots():
    """
    Generate and save example plots for documentation
    """
    # Create output directory
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configurations for examples
    configs = {
        "default": VectorConfig(),
        "custom1": VectorConfig(
            R_0=(1, 1, 1),  # Non-origin starting point
            d=2,            # Larger distance
            theta=math.pi/3 # 60 degrees
        ),
        "custom2": VectorConfig(
            R_0=(0, 0, 2),  # Point on z-axis
            d=1.5,          # Medium distance
            theta=math.pi/6 # 30 degrees
        )
    }
    
    # Generate and save plots for each configuration
    for name, config in configs.items():
        print(f"Generating plots for {name} configuration...")
        
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
        
        # Save each plot
        for plot_name, (fig, _) in plots.items():
            filename = os.path.join(output_dir, f"{name}_{plot_name}.png")
            fig.savefig(filename, dpi=150)
            print(f"Saved plot to {filename}")
        
        print()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    save_example_plots()
