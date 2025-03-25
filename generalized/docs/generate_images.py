#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath('..'))

from vector_utils import create_orthogonal_vectors

def generate_example_images():
    """Generate example images for documentation"""
    # Create output directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Set up example vectors
    R_0 = np.array([0, 0, 0])
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, d=1, theta=np.pi/4)
    
    # Generate 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter(R_0[0], R_0[1], R_0[2], color='black', s=100, label='R_0')
    
    # Plot the vectors as arrows from the origin
    vectors = [R_1, R_2, R_3]
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for vector, color, label in zip(vectors, colors, labels):
        ax.quiver(R_0[0], R_0[1], R_0[2], 
                 vector[0]-R_0[0], vector[1]-R_0[1], vector[2]-R_0[2], 
                 color=color, label=label, arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Orthogonal Vectors')
    ax.legend()
    
    # Save figure
    plt.savefig('figures/3d_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate 2D projections
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    # Define planes and titles
    planes = ['XY', 'XZ', 'YZ', 'R0']
    
    for i, (plane, ax) in enumerate(zip(planes, axs)):
        # Plot the origin
        ax.scatter(0, 0, color='black', s=100, label='R_0')
        
        # Plot the vectors as arrows from the origin
        if plane == 'XY':
            for vector, color, label in zip(vectors, colors, labels):
                ax.arrow(0, 0, vector[0], vector[1], 
                        head_width=0.05, head_length=0.1, 
                        fc=color, ec=color, label=label)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        elif plane == 'XZ':
            for vector, color, label in zip(vectors, colors, labels):
                ax.arrow(0, 0, vector[0], vector[2], 
                        head_width=0.05, head_length=0.1, 
                        fc=color, ec=color, label=label)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
        elif plane == 'YZ':
            for vector, color, label in zip(vectors, colors, labels):
                ax.arrow(0, 0, vector[1], vector[2], 
                        head_width=0.05, head_length=0.1, 
                        fc=color, ec=color, label=label)
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        else:  # R0 plane
            # Simplified R0 plane for example
            for vector, color, label in zip(vectors, colors, labels):
                ax.arrow(0, 0, vector[0], vector[1], 
                        head_width=0.05, head_length=0.1, 
                        fc=color, ec=color, label=label)
            ax.set_xlabel('Basis 1')
            ax.set_ylabel('Basis 2')
        
        ax.set_title(f'{plane} Plane Projection')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/2d_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated example images for documentation")

if __name__ == "__main__":
    generate_example_images()
