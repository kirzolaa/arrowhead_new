#!/usr/bin/env python3
"""
Script to generate parameter effect visualizations for the documentation.
This script creates 2x2 subplot figures showing the effects of changing parameters.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the vector generation and visualization functions
from vector_utils import create_orthogonal_vectors, check_orthogonality
from visualization import plot_vectors_3d

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def plot_3d_vectors_subplot(ax, R_0, d, theta, title):
    """Plot 3D vectors in a subplot"""
    # Generate vectors
    R_1, R_2, R_3 = create_orthogonal_vectors(R_0, d, theta)
    
    # Clear the axis
    ax.clear()
    
    # Plot the vectors
    ax.scatter(*R_0, color='black', s=50)
    
    # Plot the vectors as arrows
    colors = ['r', 'g', 'b']
    labels = ['R_1', 'R_2', 'R_3']
    
    for i, R in enumerate([R_1, R_2, R_3]):
        ax.quiver(*R_0, *(R - R_0), color=colors[i], label=labels[i])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max([R_1[i], R_2[i], R_3[i]]) - np.min([R_1[i], R_2[i], R_3[i]]) 
        for i in range(3)
    ])
    mid_x = np.mean([R_0[0], R_1[0], R_2[0], R_3[0]])
    mid_y = np.mean([R_0[1], R_1[1], R_2[1], R_3[1]])
    mid_z = np.mean([R_0[2], R_1[2], R_2[2], R_3[2]])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Add legend
    ax.legend()

def generate_d_effect_figure(R_0, theta, d_values, filename):
    """Generate a figure showing the effect of different d values"""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 grid of 3D subplots
    gs = GridSpec(2, 2, figure=fig)
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            axes.append(ax)
    
    # Plot each d value in a separate subplot
    for i, d in enumerate(d_values):
        if i < len(axes):
            plot_3d_vectors_subplot(axes[i], R_0, d, theta, f'd = {d}')
    
    # Add a main title
    fig.suptitle(f'Effect of Distance Parameter (d) with R_0={R_0}, θ={theta:.2f}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def generate_theta_effect_figure(R_0, d, theta_values, filename):
    """Generate a figure showing the effect of different theta values"""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 grid of 3D subplots
    gs = GridSpec(2, 2, figure=fig)
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            axes.append(ax)
    
    # Plot each theta value in a separate subplot
    for i, theta in enumerate(theta_values):
        if i < len(axes):
            plot_3d_vectors_subplot(axes[i], R_0, d, theta, f'θ = {theta:.2f}')
    
    # Add a main title
    fig.suptitle(f'Effect of Angle Parameter (θ) with R_0={R_0}, d={d}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def generate_combined_effect_figure(R_0, d_values, theta_values, filename):
    """Generate a figure showing the combined effect of different d and theta values"""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 grid of 3D subplots
    gs = GridSpec(2, 2, figure=fig)
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            axes.append(ax)
    
    # Plot combinations of d and theta values
    for i, (d, theta) in enumerate(zip(d_values, theta_values)):
        if i < len(axes):
            plot_3d_vectors_subplot(axes[i], R_0, d, theta, f'd = {d}, θ = {theta:.2f}')
    
    # Add a main title
    fig.suptitle(f'Combined Effect of Distance and Angle Parameters with R_0={R_0}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    """Main function to generate all parameter effect figures"""
    # Define parameter values
    R_0_values = [(0, 0, 0), (1, 1, 1), (0, 0, 2)]
    d_values = [0.5, 1.0, 1.5, 2.0]
    theta_values = [0, np.pi/6, np.pi/4, np.pi/3]
    combined_d_values = [0.5, 1.0, 1.5, 2.0]
    combined_theta_values = [0, np.pi/6, np.pi/4, np.pi/3]
    
    # Generate figures for each R_0 value
    for R_0 in R_0_values:
        R_0_str = '_'.join(map(str, R_0)).replace('.', 'p')
        
        # Generate d effect figure
        generate_d_effect_figure(
            R_0, 
            np.pi/4,  # Fixed theta
            d_values,
            f'figures/d_effect_R0_{R_0_str}.png'
        )
        
        # Generate theta effect figure
        generate_theta_effect_figure(
            R_0, 
            1.0,  # Fixed d
            theta_values,
            f'figures/theta_effect_R0_{R_0_str}.png'
        )
        
        # Generate combined effect figure
        generate_combined_effect_figure(
            R_0,
            combined_d_values,
            combined_theta_values,
            f'figures/combined_effect_R0_{R_0_str}.png'
        )

if __name__ == "__main__":
    main()
