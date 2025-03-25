#!/usr/bin/env python3
"""
Arrowhead Matrix Generator and Analyzer

This script provides a unified interface for generating, analyzing, and visualizing
arrowhead matrices. It combines the functionality of the separate scripts into a
single, easy-to-use tool.

Features:
- Generate arrowhead matrices of any size
- Calculate eigenvalues and eigenvectors
- Create 2D and 3D visualizations
- Save results in organized directories
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the path so we can import the modules
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(module_dir, '../..')))
sys.path.append(module_dir)  # Add current directory to path

from vector_utils import create_perfect_orthogonal_vectors, generate_R_vector

# Import local modules
from file_utils import organize_results_directory, get_file_path
from generate_arrowhead_matrix import ArrowheadMatrix
from generate_4x4_arrowhead import ArrowheadMatrix4x4
from plot_improved import plot_eigenvalues_2d, plot_eigenvectors_no_labels


class ArrowheadMatrixAnalyzer:
    """
    A unified class for generating, analyzing, and visualizing arrowhead matrices.
    """
    
    def __init__(self, 
                 R_0=(0, 0, 0), 
                 d=0.5, 
                 theta_start=0, 
                 theta_end=2*np.pi, 
                 theta_steps=72,
                 coupling_constant=0.1, 
                 omega=1.0, 
                 matrix_size=4, 
                 perfect=True,
                 output_dir=None):
        """
        Initialize the ArrowheadMatrixAnalyzer.
        
        Parameters:
        -----------
        R_0 : tuple
            Origin vector (x, y, z)
        d : float
            Distance parameter
        theta_start : float
            Starting theta value in radians
        theta_end : float
            Ending theta value in radians
        theta_steps : int
            Number of theta values to generate matrices for
        coupling_constant : float
            Coupling constant for off-diagonal elements
        omega : float
            Angular frequency for the energy term h*ω
        matrix_size : int
            Size of the matrix to generate
        perfect : bool
            Whether to use perfect circle generation method
        output_dir : str
            Directory to save results (default is the current script directory)
        """
        self.R_0 = R_0
        self.d = d
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.theta_steps = theta_steps
        self.coupling_constant = coupling_constant
        self.omega = omega
        self.matrix_size = matrix_size
        self.perfect = perfect
        
        # Generate theta values
        self.theta_values = np.linspace(theta_start, theta_end, theta_steps, endpoint=False)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Organize the results directory
        organize_results_directory(self.output_dir)
        
        # Initialize data structures for results
        self.matrices = []
        self.all_eigenvalues = []
        self.all_eigenvectors = []
    
    def generate_matrices(self):
        """
        Generate arrowhead matrices for all theta values.
        
        Returns:
        --------
        list
            List of generated matrices
        """
        print(f"Generating {self.theta_steps} matrices for different theta values...")
        
        self.matrices = []
        
        for i, theta in enumerate(self.theta_values):
            print(f"\nGenerating matrix for theta {i} = {theta:.4f} radians")
            
            if self.matrix_size == 4:
                # Use the specialized 4x4 implementation
                arrowhead = ArrowheadMatrix4x4(
                    R_0=self.R_0,
                    d=self.d,
                    theta=theta,
                    coupling_constant=self.coupling_constant,
                    omega=self.omega,
                    perfect=self.perfect
                )
            else:
                # Use the general implementation
                arrowhead = ArrowheadMatrix(
                    R_0=self.R_0,
                    d=self.d,
                    theta=theta,
                    coupling_constant=self.coupling_constant,
                    omega=self.omega,
                    perfect=self.perfect,
                    matrix_size=self.matrix_size
                )
            
            # Generate the matrix
            matrix = arrowhead.generate_matrix()
            
            # Print the matrix details
            arrowhead.print_matrix_details(matrix)
            
            # Save the results
            arrowhead.save_results(matrix, self.output_dir, i)
            
            self.matrices.append(matrix)
        
        print(f"\nGenerated {self.theta_steps} matrices for different theta values.")
        return self.matrices
    
    def calculate_eigenvalues_eigenvectors(self):
        """
        Calculate eigenvalues and eigenvectors for all matrices.
        
        Returns:
        --------
        tuple
            (all_eigenvalues, all_eigenvectors)
        """
        print("\nCalculating eigenvalues and eigenvectors...")
        
        self.all_eigenvalues = []
        self.all_eigenvectors = []
        
        for i, matrix in enumerate(self.matrices):
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = linalg.eigh(matrix)
            
            # Sort by eigenvalues
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            self.all_eigenvalues.append(eigenvalues)
            self.all_eigenvectors.append(eigenvectors)
            
            # Save eigenvalues and eigenvectors
            np_dir = os.path.join(self.output_dir, 'numpy')
            os.makedirs(np_dir, exist_ok=True)
            
            np.save(os.path.join(np_dir, f"eigenvalues_theta_{i}.npy"), eigenvalues)
            np.save(os.path.join(np_dir, f"eigenvectors_theta_{i}.npy"), eigenvectors)
        
        print(f"Calculated eigenvalues and eigenvectors for {len(self.matrices)} matrices.")
        return self.all_eigenvalues, self.all_eigenvectors
    
    def load_results(self):
        """
        Load previously calculated eigenvalues and eigenvectors.
        
        Returns:
        --------
        tuple
            (all_eigenvalues, all_eigenvectors)
        """
        print("\nLoading previously calculated eigenvalues and eigenvectors...")
        
        self.all_eigenvalues = []
        self.all_eigenvectors = []
        
        # Find all eigenvalue files
        i = 0
        numpy_dir = os.path.join(self.output_dir, 'numpy')
        
        while True:
            eigenvalues_file = os.path.join(numpy_dir, f"eigenvalues_theta_{i}.npy")
            eigenvectors_file = os.path.join(numpy_dir, f"eigenvectors_theta_{i}.npy")
            
            if not os.path.exists(eigenvalues_file) or not os.path.exists(eigenvectors_file):
                break
                
            # Load the data
            eigenvalues = np.load(eigenvalues_file)
            eigenvectors = np.load(eigenvectors_file)
            
            self.all_eigenvalues.append(eigenvalues)
            self.all_eigenvectors.append(eigenvectors)
            
            i += 1
        
        if not self.all_eigenvalues:
            print("No eigenvalue/eigenvector files found. Please run generate_matrices() first.")
            return None, None
        
        # Update theta values if needed
        if len(self.all_eigenvalues) != len(self.theta_values):
            self.theta_values = np.linspace(self.theta_start, self.theta_end, len(self.all_eigenvalues), endpoint=False)
        
        print(f"Loaded {len(self.all_eigenvalues)} sets of eigenvalues and eigenvectors.")
        return self.all_eigenvalues, self.all_eigenvectors
    
    def create_plots(self):
        """
        Create plots for eigenvalues and eigenvectors.
        """
        print("\nCreating plots...")
        
        if not self.all_eigenvalues or not self.all_eigenvectors:
            eigenvalues, eigenvectors = self.load_results()
            if eigenvalues is None:
                return
        
        # Create 2D plots for eigenvalues
        print("Creating 2D eigenvalue plots...")
        plot_eigenvalues_2d(self.theta_values, self.all_eigenvalues, self.output_dir)
        
        # Create plots for eigenvectors without labels
        print("Creating eigenvector plots without labels...")
        plot_eigenvectors_no_labels(self.theta_values, self.all_eigenvectors, self.output_dir)
        
        plots_dir = os.path.join(self.output_dir, 'plots')
        print("Plots created successfully:")
        print(f"  - {os.path.join(plots_dir, 'all_eigenvalues_2d.png')}")
        for i in range(min(4, self.matrix_size)):
            print(f"  - {os.path.join(plots_dir, f'eigenvalue_{i}_2d.png')}")
        print(f"  - {os.path.join(plots_dir, 'eigenvectors_no_labels.png')}")
        for i in range(min(4, self.matrix_size)):
            print(f"  - {os.path.join(plots_dir, f'eigenvector_{i}_no_labels.png')}")
    
    def plot_r_vectors(self):
        """
        Create a 3D plot of the R vectors.
        """
        print("\nCreating R vectors plot...")
        
        # Generate R vectors
        r_vectors = []
        for theta in self.theta_values:
            r_vector = generate_R_vector(self.R_0, self.d, theta, perfect=self.perfect)
            r_vectors.append(r_vector)
        
        r_vectors = np.array(r_vectors)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the R vectors
        ax.scatter(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], 
                   c=np.arange(len(r_vectors)), cmap='hsv', s=30)
        
        # Plot the origin
        origin = np.array(self.R_0)
        ax.scatter([origin[0]], [origin[1]], [origin[2]], c='black', s=100, marker='o')
        
        # Plot the x=y=z line
        line_points = np.array([[-1, -1, -1], [1, 1, 1]])
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'k--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('R Vectors in 3D Space')
        
        # Set equal aspect ratio
        max_range = np.array([
            r_vectors[:, 0].max() - r_vectors[:, 0].min(),
            r_vectors[:, 1].max() - r_vectors[:, 1].min(),
            r_vectors[:, 2].max() - r_vectors[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (r_vectors[:, 0].max() + r_vectors[:, 0].min()) * 0.5
        mid_y = (r_vectors[:, 1].max() + r_vectors[:, 1].min()) * 0.5
        mid_z = (r_vectors[:, 2].max() + r_vectors[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save the plot
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'r_vectors_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"R vectors plot created: {os.path.join(plots_dir, 'r_vectors_3d.png')}")
    
    def run_all(self):
        """
        Run the complete analysis pipeline:
        1. Generate matrices
        2. Calculate eigenvalues and eigenvectors
        3. Create plots
        4. Plot R vectors
        """
        self.generate_matrices()
        self.calculate_eigenvalues_eigenvectors()
        self.create_plots()
        self.plot_r_vectors()
        print("\nComplete analysis finished successfully!")


def main():
    """
    Main function to parse command line arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(description='Arrowhead Matrix Generator and Analyzer')
    
    parser.add_argument('--r0', type=float, nargs=3, default=[0, 0, 0],
                        help='Origin vector (x, y, z)')
    parser.add_argument('--d', type=float, default=0.5,
                        help='Distance parameter')
    parser.add_argument('--theta-start', type=float, default=0,
                        help='Starting theta value in radians')
    parser.add_argument('--theta-end', type=float, default=2*np.pi,
                        help='Ending theta value in radians')
    parser.add_argument('--theta-steps', type=int, default=72,
                        help='Number of theta values to generate matrices for')
    parser.add_argument('--coupling', type=float, default=0.1,
                        help='Coupling constant for off-diagonal elements')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Angular frequency for the energy term h*ω')
    parser.add_argument('--size', type=int, default=4,
                        help='Size of the matrix to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--load-only', action='store_true',
                        help='Only load existing results and create plots')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only create plots from existing results')
    parser.add_argument('--perfect', action='store_true', default=True,
                        help='Whether to use perfect circle generation method')
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = ArrowheadMatrixAnalyzer(
        R_0=tuple(args.r0),
        d=args.d,
        theta_start=args.theta_start,
        theta_end=args.theta_end,
        theta_steps=args.theta_steps,
        coupling_constant=args.coupling,
        omega=args.omega,
        matrix_size=args.size,
        perfect=args.perfect,
        output_dir=args.output_dir
    )
    
    if args.plot_only:
        # Only create plots
        analyzer.load_results()
        analyzer.create_plots()
        analyzer.plot_r_vectors()
    elif args.load_only:
        # Load results and create plots
        analyzer.load_results()
        analyzer.create_plots()
        analyzer.plot_r_vectors()
    else:
        # Run the complete analysis
        analyzer.run_all()


if __name__ == "__main__":
    main()
