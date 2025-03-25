#!/usr/bin/env python3
"""
Script to generate arrowhead matrices for theta values from 0 to 360 degrees in 5-degree steps
and plot the eigenvalues and eigenvectors without connecting the points.
Organizes output files into appropriate subdirectories.
"""

import os
import sys
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg

def organize_results_directory(results_dir):
    """
    Organize the results directory by file type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    
    Returns:
    --------
    dict
        Dictionary mapping file extensions to subdirectories
    """
    # Create subdirectories if they don't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Move files to appropriate subdirectories
    for file_path in glob.glob(os.path.join(results_dir, '*')):
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip('.')
        
        # Skip if extension not in our list
        if ext not in subdirs:
            continue
        
        # Get destination directory
        dest_dir = subdirs[ext]
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Move file to destination directory
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"Moved {filename} to {os.path.relpath(dest_path, results_dir)}")
    
    return subdirs

def get_file_path(results_dir, filename, file_type=None):
    """
    Get the appropriate path for a file based on its type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    filename : str
        Name of the file
    file_type : str, optional
        Type of the file (extension without the dot)
        If None, will be determined from the filename
    
    Returns:
    --------
    str
        Path to the file
    """
    # Create the directory structure if it doesn't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Determine file type if not provided
    if file_type is None:
        _, ext = os.path.splitext(filename)
        file_type = ext.lstrip('.')
    
    # Get the appropriate directory
    if file_type in subdirs:
        return os.path.join(subdirs[file_type], filename)
    else:
        return os.path.join(results_dir, filename)

class ArrowheadMatrixPlotter:
    """
    Class to generate arrowhead matrices and plot eigenvalues and eigenvectors
    without connecting the points.
    """
    
    def __init__(self, R_0=(0, 0, 0), d=0.5, coupling_constant=0.1, omega=1.0):
        """
        Initialize the ArrowheadMatrixPlotter.
        
        Parameters:
        -----------
        R_0 : tuple
            Origin point
        d : float
            Distance from origin
        coupling_constant : float
            Coupling constant for the matrix
        omega : float
            Omega value for the matrix
        """
        self.R_0 = R_0
        self.d = d
        self.coupling_constant = coupling_constant
        self.omega = omega
        self.matrix_size = 4
    
    def generate_r_vector(self, theta):
        """
        Generate the R_theta vector for a given theta value.
        
        Parameters:
        -----------
        theta : float
            Theta value in radians
            
        Returns:
        --------
        R_vector : tuple
            The R_theta vector
        """
        # Define basis vectors for the plane orthogonal to (1,1,1)
        basis1 = np.array([1, -1/2, -1/2]) / np.sqrt(1.5)  # Normalized
        basis2 = np.array([0, -1/2, 1/2]) / np.sqrt(0.5)   # Normalized
        
        # Generate the point using the parametric circle equation
        R_0 = np.array(self.R_0)
        displacement = self.d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
        R_vector = tuple(R_0 + displacement)
        
        return R_vector
    
    def generate_matrix(self, theta):
        """
        Generate the 4x4 arrowhead matrix for a given theta value.
        
        Parameters:
        -----------
        theta : float
            Theta value in radians
            
        Returns:
        --------
        matrix : numpy.ndarray
            The 4x4 arrowhead matrix
        """
        # Generate the R_theta vector
        R_vector = self.generate_r_vector(theta)
        
        # Initialize the matrix with zeros
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        
        # Set the diagonal elements
        matrix[0, 0] = 0
        matrix[1, 1] = self.omega**2 * R_vector[0]**2
        matrix[2, 2] = self.omega**2 * R_vector[1]**2
        matrix[3, 3] = self.omega**2 * R_vector[2]**2
        
        # Set the off-diagonal elements (arrowhead structure)
        for i in range(1, self.matrix_size):
            matrix[0, i] = matrix[i, 0] = self.coupling_constant
        
        return matrix, R_vector
    
    def calculate_eigenvalues_eigenvectors(self, matrix):
        """
        Calculate the eigenvalues and eigenvectors of the matrix.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            The arrowhead matrix
            
        Returns:
        --------
        eigenvalues : numpy.ndarray
            The eigenvalues of the matrix
        eigenvectors : numpy.ndarray
            The eigenvectors of the matrix
        """
        eigenvalues, eigenvectors = linalg.eigh(matrix)
        return eigenvalues, eigenvectors
    
    def plot_eigenvalues_no_connections(self, theta_values, all_eigenvalues, output_dir="./"):
        """
        Plot eigenvalues for different theta values in 3D without connecting the points.
        
        Parameters:
        -----------
        theta_values : list of float
            List of theta values in radians
        all_eigenvalues : list of numpy.ndarray
            List of eigenvalues for each theta
        output_dir : str
            Directory to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert theta values to degrees for display
        theta_values_deg = np.degrees(theta_values)
        
        # Prepare data for 3D scatter plot
        x = []  # Theta values
        y = []  # Eigenvalue indices
        z = []  # Eigenvalue values
        c = []  # Colors based on eigenvalue index
        
        # Colors for different eigenvalues
        colors = ['r', 'g', 'b', 'purple']
        
        # For each theta value
        for i, (theta, eigenvalues) in enumerate(zip(theta_values_deg, all_eigenvalues)):
            # For each eigenvalue
            for j, eigenvalue in enumerate(eigenvalues):
                x.append(theta)
                y.append(j)
                z.append(eigenvalue)
                c.append(colors[j])
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=c, marker='o', s=50)
        
        # Set labels and title
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Eigenvalue Index')
        ax.set_zlabel('Eigenvalue')
        ax.set_title('Eigenvalues vs Theta (No Connections)')
        
        # Set y-axis ticks to integers
        ax.set_yticks(range(4))
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                      markersize=10, label=f'Eigenvalue {i}') for i in range(4)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save the plot to the plots subdirectory
        plot_path = get_file_path(output_dir, "eigenvalues_no_connections.png", "png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_eigenvectors_no_connections(self, theta_values, all_eigenvectors, output_dir="./"):
        """
        Plot the eigenvector endpoints for all theta values in 3D space without connecting them.
        Shows all 4 eigenvector endpoints for each matrix.
        
        Parameters:
        -----------
        theta_values : list of float
            List of theta values in radians
        all_eigenvectors : list of numpy.ndarray
            List of eigenvectors for each theta
        output_dir : str
            Directory to save the plots
        """
        # Create a figure for all eigenvector endpoints
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert theta values to degrees for display
        theta_values_deg = np.degrees(theta_values)
        
        # Colors for different theta values
        cmap = plt.cm.viridis
        theta_colors = [cmap(i/len(theta_values)) for i in range(len(theta_values))]
        
        # Markers for different eigenvectors
        markers = ['o', 's', '^', 'd']  # circle, square, triangle, diamond
        
        # For each theta value
        for i, (theta, eigenvectors) in enumerate(zip(theta_values_deg, all_eigenvectors)):
            # For each eigenvector (0-3) of this matrix
            for ev_idx in range(4):
                # Get the eigenvector
                eigenvector = eigenvectors[:, ev_idx]
                
                # Plot the eigenvector endpoint
                ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                          c=[theta_colors[i]], marker=markers[ev_idx], s=100, 
                          alpha=0.8)
                
                # No text labels
        
        # Set labels and title
        ax.set_xlabel('X Component')
        ax.set_ylabel('Y Component')
        ax.set_zlabel('Z Component')
        ax.set_title('All Eigenvector Endpoints for Different Theta Values')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add a colorbar to show the theta values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
        cbar.set_label('Theta (degrees)')
        
        # Add a legend for the eigenvector markers
        legend_elements = [
            plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='gray', 
                      markersize=10, label=f'Eigenvector {i}') for i in range(4)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save the plot to the plots subdirectory
        plot_path = get_file_path(output_dir, "eigenvectors_no_connections.png", "png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots for each eigenvector (without connections)
        for ev_idx in range(4):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # For each theta value, plot the corresponding eigenvector endpoint
            for i, (theta, eigenvectors) in enumerate(zip(theta_values_deg, all_eigenvectors)):
                # Get the eigenvector for this eigenvalue index
                eigenvector = eigenvectors[:, ev_idx]
                
                # Plot the eigenvector endpoint
                ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                           c=[theta_colors[i]], marker='o', s=100)
                
                # No text labels
            
            # Set labels and title
            ax.set_xlabel('X Component')
            ax.set_ylabel('Y Component')
            ax.set_zlabel('Z Component')
            ax.set_title(f'Eigenvector {ev_idx} Endpoints for Different Theta Values')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Add a colorbar to show the theta values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
            cbar.set_label('Theta (degrees)')
            
            # Save the plot to the plots subdirectory
            plot_path = get_file_path(output_dir, f"eigenvector_{ev_idx}_no_connections.png", "png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """
    Main function to generate arrowhead matrices for different theta values,
    calculate eigenvalues and eigenvectors, and create plots without connections.
    """
    # Parameters
    R_0 = (0, 0, 0)
    d = 0.5
    theta_start = 0
    theta_end = 360  # Degrees
    theta_step = 5   # 5-degree steps
    coupling_constant = 0.1
    omega = 1.0
    
    # Generate theta values in degrees, then convert to radians for calculations
    theta_values_deg = np.arange(theta_start, theta_end, theta_step)
    theta_values = np.radians(theta_values_deg)  # Convert to radians for calculations
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize the results directory
    print("Organizing results directory...")
    organize_results_directory(output_dir)
    
    # Create the matrix generator
    matrix_generator = ArrowheadMatrixPlotter(
        R_0=R_0,
        d=d,
        coupling_constant=coupling_constant,
        omega=omega
    )
    
    # Lists to store R vectors, eigenvalues, and eigenvectors for all theta values
    r_vectors = []
    all_eigenvalues = []
    all_eigenvectors = []
    
    # Generate a matrix for each theta value
    for i, theta in enumerate(theta_values):
        print(f"\nGenerating matrix for theta {i} = {theta:.4f} radians ({theta_values_deg[i]:.1f} degrees)")
        
        # Generate the matrix and R vector
        matrix, R_vector = matrix_generator.generate_matrix(theta)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = matrix_generator.calculate_eigenvalues_eigenvectors(matrix)
        
        # Store the R vector, eigenvalues, and eigenvectors
        r_vectors.append(R_vector)
        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)
        
        # Save the matrix and eigenvalues/eigenvectors
        np.save(os.path.join(output_dir, f"arrowhead_matrix_4x4_theta_{i}.npy"), matrix)
        np.save(os.path.join(output_dir, f"eigenvalues_theta_{i}.npy"), eigenvalues)
        np.save(os.path.join(output_dir, f"eigenvectors_theta_{i}.npy"), eigenvectors)
    
    print(f"\nGenerated {len(theta_values)} matrices for different theta values.")
    
    # Create the plots without connections
    print("\nCreating plots without connections...")
    
    # Plot the eigenvalues without connections
    print("Plotting eigenvalues without connections...")
    matrix_generator.plot_eigenvalues_no_connections(theta_values, all_eigenvalues, output_dir)
    
    # Plot the eigenvectors without connections
    print("Plotting eigenvectors without connections...")
    matrix_generator.plot_eigenvectors_no_connections(theta_values, all_eigenvectors, output_dir)
    
    # Organize the results directory again after creating the plots
    print("\nOrganizing results directory...")
    organize_results_directory(output_dir)
    
    plots_dir = os.path.join(output_dir, 'plots')
    numpy_dir = os.path.join(output_dir, 'numpy')
    
    print("\nAll plots saved to the plots directory:")
    print(f"  - {os.path.join(plots_dir, 'eigenvalues_no_connections.png')}")
    print(f"  - {os.path.join(plots_dir, 'eigenvectors_no_connections.png')}")
    for i in range(4):
        print(f"  - {os.path.join(plots_dir, f'eigenvector_{i}_no_connections.png')}")
    
    print("\nAll data files saved to the numpy directory.")

if __name__ == "__main__":
    main()
