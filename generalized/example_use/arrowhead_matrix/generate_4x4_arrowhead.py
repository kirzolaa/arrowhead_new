#!/usr/bin/env python3
"""
Generate a 4x4 arrowhead matrix based on orthogonal vectors.

This script generates a 4x4 arrowhead matrix where:
- The first diagonal element (D_00) is the sum of all VX potentials plus h*ω
- The rest of the diagonal elements follow the pattern:
  D_11 = V_a(R0) + V_x(R1) + V_x(R2)
  D_22 = V_x(R0) + V_a(R1) + V_x(R2)
  D_33 = V_x(R0) + V_x(R1) + V_a(R2)
- The off-diagonal elements are coupling constants
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar  # Reduced Planck constant
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vector_utils import create_perfect_orthogonal_vectors, generate_R_vector


class ArrowheadMatrix4x4:
    """
    Class to generate and manipulate a 4x4 arrowhead matrix based on a single orthogonal vector.
    """
    
    def __init__(self, R_0=(0, 0, 0), d=0.02, theta=0, 
                 coupling_constant=0.1, omega=0.005, perfect=True):
        """
        Initialize the ArrowheadMatrix4x4 generator for a single theta value.
        
        Parameters:
        -----------
        R_0 : tuple
            Origin vector (x, y, z)
        d : float
            Distance parameter (lowered from 0.1 to 0.05 to make potentials more separate)
        theta : float
            Theta value in radians
        coupling_constant : float
            Coupling constant for off-diagonal elements
        omega : float
            Angular frequency for the energy term h*ω
        perfect : bool
            Whether to use perfect circle generation method
        """
        self.R_0 = np.array(R_0)
        self.d = d
        self.theta = theta
        self.coupling_constant = coupling_constant
        self.omega = omega
        self.perfect = perfect
        
        # Generate the R vector for this theta
        if perfect:
            self.R_vector = create_perfect_orthogonal_vectors(self.R_0, self.d, theta)
        else:
            self.R_vector = generate_R_vector(self.R_0, self.d, theta)
        
        # Matrix size is 4x4
        self.matrix_size = 4
        
    def potential_vx(self, R):
        """
        Calculate the VX potential at position R.
        This is a parabolic potential: a*x^2 + b*x + c
        
        Parameters:
        -----------
        R : numpy.ndarray
            Position vector (x, y, z)
            
        Returns:
        --------
        float
            Potential value at position R
        """
        # Parabolic potential with reduced curvature to create smoother landscape
        a = 0.05  # Modified curvature parameter for VX
        b = 0
        c = 0
        
        # Use magnitude of R for a more isotropic potential
        r_mag = np.linalg.norm(R)
        return a * r_mag**2 + b * r_mag + c
    
    def potential_va(self, R):
        """
        Calculate the VA potential at position R.
        This is exactly like the VX potential but shifted on both x and y axes
        
        Parameters:
        -----------
        R : numpy.ndarray
            Position vector (x, y, z)
            
        Returns:
        --------
        float
            Potential value at position R
        """
        a = 0.2  # Modified curvature parameter for VA
        x_shift = 22.5  # Modified x-shift value
        y_shift = 567.7222222222222  # Modified y-shift value
        c = 0  # Same vertical shift as VX
        
        # Create a shifted version of VX
        # We shift the input coordinates before calculating the potential
        x, y, z = R
        shifted_R = np.array([x - x_shift, y - y_shift, z])
        
        # Calculate the potential using the same formula as VX
        r_mag = np.linalg.norm(shifted_R)
        return a * r_mag**2 + c
        
    
    def generate_matrix(self):
        """
        Generate the 4x4 arrowhead matrix based on a single R vector's components.
        
        Returns:
        --------
        numpy.ndarray
            The 4x4 arrowhead matrix
        """
        # Initialize the matrix with zeros
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        
        # Extract the three components of the R vector
        R0 = np.array([self.R_vector[0], 0, 0])  # x component
        R1 = np.array([0, self.R_vector[1], 0])  # y component
        R2 = np.array([0, 0, self.R_vector[2]])  # z component
        
        # Calculate VX and VA for each component
        vx0 = self.potential_vx(R0)
        vx1 = self.potential_vx(R1)
        vx2 = self.potential_vx(R2)
        
        va0 = self.potential_va(R0)
        va1 = self.potential_va(R1)
        va2 = self.potential_va(R2)
        
        # Calculate the sum of all VX potentials
        vxx = vx0 + vx1 + vx2
        
        # Set the first diagonal element (D_00)
        # D_00 = VXX + h*ω
        matrix[0, 0] = vxx + hbar * self.omega
        
        # Set the rest of the diagonal elements
        # D_11 = V_a(R0) + V_x(R1) + V_x(R2)
        matrix[1, 1] = va0 + vx1 + vx2
        
        # D_22 = V_x(R0) + V_a(R1) + V_x(R2)
        matrix[2, 2] = vx0 + va1 + vx2
        
        # D_33 = V_x(R0) + V_x(R1) + V_a(R2)
        matrix[3, 3] = vx0 + vx1 + va2
        
        # Set the off-diagonal elements (coupling constants)
        for i in range(1, self.matrix_size):
            matrix[0, i] = self.coupling_constant
            matrix[i, 0] = self.coupling_constant
        
        return matrix
    
    def print_matrix_details(self, matrix):
        """
        Print detailed information about the matrix.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            The arrowhead matrix
        """
        print("4x4 Arrowhead Matrix Details:")
        print("-----------------------------")
        print(f"Origin vector R_0: {self.R_0}")
        print(f"Distance parameter d: {self.d}")
        print(f"Theta value: {self.theta} radians")
        print(f"Coupling constant: {self.coupling_constant}")
        print(f"Angular frequency ω: {self.omega}")
        print(f"Reduced Planck constant ħ: {hbar}")
        print(f"Energy term ħω: {hbar * self.omega}")
        print("\nGenerated R vector:")
        print(f"  R (θ = {self.theta:.4f}): {self.R_vector}")
        
        # Extract the three components of the R vector
        R0 = np.array([self.R_vector[0], 0, 0])  # x component
        R1 = np.array([0, self.R_vector[1], 0])  # y component
        R2 = np.array([0, 0, self.R_vector[2]])  # z component
        
        # Calculate VX and VA for each component
        vx0 = self.potential_vx(R0)
        vx1 = self.potential_vx(R1)
        vx2 = self.potential_vx(R2)
        
        va0 = self.potential_va(R0)
        va1 = self.potential_va(R1)
        va2 = self.potential_va(R2)
        
        # Calculate the sum of all VX potentials
        vxx = vx0 + vx1 + vx2
        
        print("\nComponent-wise potential values:")
        print(f"  R0 (x component): VX = {vx0:.4f}, VA = {va0:.4f}")
        print(f"  R1 (y component): VX = {vx1:.4f}, VA = {va1:.4f}")
        print(f"  R2 (z component): VX = {vx2:.4f}, VA = {va2:.4f}")
        print(f"  VXX (sum of all VX): {vxx:.4f}")
        
        print("\nDiagonal elements:")
        print(f"  D_00 = VXX + ħω = {vxx:.4f} + {hbar * self.omega} = {matrix[0, 0]}")
        print(f"  D_11 = VA(R0) + VX(R1) + VX(R2) = {va0:.4f} + {vx1:.4f} + {vx2:.4f} = {matrix[1, 1]}")
        print(f"  D_22 = VX(R0) + VA(R1) + VX(R2) = {vx0:.4f} + {va1:.4f} + {vx2:.4f} = {matrix[2, 2]}")
        print(f"  D_33 = VX(R0) + VX(R1) + VA(R2) = {vx0:.4f} + {vx1:.4f} + {va2:.4f} = {matrix[3, 3]:.16f}")
        
        print("\nArrowhead Matrix:")
        # Set NumPy print options to show more decimal places
        np.set_printoptions(precision=8)
        print(matrix)
        # Reset NumPy print options to default
        np.set_printoptions(precision=8, suppress=True)
    
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
    
    def plot_r_vectors(self, r_vectors, output_dir="./"):
        """
        Plot the R_theta vector endpoints in 3D.
        
        Parameters:
        -----------
        r_vectors : list of numpy.ndarray
            List of R_theta vectors
        output_dir : str
            Directory to save the plot
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each R_theta vector endpoint
        x_points = [r[0] for r in r_vectors]
        y_points = [r[1] for r in r_vectors]
        z_points = [r[2] for r in r_vectors]
        
        ax.scatter(x_points, y_points, z_points, c='r', marker='o', s=100, label='R_theta endpoints')
        
        # Plot the origin
        ax.scatter([0], [0], [0], c='b', marker='o', s=100, label='Origin')
        
        # Plot the x=y=z line
        line = np.linspace(-1, 1, 100)
        ax.plot(line, line, line, 'g--', label='x=y=z line')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('R_theta Vector Endpoints')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, "r_vectors_3d.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_eigenvalues(self, theta_values, all_eigenvalues, output_dir="./"):
        """
        Plot the eigenvalues for each theta value in 3D.
        
        Parameters:
        -----------
        theta_values : list of float
            List of theta values
        all_eigenvalues : list of numpy.ndarray
            List of eigenvalues for each theta
        output_dir : str
            Directory to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a meshgrid for theta and eigenvalue index
        theta_grid, index_grid = np.meshgrid(theta_values, np.arange(4))
        
        # Prepare the eigenvalues data
        z_data = np.array([eigenvalues for eigenvalues in all_eigenvalues]).T
        
        # Plot the eigenvalues as a surface
        surf = ax.plot_surface(theta_grid, index_grid, z_data, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Eigenvalue')
        
        # Set labels and title
        ax.set_xlabel('Theta (radians)')
        ax.set_ylabel('Eigenvalue Index')
        ax.set_zlabel('Eigenvalue')
        ax.set_title('Eigenvalues vs Theta')
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, "eigenvalues_3d.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_eigenvectors(self, theta_values, all_eigenvectors, output_dir="./"):
        """
        Plot the eigenvector endpoints for all theta values in 3D space.
        Creates one plot per eigenvalue, showing how the eigenvector changes with theta.
        
        Parameters:
        -----------
        theta_values : list of float
            List of theta values
        all_eigenvectors : list of numpy.ndarray
            List of eigenvectors for each theta
        output_dir : str
            Directory to save the plots
        """
        # Colors for different theta values
        cmap = plt.cm.viridis
        colors = [cmap(i/len(theta_values)) for i in range(len(theta_values))]
        
        # Create a separate plot for each eigenvalue
        for ev_idx in range(4):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # For each theta value, plot the corresponding eigenvector endpoint
            for i, (theta, eigenvectors) in enumerate(zip(theta_values, all_eigenvectors)):
                # Get the eigenvector for this eigenvalue index
                eigenvector = eigenvectors[:, ev_idx]
                
                # Plot the eigenvector endpoint
                ax.scatter(eigenvector[1], eigenvector[2], eigenvector[3], 
                           c=[colors[i]], marker='o', s=100, 
                           label=f'θ = {theta:.2f}')
                
                # Add text label with theta value
                ax.text(eigenvector[1], eigenvector[2], eigenvector[3], 
                        f'{theta:.2f}', size=8)
            
            # Connect all points with a line to show the path as theta changes
            x_points = [eigenvectors[:, ev_idx][1] for eigenvectors in all_eigenvectors]
            y_points = [eigenvectors[:, ev_idx][2] for eigenvectors in all_eigenvectors]
            z_points = [eigenvectors[:, ev_idx][3] for eigenvectors in all_eigenvectors]
            
            # Add the first point again to close the loop
            x_points.append(x_points[0])
            y_points.append(y_points[0])
            z_points.append(z_points[0])
            
            ax.plot(x_points, y_points, z_points, 'k-', alpha=0.5)
            
            # Set labels and title
            ax.set_xlabel('X (First Component)')
            ax.set_ylabel('Y (Second Component)')
            ax.set_zlabel('Z (Third Component)')
            ax.set_title(f'Eigenvector {ev_idx} Endpoints for Different Theta Values')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Add a colorbar to show the theta values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2*np.pi))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
            cbar.set_label('Theta (radians)')
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f"eigenvector_{ev_idx}_3d.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined plot showing all eigenvectors
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for different eigenvalues
        ev_colors = ['r', 'g', 'b', 'purple']
        
        # For each eigenvalue index, plot all theta points
        for ev_idx in range(4):
            x_points = [eigenvectors[:, ev_idx][1] for eigenvectors in all_eigenvectors]
            y_points = [eigenvectors[:, ev_idx][2] for eigenvectors in all_eigenvectors]
            z_points = [eigenvectors[:, ev_idx][3] for eigenvectors in all_eigenvectors]
            
            # Add the first point again to close the loop
            x_points.append(x_points[0])
            y_points.append(y_points[0])
            z_points.append(z_points[0])
            
            # Plot the points and connect them with lines
            ax.scatter(x_points[:-1], y_points[:-1], z_points[:-1], 
                       c=ev_colors[ev_idx], marker='o', s=50, 
                       label=f'Eigenvector {ev_idx}')
            ax.plot(x_points, y_points, z_points, c=ev_colors[ev_idx], linestyle='-', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (First Component)')
        ax.set_ylabel('Y (Second Component)')
        ax.set_zlabel('Z (Third Component)')
        ax.set_title('All Eigenvector Endpoints in 3D Space')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add a legend
        ax.legend()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, "all_eigenvectors_3d.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, matrix, output_dir="./", theta_index=0):
        """
        Save the matrix to files and calculate eigenvalues and eigenvectors.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            The arrowhead matrix
        output_dir : str
            Directory to save the results
        theta_index : int
            Index of the theta value, used for filename
            
        Returns:
        --------
        eigenvalues : numpy.ndarray
            The eigenvalues of the matrix
        eigenvectors : numpy.ndarray
            The eigenvectors of the matrix
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.calculate_eigenvalues_eigenvectors(matrix)
        
        # Save the matrix as a numpy file
        np.save(os.path.join(output_dir, f"arrowhead_matrix_4x4_theta_{theta_index}.npy"), matrix)
        
        # Save the matrix as a CSV file
        np.savetxt(os.path.join(output_dir, f"arrowhead_matrix_4x4_theta_{theta_index}.csv"), matrix, delimiter=",")
        
        # Save eigenvalues and eigenvectors
        np.save(os.path.join(output_dir, f"eigenvalues_theta_{theta_index}.npy"), eigenvalues)
        np.save(os.path.join(output_dir, f"eigenvectors_theta_{theta_index}.npy"), eigenvectors)
        
        # Save the matrix details to a text file
        with open(os.path.join(output_dir, f"arrowhead_matrix_4x4_theta_{theta_index}_details.txt"), "w") as f:
            # Redirect stdout to the file
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_matrix_details(matrix)
            print("\nEigenvalues:")
            for i, eigenvalue in enumerate(eigenvalues):
                print(f"  λ_{i}: {eigenvalue}")
            
            print("\nEigenvectors (columns):")
            for i in range(len(eigenvectors)):
                print(f"  v_{i}: {eigenvectors[:, i]}")
            sys.stdout = original_stdout
            
        return eigenvalues, eigenvectors


def main():
    """
    Main function to generate 4x4 arrowhead matrices for different theta values,
    calculate eigenvalues and eigenvectors, and create plots.
    """
    # Parameters
    R_0 = (0, 0, 0)
    d = 0.5
    theta_start = 0
    theta_end = 360  # Degrees
    theta_step = 1   # Modified theta step
    coupling_constant = 0.1
    omega = 1.0
    
    # Generate theta values in degrees, then convert to radians for calculations
    # Add 360 degrees to ensure we complete the full cycle
    theta_values_deg = np.append(np.arange(theta_start, theta_end, theta_step), 360.0)
    theta_values = np.radians(theta_values_deg)  # Convert to radians for calculations
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store R vectors, eigenvalues, and eigenvectors for all theta values
    r_vectors = []
    all_eigenvalues = []
    all_eigenvectors = []
    
    # Generate a matrix for each theta value
    for i, theta in enumerate(theta_values):
        print(f"\nGenerating matrix for theta {i} = {theta:.4f} radians")
        
        # Create the arrowhead matrix generator for this theta
        arrowhead = ArrowheadMatrix4x4(
            R_0=R_0,
            d=d,
            theta=theta,
            coupling_constant=coupling_constant,
            omega=omega,
            perfect=True
        )
        
        # Generate the matrix
        matrix = arrowhead.generate_matrix()
        
        
        # Print the matrix details
        arrowhead.print_matrix_details(matrix)
        
        # Save the results and get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = arrowhead.save_results(matrix, output_dir, i)
        
        # Store the R vector, eigenvalues, and eigenvectors
        r_vectors.append(arrowhead.R_vector)
        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)
    
    print(f"\nGenerated {len(theta_values)} matrices for different theta values.")
    
    # Create the plots
    print("\nCreating plots...")
    
    # Create an instance of ArrowheadMatrix4x4 to use the plotting methods
    arrowhead = ArrowheadMatrix4x4(R_0=R_0, d=d, theta=0, coupling_constant=coupling_constant, omega=omega)
    
    # Plot the R_theta vector endpoints
    print("Plotting R_theta vector endpoints...")
    arrowhead.plot_r_vectors(r_vectors, output_dir)
    
    # Plot the eigenvalues
    print("Plotting eigenvalues...")
    arrowhead.plot_eigenvalues(theta_values, all_eigenvalues, output_dir)
    
    # Plot the eigenvectors
    print("Plotting eigenvectors...")
    arrowhead.plot_eigenvectors(theta_values, all_eigenvectors, output_dir)
    
    print("\nAll plots saved to the results directory.")


if __name__ == "__main__":
    main()
