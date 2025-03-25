#!/usr/bin/env python3
"""
Generate arrowhead matrices based on orthogonal vectors.

This script generates arrowhead matrices where:
- The first diagonal element (D_00) is the sum of all VX potentials plus h*ω
- The rest of the diagonal elements are D_ii = VXX + VA(R[i-1]) - VX(R[i-1])
- The off-diagonal elements are coupling constants
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar  # Reduced Planck constant

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vector_utils import create_perfect_orthogonal_vectors, generate_R_vector


class ArrowheadMatrix:
    """
    Class to generate and manipulate arrowhead matrices based on a single orthogonal vector.
    """
    
    def __init__(self, R_0=(0, 0, 0), d=0.5, theta=0, 
                 coupling_constant=0.1, omega=1.0, perfect=True, matrix_size=4):
        """
        Initialize the ArrowheadMatrix generator for a single theta value.
        
        Parameters:
        -----------
        R_0 : tuple
            Origin vector (x, y, z)
        d : float
            Distance parameter
        theta : float
            Theta value in radians
        coupling_constant : float
            Coupling constant for off-diagonal elements
        omega : float
            Angular frequency for the energy term h*ω
        perfect : bool
            Whether to use perfect circle generation method
        matrix_size : int
            Size of the matrix to generate (default 4x4)
        """
        self.R_0 = np.array(R_0)
        self.d = d
        self.theta = theta
        self.coupling_constant = coupling_constant
        self.omega = omega
        self.perfect = perfect
        self.matrix_size = matrix_size
        
        # Generate the R vector for this theta
        if perfect:
            self.R_vector = create_perfect_orthogonal_vectors(self.R_0, self.d, theta)
        else:
            self.R_vector = generate_R_vector(self.R_0, self.d, theta)
        
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
        # Example parabolic potential: 0.5*x^2
        a = 0.5
        b = 0
        c = 0
        return a * R[0]**2 + b * R[0] + c
    
    def potential_va(self, R):
        """
        Calculate the VA potential at position R.
        This is a shifted parabolic potential: a*(x-x0)^2 + b*(x-x0) + c
        
        Parameters:
        -----------
        R : numpy.ndarray
            Position vector (x, y, z)
            
        Returns:
        --------
        float
            Potential value at position R
        """
        # Example shifted parabolic potential: 0.5*(x-1)^2
        a = 0.5
        b = 0
        c = 0
        x0 = 1.0  # Shift parameter
        return a * (R[0] - x0)**2 + b * (R[0] - x0) + c
    
    def generate_matrix(self):
        """
        Generate the arrowhead matrix based on a single R vector's components.
        
        Returns:
        --------
        numpy.ndarray
            The arrowhead matrix
        """
        # Initialize the matrix with zeros
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        
        # Extract the three components of the R vector
        R0 = np.array([self.R_vector[0], 0, 0])  # x component
        R1 = np.array([0, self.R_vector[1], 0])  # y component
        R2 = np.array([0, 0, self.R_vector[2]])  # z component
        
        # Calculate VX and VA for each component
        vx_values = []
        va_values = []
        
        # Calculate for each component
        for i in range(3):
            if i == 0:
                vx_values.append(self.potential_vx(R0))
                va_values.append(self.potential_va(R0))
            elif i == 1:
                vx_values.append(self.potential_vx(R1))
                va_values.append(self.potential_va(R1))
            elif i == 2:
                vx_values.append(self.potential_vx(R2))
                va_values.append(self.potential_va(R2))
        
        # Calculate the sum of all VX potentials
        vxx = sum(vx_values)
        
        # Set the first diagonal element (D_00)
        # D_00 = VXX + h*ω
        matrix[0, 0] = vxx + hbar * self.omega
        
        # Set the rest of the diagonal elements
        for i in range(1, min(4, self.matrix_size)):
            if i == 1:
                # D_11 = V_a(R0) + V_x(R1) + V_x(R2)
                matrix[i, i] = va_values[0] + vx_values[1] + vx_values[2]
            elif i == 2:
                # D_22 = V_x(R0) + V_a(R1) + V_x(R2)
                matrix[i, i] = vx_values[0] + va_values[1] + vx_values[2]
            elif i == 3:
                # D_33 = V_x(R0) + V_x(R1) + V_a(R2)
                matrix[i, i] = vx_values[0] + vx_values[1] + va_values[2]
            else:
                # For any additional diagonal elements, use a similar pattern
                matrix[i, i] = vxx + va_values[i % 3] - vx_values[i % 3]
        
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
        print("Arrowhead Matrix Details:")
        print("-------------------------")
        print(f"Matrix size: {self.matrix_size}x{self.matrix_size}")
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
        vx_values = []
        va_values = []
        
        # Calculate for each component
        for i in range(3):
            if i == 0:
                vx_values.append(self.potential_vx(R0))
                va_values.append(self.potential_va(R0))
            elif i == 1:
                vx_values.append(self.potential_vx(R1))
                va_values.append(self.potential_va(R1))
            elif i == 2:
                vx_values.append(self.potential_vx(R2))
                va_values.append(self.potential_va(R2))
        
        # Calculate the sum of all VX potentials
        vxx = sum(vx_values)
        
        print("\nComponent-wise potential values:")
        print(f"  R0 (x component): VX = {vx_values[0]:.4f}, VA = {va_values[0]:.4f}")
        print(f"  R1 (y component): VX = {vx_values[1]:.4f}, VA = {va_values[1]:.4f}")
        print(f"  R2 (z component): VX = {vx_values[2]:.4f}, VA = {va_values[2]:.4f}")
        print(f"  VXX (sum of all VX): {vxx:.4f}")
        
        print("\nDiagonal elements:")
        print(f"  D_00 = VXX + ħω = {vxx:.4f} + {hbar * self.omega} = {matrix[0, 0]}")
        
        for i in range(1, min(4, self.matrix_size)):
            if i == 1:
                print(f"  D_{i}{i} = VA(R0) + VX(R1) + VX(R2) = {va_values[0]:.4f} + {vx_values[1]:.4f} + {vx_values[2]:.4f} = {matrix[i, i]}")
            elif i == 2:
                print(f"  D_{i}{i} = VX(R0) + VA(R1) + VX(R2) = {vx_values[0]:.4f} + {va_values[1]:.4f} + {vx_values[2]:.4f} = {matrix[i, i]}")
            elif i == 3:
                print(f"  D_{i}{i} = VX(R0) + VX(R1) + VA(R2) = {vx_values[0]:.4f} + {vx_values[1]:.4f} + {va_values[2]:.4f} = {matrix[i, i]}")
        
        # For any additional diagonal elements beyond the first 4
        for i in range(4, self.matrix_size):
            print(f"  D_{i}{i} = VXX + VA(R{i%3}) - VX(R{i%3}) = {vxx:.4f} + {va_values[i%3]:.4f} - {vx_values[i%3]:.4f} = {matrix[i, i]}")
        
        print("\nArrowhead Matrix:")
        print(matrix)
    
    # Plotting functionality removed as requested
    
    def save_results(self, matrix, output_dir="./", theta_index=0):
        """
        Save the matrix to files.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            The arrowhead matrix
        output_dir : str
            Directory to save the results
        theta_index : int
            Index of the theta value, used for filename
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the matrix as a numpy file
        np.save(os.path.join(output_dir, f"arrowhead_matrix_theta_{theta_index}.npy"), matrix)
        
        # Save the matrix as a CSV file
        np.savetxt(os.path.join(output_dir, f"arrowhead_matrix_theta_{theta_index}.csv"), matrix, delimiter=",")
        
        # Save the matrix details to a text file
        with open(os.path.join(output_dir, f"arrowhead_matrix_theta_{theta_index}_details.txt"), "w") as f:
            # Redirect stdout to the file
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_matrix_details(matrix)
            sys.stdout = original_stdout


def main():
    """
    Main function to generate arrowhead matrices for different theta values.
    """
    # Parameters
    R_0 = (0, 0, 0)
    d = 0.5
    theta_start = 0
    theta_end = 2 * np.pi
    theta_steps = 72  # Number of theta values to generate matrices for
    coupling_constant = 0.1
    omega = 1.0
    matrix_size = 4  # Size of each matrix
    
    # Generate theta values
    theta_values = np.linspace(theta_start, theta_end, theta_steps, endpoint=False)
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate a matrix for each theta value
    for i, theta in enumerate(theta_values):
        print(f"\nGenerating matrix for theta {i} = {theta:.4f} radians")
        
        # Create the arrowhead matrix generator for this theta
        arrowhead = ArrowheadMatrix(
            R_0=R_0,
            d=d,
            theta=theta,
            coupling_constant=coupling_constant,
            omega=omega,
            perfect=True,
            matrix_size=matrix_size
        )
        
        # Generate the matrix
        matrix = arrowhead.generate_matrix()
        
        # Print the matrix details
        arrowhead.print_matrix_details(matrix)
        
        # Save the results
        arrowhead.save_results(matrix, output_dir, i)
    
    print(f"\nGenerated {theta_steps} matrices for different theta values.")


if __name__ == "__main__":
    main()
