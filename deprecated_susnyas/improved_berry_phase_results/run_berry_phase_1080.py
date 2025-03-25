import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from improved_berry_phase import (
    hamiltonian, berry_connection_analytical, berry_phase_integration,
    calculate_numerical_berry_phase, analyze_degeneracy, analyze_parity_flips,
    generate_summary_report
)

def run_berry_phase_calculation(theta_max_degrees=360, num_points=1000, output_suffix=""):
    """
    Run the Berry phase calculation with a specified theta range.
    
    Parameters:
    theta_max_degrees (float): Maximum theta value in degrees
    num_points (int): Number of points to use in the calculation
    output_suffix (str): Suffix to add to the output directory name
    
    Returns:
    dict: Results of the calculation
    """
    # Parameters
    c = 0.2  # Fixed coupling constant for all connections
    omega = 1.0  # Frequency parameter
    
    # Potential parameters
    a = 1.0  # Coefficient for the quadratic term
    b = 0.5  # Coefficient for the linear term
    c_const = 0.0  # Constant term
    
    # Shifts for the Va potential
    x_shift = 0.2  # Shift for the Va potential on the x-axis
    y_shift = 0.2  # Shift for the Va potential on the y-axis
    
    d = 1.0  # Parameter for R_theta (distance or other parameter)
    
    # Convert max theta from degrees to radians
    theta_max_radians = theta_max_degrees * np.pi / 180
    
    # Create theta values array
    theta_vals = np.linspace(0, theta_max_radians, num_points)
    
    # Set up output directory
    output_dir = f'improved_berry_phase_results{output_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize arrays for storing eigenvalues, eigenstates, and R_theta vectors
    eigenvalues = []
    eigenstates = []
    r_theta_vectors = []
    Vx_values = []
    Va_values = []
    
    # Loop over theta values to compute the eigenvalues and eigenstates
    for theta in theta_vals:
        # Calculate the Hamiltonian, R_theta, Vx, and Va for this theta
        H, r_theta_vector, Vx, Va = hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d)
        
        r_theta_vectors.append(r_theta_vector)
        Vx_values.append(Vx)
        Va_values.append(Va)
        
        # Calculate eigenvalues and eigenstates
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Sort eigenvalues and eigenstates
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Store the results
        eigenvalues.append(eigvals)
        eigenstates.append(eigvecs)
    
    # Convert lists to numpy arrays
    eigenvalues = np.array(eigenvalues)
    eigenstates = np.array(eigenstates)
    r_theta_vectors = np.array(r_theta_vectors)
    Vx_values = np.array(Vx_values)
    Va_values = np.array(Va_values)
    
    # Calculate the analytical Berry connection
    A_analytical = berry_connection_analytical(theta_vals, c)
    
    # Calculate the analytical Berry phase by integrating the Berry connection
    berry_phases_analytical = berry_phase_integration(A_analytical, theta_vals)
    
    # Print the analytical Berry phases
    print("Analytical Berry Phases:")
    for i, phase in enumerate(berry_phases_analytical):
        print(f"Berry Phase for state {i}: {phase}")
    
    # Calculate the numerical Berry phase using the overlap method
    numerical_berry_phases = calculate_numerical_berry_phase(theta_vals, eigenstates)
    
    # Print the numerical Berry phases
    print("\nNumerical Berry Phases (Overlap Method):")
    for i, phase in enumerate(numerical_berry_phases):
        print(f"Berry Phase for state {i}: {phase}")
    
    # Compare analytical and numerical results
    print("\nComparison (Analytical - Numerical):")
    for i in range(len(berry_phases_analytical)):
        diff = berry_phases_analytical[i] - numerical_berry_phases[i]
        print(f"State {i} difference: {diff}")
    
    # Parameters dictionary for the report
    params = {
        'x_shift': x_shift,
        'y_shift': y_shift,
        'd': d,
        'omega': omega,
        'a': a,
        'b': b,
        'c': c,
        'theta_max_degrees': theta_max_degrees,
        'num_points': num_points
    }
    
    # Generate a comprehensive summary report
    report = generate_summary_report(berry_phases_analytical, numerical_berry_phases, eigenvalues, eigenstates, theta_vals, params)
    
    # Save the report to a file
    report_filename = f"{output_dir}/improved_berry_phase_summary_x{x_shift}_y{y_shift}_d{d}_w{omega}_a{a}_b{b}_theta{theta_max_degrees}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_filename}")
    
    # Plot the eigenvalue evolution
    plt.figure(figsize=(10, 6))
    for i in range(eigenvalues.shape[1]):
        plt.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')
    
    plt.xlabel('Theta (θ)')
    plt.ylabel('Energy')
    plt.title('Eigenvalue Evolution with θ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eigenvalue_evolution.png')
    plt.close()
    
    # Plot the normalized eigenvalue evolution
    plt.figure(figsize=(10, 6))
    
    # Normalize eigenvalues for better visualization
    normalized_eigenvalues = np.zeros_like(eigenvalues)
    for i in range(eigenvalues.shape[1]):
        # Normalize each eigenvalue to [0, 1] range
        min_val = np.min(eigenvalues[:, i])
        max_val = np.max(eigenvalues[:, i])
        normalized_eigenvalues[:, i] = (eigenvalues[:, i] - min_val) / (max_val - min_val)
        
        plt.plot(theta_vals * 180 / np.pi, normalized_eigenvalues[:, i], label=f'State {i}')
        # Save normalized data to file
        normalized_data = np.column_stack((theta_vals * 180 / np.pi, normalized_eigenvalues[:, i]))
        np.savetxt(f'{output_dir}/eigenstate{i}_vs_theta_normalized.txt', normalized_data, header='Theta (degrees)\tNormalized Energy', comments='')
    
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Normalized Energy')
    plt.title('Normalized Eigenvalue Evolution with θ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normalized_eigenvalue_evolution.png')
    plt.close()
    
    # Plot the analytical Berry connection
    plt.figure(figsize=(10, 6))
    for i in range(A_analytical.shape[0]):
        plt.plot(theta_vals, np.real(A_analytical[i, :]), label=f'State {i} (Analytical)')
    
    plt.xlabel('Theta (θ)')
    plt.ylabel('Berry Connection')
    plt.title('Analytical Berry Connection vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analytical_berry_connection.png')
    plt.close()
    
    # Plot the numerical Berry phase vs theta
    plt.figure(figsize=(10, 6))
    
    # Plot each state's Berry phase
    for i in range(len(numerical_berry_phases)):
        plt.plot(theta_vals * 180 / np.pi, np.ones_like(theta_vals) * numerical_berry_phases[i], 
                 label=f'State {i}: {numerical_berry_phases[i]:.4f}')
    
    # Add reference lines for important values
    plt.axhline(y=np.pi, color='r', linestyle='--', label='π')
    plt.axhline(y=-np.pi, color='r', linestyle='--', label='-π')
    plt.axhline(y=0, color='k', linestyle='--')
    
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Berry Phase')
    plt.title('Numerical Berry Phase vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/berry_phase_vs_theta.png')
    plt.close()
    
    # Calculate the scale for better zoom
    max_coord = np.max(np.abs(r_theta_vectors)) * 1.2  # 20% margin
    marker_indices = np.linspace(0, len(r_theta_vectors)-1, 100, dtype=int)
    line_length = max_coord
    line_points = np.array([[-line_length, -line_length, -line_length], [line_length, line_length, line_length]])
    
    # First create the zoomed 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path traced by R_theta
    ax.plot(r_theta_vectors[:, 0], r_theta_vectors[:, 1], r_theta_vectors[:, 2], 'b-', linewidth=2, label='R_theta path')
    
    # Add markers for a few points to show the direction of the path
    ax.scatter(r_theta_vectors[marker_indices, 0], r_theta_vectors[marker_indices, 1], r_theta_vectors[marker_indices, 2], 
               c='r', s=30, label='Markers')
    
    # Plot the origin
    ax.scatter([0], [0], [0], c='k', s=100, label='Origin')
    
    # Plot the x=y=z line
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'g-', linewidth=2, label='x=y=z line')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('R_theta Vector in 3D Space (Zoomed)')
    
    # Set equal aspect ratio and zoom in
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_zlim(-max_coord, max_coord)
    
    # Add a legend
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r_theta_3d.png', dpi=300)
    plt.close()
    
    # Now create the 2x2 subplot with projections
    fig = plt.figure(figsize=(16, 14))
    
    # XY Projection (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(r_theta_vectors[:, 0], r_theta_vectors[:, 1], 'b-', linewidth=2, label='R_theta path')
    ax1.scatter(r_theta_vectors[marker_indices, 0], r_theta_vectors[marker_indices, 1], c='r', s=30, label='Markers')
    ax1.scatter(0, 0, c='k', s=100, label='Origin')
    ax1.plot(line_points[:, 0], line_points[:, 1], 'g-', linewidth=2, label='x=y=z line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY Projection')
    ax1.set_xlim(-max_coord, max_coord)
    ax1.set_ylim(-max_coord, max_coord)
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Create a projection onto the plane perpendicular to x=y=z (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Define basis vectors for the plane perpendicular to x=y=z
    # The x=y=z direction is (1,1,1)/sqrt(3)
    # We need two orthogonal vectors to this direction
    basis_xyz = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized x=y=z direction
    
    # Create two orthogonal vectors to basis_xyz
    # First orthogonal vector: (1,-1,0)/sqrt(2)
    basis1 = np.array([1, -1, 0]) / np.sqrt(2)
    
    # Second orthogonal vector: cross product of basis_xyz and basis1
    basis2 = np.cross(basis_xyz, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)  # Normalize
    
    # Project the R_theta vectors onto the plane perpendicular to x=y=z
    projected_points = np.zeros((len(r_theta_vectors), 2))
    for i, vec in enumerate(r_theta_vectors):
        # Project onto the two basis vectors
        projected_points[i, 0] = np.dot(vec, basis1)
        projected_points[i, 1] = np.dot(vec, basis2)
    
    # Plot the projected circle
    ax2.plot(projected_points[:, 0], projected_points[:, 1], 'b-', linewidth=2)
    ax2.scatter(projected_points[marker_indices, 0], projected_points[marker_indices, 1], c='r', s=30)
    ax2.scatter(0, 0, c='k', s=100)
    
    # The x=y=z line projects to a point at the origin in this view
    ax2.plot(0, 0, 'go', markersize=10)
    
    # Set labels and title
    ax2.set_xlabel('Basis Vector 1')
    ax2.set_ylabel('Basis Vector 2')
    ax2.set_title('Projection onto Plane ⊥ to x=y=z Line')
    
    # Set equal aspect ratio and limits
    max_proj = np.max(np.abs(projected_points)) * 1.2
    ax2.set_xlim(-max_proj, max_proj)
    ax2.set_ylim(-max_proj, max_proj)
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # XZ Projection (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(r_theta_vectors[:, 0], r_theta_vectors[:, 2], 'b-', linewidth=2)
    ax3.scatter(r_theta_vectors[marker_indices, 0], r_theta_vectors[marker_indices, 2], c='r', s=30)
    ax3.scatter(0, 0, c='k', s=100)
    ax3.plot(line_points[:, 0], line_points[:, 2], 'g-', linewidth=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_xlim(-max_coord, max_coord)
    ax3.set_ylim(-max_coord, max_coord)
    ax3.grid(True)
    ax3.set_aspect('equal')
    
    # YZ Projection (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(r_theta_vectors[:, 1], r_theta_vectors[:, 2], 'b-', linewidth=2)
    ax4.scatter(r_theta_vectors[marker_indices, 1], r_theta_vectors[marker_indices, 2], c='r', s=30)
    ax4.scatter(0, 0, c='k', s=100)
    ax4.plot(line_points[:, 1], line_points[:, 2], 'g-', linewidth=2)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.set_xlim(-max_coord, max_coord)
    ax4.set_ylim(-max_coord, max_coord)
    ax4.grid(True)
    ax4.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/r_theta_3d_with_projections.png', dpi=300)
    plt.close()
    
    # Plot the potential components
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Vx Components')
    plt.title('Vx Components vs Theta')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Va Components')
    plt.title('Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/potential_components.png')
    plt.close()
    
    # Create a summary file with links to all generated files
    summary_file = f"{output_dir}/summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Berry Phase Analysis Summary\n")
        f.write("=========================\n\n")
        f.write(f"Detailed Report: {os.path.basename(report_filename)}\n\n")
        f.write("Generated Plots:\n")
        f.write(f"  - Eigenvalue Evolution: eigenvalue_evolution.png\n")
        f.write(f"  - Normalized Eigenvalue Evolution: normalized_eigenvalue_evolution.png\n")
        f.write(f"  - Analytical Berry Connection: analytical_berry_connection.png\n")
        f.write(f"  - Numerical Berry Phase vs Theta: berry_phase_vs_theta.png\n")
        f.write(f"  - R_theta 3D Visualization (Zoomed): r_theta_3d.png\n")
        f.write(f"  - R_theta Projections (XY, x=y=z plane, XZ, YZ): r_theta_3d_with_projections.png\n")
        f.write(f"  - Potential Components: potential_components.png\n\n")
        f.write("Normalized Data Files:\n")
        for i in range(eigenvalues.shape[1]):
            f.write(f"  - State {i}: eigenstate{i}_vs_theta_normalized.txt\n")
    
    print(f"Summary file created: {summary_file}")
    
    # Return results dictionary
    return {
        'theta_vals': theta_vals,
        'eigenvalues': eigenvalues,
        'eigenstates': eigenstates,
        'r_theta_vectors': r_theta_vectors,
        'berry_phases_analytical': berry_phases_analytical,
        'numerical_berry_phases': numerical_berry_phases,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    # Run the calculation with theta values from 0 to 1080 degrees
    results_1080 = run_berry_phase_calculation(
        theta_max_degrees=1080,
        num_points=2000,  # Double the points for smoother curves
        output_suffix="_1080"
    )
    
    print("\nCalculation completed for theta values from 0 to 1080 degrees.")
    print(f"Results saved to: {results_1080['output_dir']}")
