import numpy as np
import os
import sys
import datetime
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from new_bph import Hamiltonian
from scripts.find_adaptive_d_values import analyze_specific_d_value
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_vectors


def generate_triple_points(base_point, d_value):
    """
    Generate three points on a perfect orthogonal circle to the (1,1,1) line,
    spaced 120 degrees apart, with the circle centered at the origin.
    
    Parameters:
    - base_point: The first point (numpy array [x, y, z])
    - d_value: The d value to use for the circle radius
    
    Returns:
    - List of three R vectors
    """
    # Ensure base_point is a numpy array
    base_point = np.array(base_point)
    
    # Define the (1,1,1) direction
    direction_111 = np.array([1, 1, 1]) / np.sqrt(3)
    
    # Find two orthogonal vectors to form a basis in the plane perpendicular to (1,1,1)
    # First vector: cross product of (1,1,1) with any non-parallel vector, e.g., (1,0,0)
    v1 = np.cross(direction_111, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)  # Normalize
    
    # Second vector: cross product of (1,1,1) with v1
    v2 = np.cross(direction_111, v1)
    v2 = v2 / np.linalg.norm(v2)  # Normalize
    
    # Project base_point onto the plane perpendicular to (1,1,1)
    # Remove the component along (1,1,1)
    base_proj = base_point - np.dot(base_point, direction_111) * direction_111
    
    # Calculate the distance from the origin in the projected plane
    radius = np.linalg.norm(base_proj)
    
    # Calculate the angle of the base point in the projected plane
    base_angle = np.arctan2(np.dot(base_proj, v2), np.dot(base_proj, v1))
    
    # Generate three points 120 degrees apart on the circle
    points = []
    for angle_offset in [0, 2*np.pi/3, 4*np.pi/3]:
        # Compute the angle for this point
        angle = base_angle + angle_offset
        
        # Compute point on the circle in the v1-v2 plane
        point = radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
        
        # The point is already in 3D space, centered at the origin and on the plane perpendicular to (1,1,1)
        points.append(point)
    
    return points


def find_va_vx_intersection(d_start, d_end, step, aVx, aVa, x_shift, c_const, omega, R_0, epsilon=1e-10):
    """
    Find points where Va-Vx is numerically equal within a small range of d values.
    Returns a list of (d, theta) tuples where the intersection occurs.
    """
    points = []
    d_values = np.arange(d_start, d_end, step)
    
    # Create theta ranges for 0 and 2π
    theta_0 = np.array([0])
    theta_2pi = np.array([2*np.pi])
    
    for d in tqdm(d_values, desc=f"Searching for Va-Vx intersections around R_0={R_0}"):
        # Create Hamiltonian for theta=0 and theta=2π (should be the same point)
        H0 = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_0)
        H2pi = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_2pi)
        
        # Get R vectors for both thetas
        R0 = H0.R_theta(0)
        R2pi = H2pi.R_theta(2*np.pi)
        
        # Calculate Va and Vx for both R vectors
        Va_0 = H0.V_a(R0)
        Vx_0 = H0.V_x(R0)
        Va_2pi = H2pi.V_a(R2pi)
        Vx_2pi = H2pi.V_x(R2pi)
        
        # Calculate Va - Vx for both thetas
        Va_minus_Vx_0 = Va_0 - Vx_0
        Va_minus_Vx_2pi = Va_2pi - Vx_2pi
        
        # Check if the difference is within epsilon (for all elements if arrays)
        if np.all(np.abs(Va_minus_Vx_0 - Va_minus_Vx_2pi) < epsilon):
            points.append((d, 0, R_0))  # Store d, theta, and R_0
            print(f"Found intersection at d={d:.15f}, Va-Vx={Va_minus_Vx_0}, R_0={R_0}")
    
    return points


def find_triple_ci_points(base_R0=np.array([0.05, -0.025, -0.025]), 
                          d_value=0.06123724356957945,
                          d_start=0.061, d_end=0.062, d_step=1e-5,
                          aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0,
                          epsilon=1e-10):
    """
    Generate three points on a perfect orthogonal circle and search for CIs around them.
    
    Parameters:
    - base_R0: The base R0 point
    - d_value: The d value for generating the circle
    - d_start, d_end, d_step: Range for searching d values
    - aVx, aVa, x_shift, c_const, omega: Hamiltonian parameters
    - epsilon: Tolerance for numerical equality
    """
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'triple_ci_points_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Generate the three points
    triple_points = generate_triple_points(base_R0, d_value)
    
    # Save the triple points
    with open(os.path.join(parent_dir, 'triple_points.txt'), 'w') as f:
        f.write("# Triple points on orthogonal circle\n")
        for i, point in enumerate(triple_points):
            f.write(f"Point {i+1}: {point}\n")
    
    # Visualize the triple points in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    for i, point in enumerate(triple_points):
        ax.scatter(point[0], point[1], point[2], s=100, label=f'Point {i+1}')
    
    # Plot the (1,1,1) line
    line_points = np.array([[-1, -1, -1], [1, 1, 1]]) * 0.1
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'r--', label='(1,1,1) Line')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triple Points on Orthogonal Circle')
    ax.legend()
    
    # Save the figure
    plt.savefig(os.path.join(parent_dir, 'triple_points_3d.png'), dpi=150)
    plt.close()
    
    # Project the points onto a 2D plane orthogonal to (1,1,1)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define the (1,1,1) direction
    direction_111 = np.array([1, 1, 1]) / np.sqrt(3)
    
    # Find two orthogonal vectors to form a basis in the plane perpendicular to (1,1,1)
    # These vectors will be centered at the origin (0,0,0)
    v1 = np.cross(direction_111, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(direction_111, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Project and plot each point
    for i, point in enumerate(triple_points):
        # Project point onto the plane perpendicular to (1,1,1)
        # First, remove the component along (1,1,1)
        proj_point = point - np.dot(point, direction_111) * direction_111
        
        # Get coordinates in the v1-v2 basis
        x_proj = np.dot(proj_point, v1)
        y_proj = np.dot(proj_point, v2)
        
        ax.scatter(x_proj, y_proj, s=100, label=f'Point {i+1}')
    
    # Add origin (0,0,0) projected onto the plane
    # The origin projects to (0,0) in the v1-v2 basis
    ax.scatter(0, 0, color='red', s=50, label='Origin')
    
    # Add circle centered at the origin
    theta = np.linspace(0, 2*np.pi, 100)
    # Calculate radius as distance from origin to any of the projected points
    radius = np.linalg.norm(triple_points[0] - np.dot(triple_points[0], direction_111) * direction_111)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Circle')
    
    # Set labels and title
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    ax.set_title('Projection onto Plane Orthogonal to (1,1,1)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(parent_dir, 'triple_points_2d_projection.png'), dpi=150)
    plt.close()
    
    # Search for CI points around each of the triple points
    all_ci_points = []
    
    for i, R0 in enumerate(triple_points):
        print(f"\nSearching for CIs around Point {i+1}: {R0}")
        ci_points = find_va_vx_intersection(
            d_start, d_end, d_step, 
            aVx, aVa, x_shift, c_const, omega, R0, epsilon
        )
        all_ci_points.extend(ci_points)
    
    # Save all CI points
    with open(os.path.join(parent_dir, 'all_ci_points.txt'), 'w') as f:
        f.write("# d\ttheta\tR0_x\tR0_y\tR0_z\n")
        for d, theta, R0 in all_ci_points:
            f.write(f"{d:.15f}\t{theta}\t{R0[0]}\t{R0[1]}\t{R0[2]}\n")
    
    # Visualize CI points in 3D
    if all_ci_points:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract R0 values from all CI points
        R0_values = np.array([point[2] for point in all_ci_points])
        
        # Plot the CI points
        ax.scatter(R0_values[:, 0], R0_values[:, 1], R0_values[:, 2], s=100, c='r', label='CI Points')
        
        # Plot the triple points
        for i, point in enumerate(triple_points):
            ax.scatter(point[0], point[1], point[2], s=150, marker='*', label=f'Base Point {i+1}')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('CI Points and Base Points')
        ax.legend()
        
        # Save the figure
        plt.savefig(os.path.join(parent_dir, 'ci_points_3d.png'), dpi=150)
        plt.close()
        
        # Project and plot CI points in 2D
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Project and plot each CI point
        for d, theta, R0 in all_ci_points:
            # Project point onto the plane perpendicular to (1,1,1)
            proj_point = R0 - np.dot(R0, direction_111) * direction_111
            
            # Get coordinates in the v1-v2 basis
            x_proj = np.dot(proj_point, v1)
            y_proj = np.dot(proj_point, v2)
            
            ax.scatter(x_proj, y_proj, s=80, c='r')
        
        # Project and plot each base point
        for i, point in enumerate(triple_points):
            # Project point onto the plane perpendicular to (1,1,1)
            proj_point = point - np.dot(point, direction_111) * direction_111
            
            # Get coordinates in the v1-v2 basis
            x_proj = np.dot(proj_point, v1)
            y_proj = np.dot(proj_point, v2)
            
            ax.scatter(x_proj, y_proj, s=100, marker='*', label=f'Base Point {i+1}')
        
        # Add origin (0,0,0) projected onto the plane
        # The origin projects to (0,0) in the v1-v2 basis
        ax.scatter(0, 0, color='black', s=50, label='Origin')
        
        # Add circle centered at the origin
        theta = np.linspace(0, 2*np.pi, 100)
        radius = np.linalg.norm(triple_points[0] - np.dot(triple_points[0], direction_111) * direction_111)
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        ax.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Circle')
        
        # Set labels and title
        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_title('CI Points Projection onto Plane Orthogonal to (1,1,1)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(parent_dir, 'ci_points_2d_projection.png'), dpi=150)
        plt.close()
    
    return all_ci_points, triple_points


def plot_potentials(triple_points, d_value, aVx, aVa, x_shift, c_const, omega, output_dir, point_type='original'):
    """
    Plot Va, Vx, Va-Vx, and Vx-Va for each triple point.
    
    Parameters:
    - triple_points: List of three base points
    - d_value: The d value to use
    - aVx, aVa, x_shift, c_const, omega: Hamiltonian parameters
    - output_dir: Directory to save the plots
    - point_type: Type of points ('original' or 'nested')
    """
    # Create a subdirectory for potential plots
    potentials_dir = os.path.join(output_dir, 'potential_plots')
    os.makedirs(potentials_dir, exist_ok=True)
    
    # Create a subdirectory for this point type
    type_dir = os.path.join(potentials_dir, point_type)
    os.makedirs(type_dir, exist_ok=True)
    
    # Create theta range for plotting
    theta_range = np.linspace(0, 2*np.pi, 100)
    
    # For each triple point
    for i, R0 in enumerate(triple_points):
        # Create a Hamiltonian with the given parameters
        H = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R0, d_value, theta_range)
        
        # Get R vectors for each theta
        R_vectors = [H.R_theta(theta) for theta in theta_range]
        
        # Calculate Va and Vx for each R vector
        Va_values = np.array([H.V_a(R) for R in R_vectors])
        Vx_values = np.array([H.V_x(R) for R in R_vectors])
        
        # Create figure for Va and Vx
        plt.figure(figsize=(12, 8))
        
        # Plot Va for each component
        for j in range(Va_values.shape[1]):
            plt.plot(theta_range/np.pi, Va_values[:, j], label=f'Va[{j+1}]')
        
        # Plot Vx for each component
        for j in range(Vx_values.shape[1]):
            plt.plot(theta_range/np.pi, Vx_values[:, j], '--', label=f'Vx[{j+1}]')
        
        plt.title(f'Va and Vx vs Theta for R0={R0}, d={d_value:.10f}')
        plt.xlabel('Theta (θ/π)')
        plt.ylabel('Potential Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(type_dir, f'Va_Vx_R0_{i+1}.png'), dpi=150)
        plt.close()
        
        # Create figure for Va-Vx
        plt.figure(figsize=(12, 8))
        
        # Plot Va-Vx for each component
        for j in range(Va_values.shape[1]):
            plt.plot(theta_range/np.pi, Va_values[:, j] - Vx_values[:, j], label=f'Va[{j+1}] - Vx[{j+1}]')
        
        plt.title(f'Va-Vx vs Theta for R0={R0}, d={d_value:.10f}')
        plt.xlabel('Theta (θ/π)')
        plt.ylabel('Va-Vx Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(type_dir, f'Va_minus_Vx_R0_{i+1}.png'), dpi=150)
        plt.close()
        
        # For nested points, add a zoomed-in version to better see the crossing
        if point_type == 'nested':
            # Create zoomed-in figure for Va-Vx
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Find where Va-Vx is close to zero for all components
            diff_values = Va_values - Vx_values
            abs_diff = np.abs(diff_values)
            min_indices = []
            
            # Find the minimum difference points for each component
            for j in range(diff_values.shape[1]):
                min_idx = np.argmin(abs_diff[:, j])
                min_indices.append(min_idx)
            
            # Get the average index to center the zoom
            center_idx = int(np.mean(min_indices))
            window = 10  # Points on either side to show
            start_idx = max(0, center_idx - window)
            end_idx = min(len(theta_range), center_idx + window)
            
            # Plot the zoomed-in region for each component
            for j in range(Va_values.shape[1]):
                ax.plot(theta_range[start_idx:end_idx]/np.pi, 
                       diff_values[start_idx:end_idx, j], 
                       linewidth=2, label=f'Va[{j+1}] - Vx[{j+1}]')
            
            ax.axhline(y=0, color='r', linestyle='--', label='Zero')
            ax.set_title(f'ZOOMED: Va-Vx vs Theta for R0={R0}, d={d_value:.10f}')
            ax.set_xlabel('Theta (θ/π)')
            ax.set_ylabel('Va-Vx Value')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(type_dir, f'Va_minus_Vx_ZOOM_R0_{i+1}.png'), dpi=150)
            plt.close()
        
        # Create figure for Vx-Va
        plt.figure(figsize=(12, 8))
        
        # Plot Vx-Va for each component
        for j in range(Vx_values.shape[1]):
            plt.plot(theta_range/np.pi, Vx_values[:, j] - Va_values[:, j], label=f'Vx[{j+1}] - Va[{j+1}]')
        
        plt.title(f'Vx-Va vs Theta for R0={R0}, d={d_value:.10f}')
        plt.xlabel('Theta (θ/π)')
        plt.ylabel('Vx-Va Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(type_dir, f'Vx_minus_Va_R0_{i+1}.png'), dpi=150)
        plt.close()
        
        # For nested points, add a zoomed-in version to better see the crossing
        if point_type == 'nested':
            # Create zoomed-in figure for Vx-Va
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Find where Vx-Va is close to zero for all components
            diff_values = Vx_values - Va_values
            abs_diff = np.abs(diff_values)
            min_indices = []
            
            # Find the minimum difference points for each component
            for j in range(diff_values.shape[1]):
                min_idx = np.argmin(abs_diff[:, j])
                min_indices.append(min_idx)
            
            # Get the average index to center the zoom
            center_idx = int(np.mean(min_indices))
            window = 10  # Points on either side to show
            start_idx = max(0, center_idx - window)
            end_idx = min(len(theta_range), center_idx + window)
            
            # Plot the zoomed-in region for each component
            for j in range(Vx_values.shape[1]):
                ax.plot(theta_range[start_idx:end_idx]/np.pi, 
                       diff_values[start_idx:end_idx, j], 
                       linewidth=2, label=f'Vx[{j+1}] - Va[{j+1}]')
            
            ax.axhline(y=0, color='r', linestyle='--', label='Zero')
            ax.set_title(f'ZOOMED: Vx-Va vs Theta for R0={R0}, d={d_value:.10f}')
            ax.set_xlabel('Theta (θ/π)')
            ax.set_ylabel('Vx-Va Value')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(type_dir, f'Vx_minus_Va_ZOOM_R0_{i+1}.png'), dpi=150)
            plt.close()
        
        # Create a combined figure showing all potentials
        plt.figure(figsize=(15, 10))
        
        # Format R0 coordinates as a string with 3 decimal places
        R0_str = f"({R0[0]:.3f}, {R0[1]:.3f}, {R0[2]:.3f})"
        
        plt.subplot(2, 2, 1)
        for j in range(Va_values.shape[1]):
            plt.plot(theta_range/np.pi, Va_values[:, j], label=f'Va[{j+1}]')
        plt.title(f'Va vs Theta for R0={R0_str}')
        plt.xlabel('Theta (\u03b8/\u03c0)')
        plt.ylabel('Va Value')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for j in range(Vx_values.shape[1]):
            plt.plot(theta_range/np.pi, Vx_values[:, j], label=f'Vx[{j+1}]')
        plt.title(f'Vx vs Theta for R0={R0_str}')
        plt.xlabel('Theta (\u03b8/\u03c0)')
        plt.ylabel('Vx Value')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for j in range(Va_values.shape[1]):
            plt.plot(theta_range/np.pi, Va_values[:, j] - Vx_values[:, j], label=f'Va[{j+1}] - Vx[{j+1}]')
        plt.title(f'Va-Vx vs Theta for R0={R0_str}')
        plt.xlabel('Theta (\u03b8/\u03c0)')
        plt.ylabel('Va-Vx Value')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for j in range(Vx_values.shape[1]):
            plt.plot(theta_range/np.pi, Vx_values[:, j] - Va_values[:, j], label=f'Vx[{j+1}] - Va[{j+1}]')
        plt.title(f'Vx-Va vs Theta for R0={R0_str}')
        plt.xlabel('Theta (\u03b8/\u03c0)')
        plt.ylabel('Vx-Va Value')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Potential Analysis for R0={R0}, d={d_value:.10f}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(type_dir, f'all_potentials_R0_{i+1}.png'), dpi=150)
        plt.close()


def plot_ci_points_detailed(triple_points, all_points, all_ci_points, output_dir):
    """
    Create detailed visualizations of CI points around each triple point.
    
    Parameters:
    - triple_points: List of three base points
    - all_ci_points: List of (d, theta, R0) tuples for all found CI points
    - output_dir: Directory to save the plots
    """
    # Define the (1,1,1) direction
    direction_111 = np.array([1, 1, 1]) / np.sqrt(3)
    
    # Find two orthogonal vectors to form a basis in the plane perpendicular to (1,1,1)
    v1 = np.cross(direction_111, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)  # Normalize
    v2 = np.cross(direction_111, v1)
    v2 = v2 / np.linalg.norm(v2)  # Normalize
    
    # Group CI points by their base point
    ci_points_by_base = {}
    for d, theta, R0 in all_ci_points:
        R0_tuple = tuple(R0)  # Convert to tuple for dictionary key
        if R0_tuple not in ci_points_by_base:
            ci_points_by_base[R0_tuple] = []
        ci_points_by_base[R0_tuple].append((d, theta))
    
    # Create individual plots for each triple point
    for i, base_point in enumerate(triple_points):
        base_tuple = tuple(base_point)
        if base_tuple in ci_points_by_base:
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the base point
            ax.scatter(base_point[0], base_point[1], base_point[2], 
                      s=150, marker='*', color='blue', label=f'Base Point {i+1}')
            
            # Plot the CI points around this base point
            for d, theta in ci_points_by_base[base_tuple]:
                # Create a small sphere of points around the base point
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                radius = 0.005  # Small radius for visualization
                x = base_point[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = base_point[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = base_point[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='red', alpha=0.3)
            
            # Plot the (1,1,1) line
            line_points = np.array([[-1, -1, -1], [1, 1, 1]]) * 0.1
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'r--', label='(1,1,1) Line')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'CI Points around Base Point {i+1}: {base_point}')
            ax.legend()
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f'ci_points_base_{i+1}_3d.png'), dpi=150)
            plt.close()
            
            # Create 2D projection plot
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Project and plot the base point
            base_proj = base_point - np.dot(base_point, direction_111) * direction_111
            x_proj = np.dot(base_proj, v1)
            y_proj = np.dot(base_proj, v2)
            ax.scatter(x_proj, y_proj, s=150, marker='*', color='blue', label=f'Base Point {i+1}')
            
            # Plot a circle around the base point to represent CI points
            theta = np.linspace(0, 2*np.pi, 100)
            small_radius = 0.01  # Small radius for visualization
            x_circle = x_proj + small_radius * np.cos(theta)
            y_circle = y_proj + small_radius * np.sin(theta)
            ax.plot(x_circle, y_circle, 'r-', alpha=0.7, label=f'CI Points (d={d:.10f})')
            
            # Add origin
            ax.scatter(0, 0, color='black', s=50, label='Origin')
            
            # Add the main circle
            radius = np.linalg.norm(base_proj)
            x_main_circle = radius * np.cos(theta)
            y_main_circle = radius * np.sin(theta)
            ax.plot(x_main_circle, y_main_circle, 'k--', alpha=0.5, label='Main Circle')
            
            # Set labels and title
            ax.set_xlabel('v1')
            ax.set_ylabel('v2')
            ax.set_title(f'CI Points around Base Point {i+1} (2D Projection)')
            ax.legend()
            ax.axis('equal')
            ax.grid(True)
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f'ci_points_base_{i+1}_2d.png'), dpi=150)
            plt.close()
    
    # Create a combined 2D projection plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all original triple points
    for i, base_point in enumerate(triple_points):
        base_proj = base_point - np.dot(base_point, direction_111) * direction_111
        x_proj = np.dot(base_proj, v1)
        y_proj = np.dot(base_proj, v2)
        ax.scatter(x_proj, y_proj, s=150, marker='*', color=f'C{i}', label=f'Original Point {i+1}')
        
        # Plot circles around base points with CI points
        if tuple(base_point) in ci_points_by_base:
            theta = np.linspace(0, 2*np.pi, 100)
            small_radius = 0.01  # Small radius for visualization
            x_circle = x_proj + small_radius * np.cos(theta)
            y_circle = y_proj + small_radius * np.sin(theta)
            ax.plot(x_circle, y_circle, color=f'C{i}', alpha=0.7)
    
    # Plot all nested triple points
    nested_points = all_points[len(triple_points):]
    for i, nested_point in enumerate(nested_points):
        # Project nested point onto the plane
        nested_proj = nested_point - np.dot(nested_point, direction_111) * direction_111
        x_proj = np.dot(nested_proj, v1)
        y_proj = np.dot(nested_proj, v2)
        
        # Determine which original point this nested point belongs to
        parent_idx = i // 3
        ax.scatter(x_proj, y_proj, s=80, marker='o', color=f'C{parent_idx}', alpha=0.7)
        
        # Plot small circles around nested points with CI points
        if tuple(nested_point) in ci_points_by_base:
            theta = np.linspace(0, 2*np.pi, 100)
            tiny_radius = 0.005  # Smaller radius for nested points
            x_circle = x_proj + tiny_radius * np.cos(theta)
            y_circle = y_proj + tiny_radius * np.sin(theta)
            ax.plot(x_circle, y_circle, color=f'C{parent_idx}', alpha=0.5, linestyle=':')
    
    # Add origin
    ax.scatter(0, 0, color='black', s=50, label='Origin')
    
    # Add the main circle
    theta = np.linspace(0, 2*np.pi, 100)
    radius = np.linalg.norm(triple_points[0] - np.dot(triple_points[0], direction_111) * direction_111)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Main Circle')
    
    # Set labels and title
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    ax.set_title('All Triple Points and CI Points (2D Projection)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'all_points_2d_projection.png'), dpi=150)
    plt.close()


def generate_detailed_report(points, d_value, aVx, aVa, x_shift, c_const, omega, output_dir, point_type):
    """
    Generate a detailed report with high numerical precision for Va-Vx values at each point.
    
    Parameters:
    - points: List of points to analyze
    - d_value: The d value to use
    - aVx, aVa, x_shift, c_const, omega: Hamiltonian parameters
    - output_dir: Directory to save the report
    - point_type: Type of points ('original' or 'nested')
    """
    # Create a subdirectory for reports
    reports_dir = os.path.join(output_dir, 'detailed_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create a report file for this point type
    report_file = os.path.join(reports_dir, f'{point_type}_points_detailed_report.txt')
    
    with open(report_file, 'w') as f:
        f.write(f"Detailed Report for {point_type.capitalize()} Points\n")
        f.write("="*50 + "\n\n")
        f.write(f"Global Parameters:\n")
        f.write(f"  d_value = {d_value:.16f}\n")
        f.write(f"  aVx = {aVx}\n")
        f.write(f"  aVa = {aVa}\n")
        f.write(f"  x_shift = {x_shift}\n")
        f.write(f"  c_const = {c_const}\n")
        f.write(f"  omega = {omega}\n\n")
        
        # Create theta range with high precision
        theta_range = np.linspace(0, 2*np.pi, 1000)  # Higher resolution for precision
        
        for i, R0 in enumerate(points):
            f.write(f"Point {i+1}: R0 = {R0}\n")
            f.write("-"*50 + "\n")
            
            # Calculate potentials with high precision
            Va_values = np.zeros((len(theta_range), 3))
            Vx_values = np.zeros((len(theta_range), 3))
            
            # Find the minimum difference between Va and Vx for each component
            min_diff_values = [float('inf'), float('inf'), float('inf')]
            min_diff_thetas = [0, 0, 0]
            min_diff_Va = [0, 0, 0]
            min_diff_Vx = [0, 0, 0]
            
            for j, theta in enumerate(theta_range):
                # Calculate R(theta)
                R_theta = np.array([
                    R0[0] + d_value * np.cos(theta),
                    R0[1] + d_value * np.sin(theta),
                    R0[2]
                ])
                
                # Calculate Va and Vx for each component
                for k in range(3):
                    # Va calculation
                    Va_values[j, k] = aVa * (R_theta[k] ** 2)
                    
                    # Vx calculation
                    if k == 0:
                        Vx_values[j, k] = aVx * ((R_theta[1] - x_shift) ** 2 + R_theta[2] ** 2) + c_const
                    elif k == 1:
                        Vx_values[j, k] = aVx * ((R_theta[0] - x_shift) ** 2 + R_theta[2] ** 2) + c_const
                    else:  # k == 2
                        Vx_values[j, k] = aVx * ((R_theta[0] - x_shift) ** 2 + R_theta[1] ** 2) + c_const
                    
                    # Check if this is the minimum difference
                    diff = abs(Va_values[j, k] - Vx_values[j, k])
                    if diff < min_diff_values[k]:
                        min_diff_values[k] = diff
                        min_diff_thetas[k] = theta
                        min_diff_Va[k] = Va_values[j, k]
                        min_diff_Vx[k] = Vx_values[j, k]
            
            # Write the minimum differences to the report
            f.write("Minimum Va-Vx differences for each component:\n")
            for k in range(3):
                f.write(f"  Component {k+1}:\n")
                f.write(f"    Theta = {min_diff_thetas[k]:.16f} rad = {min_diff_thetas[k]/np.pi:.16f}π\n")
                f.write(f"    Va = {min_diff_Va[k]:.16f}\n")
                f.write(f"    Vx = {min_diff_Vx[k]:.16f}\n")
                f.write(f"    |Va-Vx| = {min_diff_values[k]:.16f}\n")
                f.write(f"    Va-Vx = {min_diff_Va[k]-min_diff_Vx[k]:.16f}\n\n")
            
            # Find where all three components have minimum differences within a small range
            # This indicates a potential CI point
            theta_diffs = [abs(min_diff_thetas[0] - min_diff_thetas[1]),
                          abs(min_diff_thetas[1] - min_diff_thetas[2]),
                          abs(min_diff_thetas[0] - min_diff_thetas[2])]
            
            max_theta_diff = max(theta_diffs)
            f.write(f"Maximum theta difference between components: {max_theta_diff:.16f} rad = {max_theta_diff/np.pi:.16f}π\n")
            
            if max_theta_diff < 0.01:  # If all thetas are close
                f.write("POTENTIAL CI POINT DETECTED: All components have minimum Va-Vx at similar theta values\n")
                
                # Calculate the average theta and the potentials at that point
                avg_theta = sum(min_diff_thetas) / 3
                f.write(f"Average theta for CI point: {avg_theta:.16f} rad = {avg_theta/np.pi:.16f}π\n")
                
                # Calculate R(avg_theta)
                R_ci = np.array([
                    R0[0] + d_value * np.cos(avg_theta),
                    R0[1] + d_value * np.sin(avg_theta),
                    R0[2]
                ])
                f.write(f"CI point coordinates: {R_ci}\n")
                
                # Calculate potentials at the CI point
                Va_ci = np.zeros(3)
                Vx_ci = np.zeros(3)
                for k in range(3):
                    # Va calculation
                    Va_ci[k] = aVa * (R_ci[k] ** 2)
                    
                    # Vx calculation
                    if k == 0:
                        Vx_ci[k] = aVx * ((R_ci[1] - x_shift) ** 2 + R_ci[2] ** 2) + c_const
                    elif k == 1:
                        Vx_ci[k] = aVx * ((R_ci[0] - x_shift) ** 2 + R_ci[2] ** 2) + c_const
                    else:  # k == 2
                        Vx_ci[k] = aVx * ((R_ci[0] - x_shift) ** 2 + R_ci[1] ** 2) + c_const
                
                f.write("Potentials at CI point:\n")
                for k in range(3):
                    f.write(f"  Component {k+1}:\n")
                    f.write(f"    Va = {Va_ci[k]:.16f}\n")
                    f.write(f"    Vx = {Vx_ci[k]:.16f}\n")
                    f.write(f"    Va-Vx = {Va_ci[k]-Vx_ci[k]:.16f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Detailed report for {point_type} points saved to: {report_file}")


def generate_nested_triple_points(triple_points, small_radius=0.01):
    """
    Generate additional triple points around each of the original triple points.
    
    Parameters:
    - triple_points: List of original triple points
    - small_radius: Radius of the small circles around each triple point
    
    Returns:
    - List of all triple points (original and nested)
    """
    # Define the (1,1,1) direction
    direction_111 = np.array([1, 1, 1]) / np.sqrt(3)
    
    # Find two orthogonal vectors to form a basis in the plane perpendicular to (1,1,1)
    v1 = np.cross(direction_111, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)  # Normalize
    v2 = np.cross(direction_111, v1)
    v2 = v2 / np.linalg.norm(v2)  # Normalize
    
    all_points = []
    # Add the original triple points
    all_points.extend(triple_points)
    
    # For each original triple point, generate three additional points around it
    for base_point in triple_points:
        # Project base_point onto the plane perpendicular to (1,1,1)
        base_proj = base_point - np.dot(base_point, direction_111) * direction_111
        
        # Calculate the angle of the base point in the projected plane
        base_angle = np.arctan2(np.dot(base_proj, v2), np.dot(base_proj, v1))
        
        # Generate three points 120 degrees apart on a small circle around the base point
        for angle_offset in [0, 2*np.pi/3, 4*np.pi/3]:
            angle = base_angle + angle_offset
            
            # Calculate the point on the small circle around the base point
            small_circle_point = base_proj + small_radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            
            # Add back the component along (1,1,1) that the base point had
            point_3d = small_circle_point + np.dot(base_point, direction_111) * direction_111
            
            all_points.append(point_3d)
    
    return all_points


if __name__ == '__main__':
    # Use the known CI point as the base
    base_R0 = np.array([0.05, -0.025, -0.025])
    d_value = 0.06123724356957945
    
    # Search for CIs in a wider range to find more points
    d_start = 0.06123724356957945 - 1e-6
    d_end = 0.06123724356957945 + 1e-6
    d_step = 5e-7
    
    # Fix the generate_triple_points function to properly create R vectors
    def generate_triple_points_fixed(base_point, d_value):
        # Ensure base_point is a numpy array
        base_point = np.array(base_point)
        
        # Define the (1,1,1) direction
        direction_111 = np.array([1, 1, 1]) / np.sqrt(3)
        
        # Find two orthogonal vectors to form a basis in the plane perpendicular to (1,1,1)
        v1 = np.cross(direction_111, np.array([1, 0, 0]))
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        v2 = np.cross(direction_111, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        
        # Project base_point onto the plane perpendicular to (1,1,1)
        base_proj = base_point - np.dot(base_point, direction_111) * direction_111
        
        # Calculate the distance from the origin in the projected plane
        radius = np.linalg.norm(base_proj)
        
        # Calculate the angle of the base point in the projected plane
        base_angle = np.arctan2(np.dot(base_proj, v2), np.dot(base_proj, v1))
        
        # Generate three points 120 degrees apart on the circle
        points = []
        for angle_offset in [0, 2*np.pi/3, 4*np.pi/3]:
            angle = base_angle + angle_offset
            point = radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            points.append(point)
        
        return points
    
    # Generate the three points on the circle
    triple_points = generate_triple_points_fixed(base_R0, d_value)
    
    # Print the points for verification
    print("\nGenerated triple points on the circle:")
    for i, point in enumerate(triple_points):
        print(f"Point {i+1}: {point}")
    
    # Generate nested triple points around each original triple point
    nested_radius = 0.01  # Small radius for nested points
    all_points = generate_nested_triple_points(triple_points, small_radius=nested_radius)
    
    # Print the nested points
    print("\nGenerated nested triple points:")
    for i, point in enumerate(all_points[3:]):
        print(f"Nested Point {i+1}: {point}")
    
    # Create a timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f'triple_ci_points_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all points to a file
    with open(os.path.join(output_dir, 'all_points.txt'), 'w') as f:
        f.write("# Original and nested triple points\n")
        f.write("# Type\tPoint_x\tPoint_y\tPoint_z\n")
        for i, point in enumerate(triple_points):
            f.write(f"Original\t{point[0]}\t{point[1]}\t{point[2]}\n")
        for i, point in enumerate(all_points[3:]):
            f.write(f"Nested\t{point[0]}\t{point[1]}\t{point[2]}\n")
    
    # Run the full analysis on original triple points
    print("\nAnalyzing original triple points...")
    all_ci_points, _ = find_triple_ci_points(
        base_R0=base_R0,
        d_value=d_value,
        d_start=d_start,
        d_end=d_end,
        d_step=d_step,
        aVx=1.0,
        aVa=5.0,
        x_shift=0.1,
        c_const=0.1,
        omega=1.0,
        epsilon=1e-10
    )
    
    # Run analysis on nested triple points
    print("\nAnalyzing nested triple points...")
    nested_ci_points = []
    for i, nested_point in enumerate(all_points[3:]):
        print(f"\nAnalyzing nested point {i+1}: {nested_point}")
        # Use a higher precision search for nested points
        nested_d_start = d_value - 5e-7  # Slightly wider range
        nested_d_end = d_value + 5e-7
        nested_d_step = 1e-9  # Much finer step size (100x more precise)
        
        # Find CI points around this nested point with higher precision
        points = find_va_vx_intersection(
            nested_d_start, nested_d_end, nested_d_step,
            aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0,
            R_0=nested_point, epsilon=1e-12  # Higher precision tolerance
        )
        
        # Add to the list of all CI points
        nested_ci_points.extend(points)
    
    # Combine original and nested CI points
    all_ci_points.extend(nested_ci_points)
    
    # Save all CI points to a file
    with open(os.path.join(output_dir, 'all_ci_points.txt'), 'w') as f:
        f.write("# d\ttheta\tR0_x\tR0_y\tR0_z\tType\n")
        for d, theta, R0 in all_ci_points[:15]:  # First 15 are from original triple points
            f.write(f"{d:.15f}\t{theta}\t{R0[0]}\t{R0[1]}\t{R0[2]}\tOriginal\n")
        for d, theta, R0 in all_ci_points[15:]:  # Rest are from nested triple points
            f.write(f"{d:.15f}\t{theta}\t{R0[0]}\t{R0[1]}\t{R0[2]}\tNested\n")
    
    # Create detailed plots of the CI points
    print(f"\nCreating detailed plots in: {output_dir}")
    plot_ci_points_detailed(triple_points, all_points, all_ci_points, output_dir)
    
    # Create potential plots for original triple points
    print("\nCreating potential plots for original triple points...")
    plot_potentials(triple_points, d_value, aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0, 
                   output_dir=output_dir, point_type='original')
    
    # Create potential plots for nested triple points
    print("\nCreating potential plots for nested triple points...")
    nested_points = all_points[len(triple_points):]
    plot_potentials(nested_points, d_value, aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0, 
                   output_dir=output_dir, point_type='nested')
    
    # Generate detailed reports with high numerical precision
    print("\nGenerating detailed report for original triple points...")
    generate_detailed_report(triple_points, d_value, aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0,
                           output_dir=output_dir, point_type='original')
    
    print("\nGenerating detailed report for nested triple points...")
    generate_detailed_report(nested_points, d_value, aVx=1.0, aVa=5.0, x_shift=0.1, c_const=0.1, omega=1.0,
                           output_dir=output_dir, point_type='nested')
    
    # Create a summary file explaining the plot organization
    with open(os.path.join(output_dir, 'potential_plots', 'README.txt'), 'w') as f:
        f.write("Potential Plot Organization\n")
        f.write("=======================\n\n")
        f.write("1. Original Triple Points (in ./original/ directory):\n")
        for i, point in enumerate(triple_points):
            f.write(f"   - Point {i+1}: R0 = {point}\n")
        f.write("\n2. Nested Triple Points (in ./nested/ directory):\n")
        for i, point in enumerate(nested_points):
            parent_idx = i // 3
            f.write(f"   - Point {i+1}: R0 = {point} (nested around original point {parent_idx+1})\n")
        f.write("\nEach directory contains the following plot types for each point:\n")
        f.write("   - Va_Vx_R0_N.png: Combined plot of Va and Vx vs theta\n")
        f.write("   - Va_minus_Vx_R0_N.png: Plot of Va-Vx vs theta\n")
        f.write("   - Vx_minus_Va_R0_N.png: Plot of Vx-Va vs theta\n")
        f.write("   - all_potentials_R0_N.png: Combined plot of all potential relationships\n")
