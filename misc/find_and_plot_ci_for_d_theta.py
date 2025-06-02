import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from new_bph import Hamiltonian
from scripts.find_adaptive_d_values import analyze_specific_d_value
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_vectors


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
    
    for d in tqdm(d_values, desc="Searching for Va-Vx intersections"):
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
            points.append((d, 0))  # theta=0 is the reference point
            print(f"Found intersection at d={d:.15f}, Va-Vx={Va_minus_Vx_0}")
    
    return points

def find_and_plot_ci_for_d_theta(d_start=0.06123, d_end=0.06124, step=1e-10, aVx=1.0, aVa=5.0, 
                               x_shift=0.1, c_const=0.1, omega=1.0, R_0=(0.05, -0.025, -0.025), 
                               epsilon=1e-10):
    """
    Find and plot the CIs for d and theta where Va-Vx is numerically equal.
    
    Parameters:
    - d_start, d_end: range of d values to search
    - step: step size for d values
    - aVx, aVa, x_shift, c_const, omega: Hamiltonian parameters
    - R_0: reference point for the Hamiltonian
    - epsilon: tolerance for numerical equality
    """
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_dir = f'va_vx_intersection_{timestamp}'
    os.makedirs(parent_dir, exist_ok=True)
    
    # Find intersection points
    intersection_points = find_va_vx_intersection(d_start, d_end, step, aVx, aVa, x_shift, 
                                                c_const, omega, R_0, epsilon)
    
    # Save results
    with open(os.path.join(parent_dir, 'intersection_points.txt'), 'w') as f:
        f.write("# d\ttheta\n")
        for d, theta in intersection_points:
            f.write(f"{d:.15f}\t{theta}\n")
    
    # Plot results
    if intersection_points:
        d_vals = [p[0] for p in intersection_points]
        plt.figure(figsize=(12, 6))
        plt.plot(d_vals, np.zeros_like(d_vals), 'ro', markersize=8)
        plt.xlabel('d value')
        plt.title(f'Intersection points (Va-Vx) in range [{d_start}, {d_end}]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, 'intersection_points.png'), dpi=150)
        plt.close()
        
        # Print summary
        print(f"\nFound {len(intersection_points)} intersection points:")
        for i, (d, theta) in enumerate(intersection_points, 1):
            print(f"{i}. d = {d:.15f}, theta = {theta}")
    else:
        print("No intersection points found in the given range.")
    
    return intersection_points

if __name__ == '__main__':
    # Example usage with the specific range you mentioned
    find_and_plot_ci_for_d_theta(
        d_start=0.06123724356957945,
        d_end=0.06123724356957950,
        step=1e-15,  # Very small step for high precision
        aVx=1.0,
        aVa=5.0,
        x_shift=0.1,
        c_const=0.1,
        omega=1.0,
        R_0=np.array([0.05, -0.025, -0.025]),  # Must be a numpy array
        epsilon=1e-10
    )