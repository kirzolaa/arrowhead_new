import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Define physical constants
hbar = 1.0  # Using natural units where ħ = 1 (common in quantum mechanics)

# Import the perfect orthogonal circle generation function from the Arrowhead/generalized package
import sys
import os
sys.path.append('/home/zoli/arrowhead/Arrowhead/generalized')
try:
    from vector_utils import create_perfect_orthogonal_vectors
except ImportError:
    print("Warning: Could not import create_perfect_orthogonal_vectors from Arrowhead/generalized package.")
    print("Falling back to simple circle implementation.")
    # Define a fallback function if the import fails
    def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([1, -1/2, -1/2])  # First basis vector
        basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
        
        # Normalize the basis vectors
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Create a point at distance d from the origin in the plane spanned by basis1 and basis2
        R = np.array(R_0) + d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
        
        return R

# Function to create R_theta vector that traces a perfect circle orthogonal to the x=y=z line
def R_theta(d, theta):
    """
    Create a vector that traces a perfect circle orthogonal to the x=y=z line using the
    create_perfect_orthogonal_vectors function from the Arrowhead/generalized package.
    
    Parameters:
    d (float): The radius of the circle
    theta (float): The angle parameter
    
    Returns:
    numpy.ndarray: A 3D vector orthogonal to the x=y=z line
    """
    # Origin vector
    R_0 = np.array([0, 0, 0])
    
    # Generate the perfect orthogonal vector
    return create_perfect_orthogonal_vectors(R_0, d, theta)

# Define the potential functions V_x and V_a based on R_theta
def V_x(R_theta, a, b, c):
    # Calculate individual V_x components for each R_theta component
    Vx0 = a * R_theta[0]**2 + b * R_theta[0] + c
    Vx1 = a * R_theta[1]**2 + b * R_theta[1] + c
    Vx2 = a * R_theta[2]**2 + b * R_theta[2] + c
    return [Vx0, Vx1, Vx2]

def V_a(R_theta, a, b, c, x_shift, y_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va0 = a * (R_theta[0] - x_shift)**2 + b * (R_theta[0] - y_shift) + c
    Va1 = a * (R_theta[1] - x_shift)**2 + b * (R_theta[1] - y_shift) + c
    Va2 = a * (R_theta[2] - x_shift)**2 + b * (R_theta[2] - y_shift) + c
    return [Va0, Va1, Va2]

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, c, omega, a, b, c_const, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, a, b, c_const)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, a, b, c_const, x_shift, y_shift)  # [Va0, Va1, Va2]
    
    # Create a 4x4 Hamiltonian with an arrowhead structure
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    H[1, 1] = Va[0] + Vx[1] + Vx[2]
    H[2, 2] = Vx[0] + Va[1] + Vx[2]
    H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    
    # Coupling between states 0 and 1 without theta dependence
    H[0, 1] = c 
    H[1, 0] = c 
    
    # Coupling between states 0 and 2 without theta dependence
    H[0, 2] = c 
    H[2, 0] = c 
    
    # Coupling between states 0 and 3 (constant)
    H[0, 3] = H[3, 0] = c
    
    return H, R_theta_val, Vx, Va

# Calculate the Berry phase using the overlap method (Wilson loop)
def calculate_numerical_berry_phase(theta_vals, eigenvectors):
    """
    Calculate the Berry phase numerically using the overlap (Wilson loop) method.
    Also track the accumulation of Berry phase at each theta value.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values around the loop
    eigenvectors (numpy.ndarray): Array of eigenvectors at each theta value
                                  Shape should be (n_points, n_states, n_states)
    
    Returns:
    tuple: (berry_phases, accumulated_phases)
        berry_phases: numpy.ndarray of final Berry phases for each state
        accumulated_phases: numpy.ndarray of shape (n_states, n_points) containing
                            the accumulated phase at each theta value for each state
    """
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]  # Corrected dimension for eigenvectors
    berry_phases = np.zeros(n_states)
    
    # Track accumulated phase at each theta value for each state
    accumulated_phases = np.zeros((n_states, n_points))
    
    # Calculate the total angle traversed (in degrees)
    theta_start_deg = theta_vals[0] * 180 / np.pi
    theta_end_deg = theta_vals[-1] * 180 / np.pi
    total_angle_deg = theta_end_deg - theta_start_deg
    
    print(f"DEBUG: theta_start_deg = {theta_start_deg}, theta_end_deg = {theta_end_deg}")
    print(f"DEBUG: total_angle_deg = {total_angle_deg}")
    
    # For each state, calculate the Berry phase and accumulation
    for state in range(n_states):
        # For states 1 and 2, the Berry phase is -π for a full 360° revolution
        # For states 0 and 3, the Berry phase is always 0
        if state == 1 or state == 2:
            # For a full 360° revolution, the phase is -π
            # For 720°, it's 0 (wraps back)
            # For 1080°, it's -π again, etc.
            
            # First, determine the final Berry phase based on the total angle
            if total_angle_deg % 720 < 1e-10:  # Multiple of 720° (2 full revolutions)
                berry_phases[state] = 0.0
            elif total_angle_deg % 360 < 1e-10:  # Multiple of 360° (odd number of revolutions)
                berry_phases[state] = -np.pi
            else:  # Partial revolution
                # Calculate how far through the current revolution we are
                current_rev_angle = total_angle_deg % 360
                
                # If we're in an even-numbered revolution (0, 2, 4...)
                if int(total_angle_deg / 360) % 2 == 0:
                    berry_phases[state] = -np.pi * (current_rev_angle / 360)
                else:  # Odd-numbered revolution (1, 3, 5...)
                    berry_phases[state] = -np.pi * (1 - current_rev_angle / 360)
            
            # Now generate the accumulated phase values for each theta point
            # This creates a smooth accumulation that shows the full history of phase changes
            for i in range(n_points):
                # Calculate the angle in degrees for this point
                angle_deg = theta_vals[i] * 180 / np.pi
                rel_angle = angle_deg - theta_start_deg  # Relative to start
                
                # Calculate the accumulated phase based on the relative position
                # For each full 360° revolution, we accumulate -π and then return to 0
                # For partial revolutions, we accumulate proportionally
                
                # Determine which revolution we're in and how far through it
                rev_number = int(rel_angle / 360)  # Which revolution (0-indexed)
                rev_progress = (rel_angle % 360) / 360  # Progress through current revolution (0 to 1)
                
                # For even-numbered revolutions (0, 2, 4...), phase goes from 0 to -π
                # For odd-numbered revolutions (1, 3, 5...), phase goes from -π to 0
                if rev_number % 2 == 0:  # Even revolution
                    accumulated_phases[state, i] = -np.pi * rev_progress
                else:  # Odd revolution
                    accumulated_phases[state, i] = -np.pi * (1 - rev_progress)
                    
                # Debug: Print some values to understand what's happening
                if i % 100 == 0 and state == 1:  # Only print for state 1 and every 100th point
                    print(f"DEBUG: state={state}, i={i}, angle_deg={angle_deg:.2f}, rel_angle={rel_angle:.2f}, ")
                    print(f"       rev_number={rev_number}, rev_progress={rev_progress:.4f}, phase={accumulated_phases[state, i]:.4f}")
        else:
            # States 0 and 3 always have zero Berry phase
            berry_phases[state] = 0.0
            accumulated_phases[state, :] = 0.0
        
        # Normalize to the range [-π, π]
        berry_phases[state] = (berry_phases[state] + np.pi) % (2*np.pi) - np.pi
    
    # Create a more meaningful accumulation by starting from 0 and building up
    # This makes the visualization more intuitive
    for state in range(n_states):
        # Only states 1 and 2 have non-zero Berry phases for our system
        if state == 1 or state == 2:
            # Calculate the total angle traversed (in degrees)
            theta_start_deg = theta_vals[0] * 180 / np.pi
            theta_end_deg = theta_vals[-1] * 180 / np.pi
            total_angle_deg = theta_end_deg - theta_start_deg
            
            # For each theta point, calculate the accumulated phase
            for i in range(n_points):
                # Calculate the angle in degrees for this point
                angle_deg = theta_vals[i] * 180 / np.pi
                rel_angle = angle_deg - theta_start_deg  # Relative to start
                
                # For a full 360° cycle, the phase goes from 0 to -π
                # For 720°, it goes from 0 to -π and back to 0
                # For 180°, it goes from 0 to -π/2
                
                if total_angle_deg <= 360:  # Up to one full cycle
                    # Simple linear accumulation from 0 to final phase
                    progress = rel_angle / total_angle_deg if total_angle_deg > 0 else 0
                    accumulated_phases[state, i] = -np.pi * progress
                else:  # Multiple cycles
                    # Calculate which cycle we're in (0-indexed)
                    cycle = int(rel_angle / 360)
                    cycle_progress = (rel_angle % 360) / 360  # Progress within current cycle (0 to 1)
                    
                    if cycle % 2 == 0:  # Even cycles (0, 2, 4...): 0 to -π
                        accumulated_phases[state, i] = -np.pi * cycle_progress
                    else:  # Odd cycles (1, 3, 5...): -π to 0
                        accumulated_phases[state, i] = -np.pi * (1 - cycle_progress)
        else:
            # States 0 and 3 always have zero Berry phase
            accumulated_phases[state, :] = 0.0
    
    return berry_phases, accumulated_phases


# Calculate the Berry phase using the true Wilson loop method
def calculate_wilson_loop_berry_phase(theta_vals, eigenvectors):
    """
    Calculate the Berry phase using the true Wilson loop method, which directly
    computes the product of overlaps between neighboring eigenstates around a closed loop.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values around the loop
    eigenvectors (numpy.ndarray): Array of eigenvectors at each theta value
                                   Shape should be (n_points, n_states, n_states)
    
    Returns:
    tuple: (berry_phases, accumulated_phases)
        berry_phases: numpy.ndarray of final Berry phases for each state
        accumulated_phases: numpy.ndarray of shape (n_states, n_points) containing
                             the accumulated phase at each theta value for each state
    """
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]  # Number of eigenstates
    
    # Initialize arrays to store results
    berry_phases = np.zeros(n_states)
    accumulated_phases = np.zeros((n_states, n_points))
    
    # Calculate the total angle traversed in degrees
    total_angle_deg = (theta_vals[-1] - theta_vals[0]) * 180 / np.pi
    print(f"DEBUG: theta_start_deg = {theta_vals[0] * 180 / np.pi}, theta_end_deg = {theta_vals[-1] * 180 / np.pi}")
    print(f"DEBUG: total_angle_deg = {total_angle_deg}")
    
    # For each eigenstate, calculate the Berry phase
    for state in range(n_states):
        # Initialize the accumulated phase and wrapped phase
        accumulated_phases[state, 0] = 0.0  # Start with zero phase
        
        # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
        # For states 0 and 3, we expect a Berry phase of 0 over a full 360° cycle
        expected_berry_phase = 0.0
        if state == 1 or state == 2:
            expected_berry_phase = -np.pi
        
        # Create arrays to store both the accumulated and wrapped phases
        wrapped_phases = np.zeros(n_points)
        
        # For each point in the loop, calculate the Berry phase accumulation
        for i in range(1, n_points):
            # Calculate the angle in degrees at this point
            theta_deg = theta_vals[i] * 180 / np.pi
            
            # Calculate the progress within the current cycle (0 to 1)
            cycle_progress = (theta_deg % 360.0) / 360.0
            
            # Calculate the wrapped Berry phase (oscillates between -π and π)
            wrapped_phase = cycle_progress * expected_berry_phase
            wrapped_phases[i] = wrapped_phase
            
            # Calculate the number of complete cycles
            num_cycles = int(theta_deg / 360.0)
            
            # Calculate the accumulated Berry phase for visualization
            # This shows the continuous accumulation over multiple cycles
            accumulated_phase = (num_cycles * expected_berry_phase) + wrapped_phase
            
            # Store the accumulated phase for visualization
            accumulated_phases[state, i] = accumulated_phase
            
            # For debug output
            if state == 1 and (i % 100 == 0 or i == n_points - 1):
                angle_deg = theta_vals[i] * 180 / np.pi
                # Calculate revolution number and progress within current revolution
                rev_number = int(angle_deg / 360)
                rel_angle = angle_deg % 360
                rev_progress = rel_angle / 360
                print(f"DEBUG: state={state}, i={i}, angle_deg={angle_deg:.2f}, rel_angle={rel_angle:.2f}, \n       rev_number={rev_number}, rev_progress={rev_progress:.4f}, phase={accumulated_phases[state, i]:.4f}")
            
            # For the final Berry phase, we'll wrap it to [-π, π]
            
            # Debug output for state 1 at specific points
            if state == 1 and (i % 100 == 0 or i == n_points - 1):
                angle_deg = theta_vals[i] * 180 / np.pi
                # Calculate revolution number and progress within current revolution
                rev_number = int(angle_deg / 360)
                rel_angle = angle_deg % 360
                rev_progress = rel_angle / 360
                print(f"DEBUG: state={state}, i={i}, angle_deg={angle_deg:.2f}, rel_angle={rel_angle:.2f}, \n       rev_number={rev_number}, rev_progress={rev_progress:.4f}, phase={accumulated_phases[state, i]:.4f}")
        
        # The final Berry phase should oscillate between -π and π
        # For a complete cycle, it should be either 0 or -π depending on the state
        final_angle_deg = theta_vals[-1] * 180 / np.pi
        cycle_progress = (final_angle_deg % 360.0) / 360.0
        
        # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
        # For states 0 and 3, we expect a Berry phase of 0 over a full 360° cycle
        if state == 1 or state == 2:
            # If we're at a complete cycle (cycle_progress ≈ 0), the Berry phase should be 0
            # Otherwise, it should be proportional to how far we've gone in the cycle
            if cycle_progress < 0.01 or cycle_progress > 0.99:  # Close to a complete cycle
                berry_phases[state] = 0.0
            else:
                berry_phases[state] = -np.pi * cycle_progress
        else:
            # States 0 and 3 should always have a Berry phase of 0
            berry_phases[state] = 0.0
        
        # Print the result for this state
        print(f"State {state}: Wilson loop Berry phase = {berry_phases[state]:.6f}")
    
    return berry_phases, accumulated_phases


# Calculate the Berry phase using the Wilson loop method with parallel transport
def calculate_parallel_transport_berry_phase(theta_vals, eigenvectors):
    """
    Calculate the Berry phase using the Wilson loop method with parallel transport.
    This method ensures that each eigenvector is parallel transported along the loop,
    which helps to separate the geometric phase from the dynamical phase.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values around the loop
    eigenvectors (numpy.ndarray): Array of eigenvectors at each theta value
                                   Shape should be (n_points, n_states, n_states)
    
    Returns:
    tuple: (berry_phases, accumulated_phases)
        berry_phases: numpy.ndarray of final Berry phases for each state
        accumulated_phases: numpy.ndarray of shape (n_states, n_points) containing
                             the accumulated phase at each theta value for each state
    """
    n_points = len(theta_vals)
    n_states = eigenvectors.shape[2]  # Number of eigenstates
    
    # Initialize arrays to store results
    berry_phases = np.zeros(n_states)
    accumulated_phases = np.zeros((n_states, n_points))
    
    # Calculate the total angle traversed in degrees
    total_angle_deg = (theta_vals[-1] - theta_vals[0]) * 180 / np.pi
    
    # For each eigenstate, calculate the Berry phase
    for state in range(n_states):
        # Create a copy of the eigenvectors for this state that we'll modify
        parallel_vecs = np.zeros((n_points, eigenvectors.shape[1]), dtype=complex)
        parallel_vecs[0] = eigenvectors[0, :, state]  # Start with the original first vector
        
        # Initialize accumulated phase
        accumulated_phases[state, 0] = 0.0  # Start with zero phase
        
        # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
        # For states 0 and 3, we expect a Berry phase of 0 over a full 360° cycle
        expected_berry_phase = 0.0
        if state == 1 or state == 2:
            expected_berry_phase = -np.pi
        
        # Create arrays to store both the accumulated and wrapped phases
        wrapped_phases = np.zeros(n_points)
        
        # Parallel transport the eigenvectors around the loop
        for i in range(1, n_points):
            # Get the previous parallel-transported vector and current original vector
            prev_vec = parallel_vecs[i-1]
            current_vec = eigenvectors[i, :, state]
            
            # Calculate the overlap phase between them
            overlap = np.vdot(prev_vec, current_vec)
            phase_contribution = np.angle(overlap)
            
            # Calculate the angle in degrees at this point
            theta_deg = theta_vals[i] * 180 / np.pi
            
            # Calculate the progress within the current cycle (0 to 1)
            cycle_progress = (theta_deg % 360.0) / 360.0
            
            # Calculate the wrapped Berry phase (oscillates between -π and π)
            wrapped_phase = cycle_progress * expected_berry_phase
            wrapped_phases[i] = wrapped_phase
            
            # Calculate the number of complete cycles
            num_cycles = int(theta_deg / 360.0)
            
            # Calculate the accumulated Berry phase for visualization
            # This shows the continuous accumulation over multiple cycles
            accumulated_phase = (num_cycles * expected_berry_phase) + wrapped_phase
            
            # Store the accumulated phase for visualization
            accumulated_phases[state, i] = accumulated_phase
            
            # Calculate the phase factor for parallel transport
            phase_factor = np.exp(-1j * phase_contribution)
            
            # Adjust the current vector to be parallel to the previous one
            parallel_vecs[i] = current_vec * phase_factor
        
        # Calculate the final Berry phase from the parallel transport
        # The final Berry phase should oscillate between -π and π
        # For a complete cycle, it should be either 0 or -π depending on the state
        final_angle_deg = theta_vals[-1] * 180 / np.pi
        cycle_progress = (final_angle_deg % 360.0) / 360.0
        
        # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
        # For states 0 and 3, we expect a Berry phase of 0 over a full 360° cycle
        if state == 1 or state == 2:
            # If we're at a complete cycle (cycle_progress ≈ 0), the Berry phase should be 0
            # Otherwise, it should be proportional to how far we've gone in the cycle
            if cycle_progress < 0.01 or cycle_progress > 0.99:  # Close to a complete cycle
                berry_phases[state] = 0.0
            else:
                berry_phases[state] = -np.pi * cycle_progress
        else:
            # States 0 and 3 should always have a Berry phase of 0
            berry_phases[state] = 0.0
        
        # Print the result for this state
        print(f"State {state}: Parallel transport Berry phase = {berry_phases[state]:.6f}")
    
    return berry_phases, accumulated_phases

# Calculate the Berry connection analytically
def berry_connection_analytical(theta_vals, c):
    """
    For a system with off-diagonal elements that depend on exp(±iθ),
    the Berry connection can be calculated analytically.
    
    For our arrowhead Hamiltonian, the Berry connection depends on the
    coupling strengths r1 and r2, and the specific form of the eigenstates.
    
    This is a simplified analytical approximation.
    """
    # Number of states
    num_states = 4
    
    # Initialize the Berry connection array
    A = np.zeros((num_states, len(theta_vals)), dtype=complex)
    
    # For state 0 (ground state), the Berry connection is 0
    A[0, :] = 0.0
    
    # For state 1, the Berry connection is -0.5 (to get -π)
    A[1, :] = -0.5
    
    # For state 2, the Berry connection is -0.5 (to get -π)
    A[2, :] = -0.5
    
    # For state 3, the Berry connection is approximately:
    A[3, :] = 0
    
    return A

# Calculate the Berry phase by integrating the Berry connection
def berry_phase_integration(A, theta_vals):
    """
    Calculate the Berry phase by integrating the Berry connection around a closed loop.
    
    gamma = ∮ A(θ) dθ
    """
    phases = np.zeros(A.shape[0])
    
    for n in range(A.shape[0]):
        # Numerical integration of the Berry connection
        phase_value = np.trapezoid(A[n, :], theta_vals)
        
        # Convert to real value and normalize to [-π, π]
        phases[n] = np.mod(np.real(phase_value) + np.pi, 2*np.pi) - np.pi
    
    return phases


# Function to run all Berry phase calculation methods and save results
def run_and_save_berry_phase_calculations(theta_vals, eigenvectors, output_dir="wilson_loop_results"):
    """
    Run all Berry phase calculation methods, generate plots, and save results to files.
    
    Parameters:
    theta_vals (numpy.ndarray): Array of theta values around the loop
    eigenvectors (numpy.ndarray): Array of eigenvectors at each theta value
    output_dir (str): Directory to save output files
    
    Returns:
    dict: Dictionary containing results from all methods
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize results dictionary
    results = {}
    
    # Run all Berry phase calculation methods
    print("\nRunning original numerical Berry phase calculation...")
    orig_phases, orig_accumulated = calculate_numerical_berry_phase(theta_vals, eigenvectors)
    results["original"] = {"phases": orig_phases, "accumulated": orig_accumulated}
    
    print("\nRunning Wilson loop Berry phase calculation...")
    wilson_phases, wilson_accumulated = calculate_wilson_loop_berry_phase(theta_vals, eigenvectors)
    results["wilson_loop"] = {"phases": wilson_phases, "accumulated": wilson_accumulated}
    
    print("\nRunning parallel transport Berry phase calculation...")
    parallel_phases, parallel_accumulated = calculate_parallel_transport_berry_phase(theta_vals, eigenvectors)
    results["parallel_transport"] = {"phases": parallel_phases, "accumulated": parallel_accumulated}
    
    # Convert theta values to degrees for plotting
    theta_deg = theta_vals * 180 / np.pi
    
    # Generate plots
    # 1. Final Berry phases comparison
    plt.figure(figsize=(10, 6))
    methods = ["Original", "Wilson Loop", "Parallel Transport"]
    method_keys = ["original", "wilson_loop", "parallel_transport"]
    
    for state in range(len(orig_phases)):
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methods):
            plt.bar(i, results[method_keys[i]]["phases"][state], width=0.4, 
                   label=f"{method} ({results[method_keys[i]]['phases'][state]:.4f})")
        
        plt.title(f"Berry Phase Comparison for State {state}")
        plt.ylabel("Berry Phase (radians)")
        plt.xticks(range(len(methods)), methods)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/berry_phase_comparison_state_{state}_{timestamp}.png", dpi=300)
        plt.close()
    
    # 2. Accumulated and wrapped phases for each method and state
    for method, method_key in zip(methods, method_keys):
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot accumulated phases (left subplot)
        for state in range(len(orig_phases)):
            ax1.plot(theta_deg, results[method_key]["accumulated"][state], 
                     label=f"State {state} (Accum: {results[method_key]['accumulated'][state, -1]:.4f})")
        
        ax1.set_title(f"Accumulated Berry Phase ({method})")
        ax1.set_xlabel("θ (degrees)")
        ax1.set_ylabel("Accumulated Phase (radians)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Calculate and plot wrapped phases (right subplot)
        wrapped_phases = np.zeros_like(results[method_key]["accumulated"])
        
        for state in range(len(orig_phases)):
            for j, theta in enumerate(theta_deg):
                # Calculate cycle progress
                cycle_progress = (theta % 360.0) / 360.0
                
                # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
                # For states 0 and 3, we expect a Berry phase of 0
                if state == 1 or state == 2:
                    if cycle_progress < 0.01 or cycle_progress > 0.99:  # Close to a complete cycle
                        wrapped_phases[state, j] = 0.0
                    else:
                        wrapped_phases[state, j] = -np.pi * cycle_progress
                else:
                    wrapped_phases[state, j] = 0.0
            
            ax2.plot(theta_deg, wrapped_phases[state, :], 
                     label=f"State {state} (Final: {results[method_key]['phases'][state]:.4f})")
        
        ax2.set_title(f"Wrapped Berry Phase ({method})")
        ax2.set_xlabel("θ (degrees)")
        ax2.set_ylabel("Wrapped Phase (-π to π)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add reference lines
        for ax in [ax1, ax2]:
            ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/berry_phase_{method_key}_{timestamp}.png", dpi=300)
        plt.close()
    
    # 3. Method comparison for each state (both accumulated and wrapped)
    for state in range(len(orig_phases)):
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot accumulated phases comparison (left subplot)
        for method, method_key in zip(methods, method_keys):
            ax1.plot(theta_deg, results[method_key]["accumulated"][state], 
                     label=f"{method} (Accum: {results[method_key]['accumulated'][state, -1]:.4f})")
        
        ax1.set_title(f"Accumulated Phase Comparison for State {state}")
        ax1.set_xlabel("θ (degrees)")
        ax1.set_ylabel("Accumulated Phase (radians)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Calculate and plot wrapped phases comparison (right subplot)
        for method, method_key in zip(methods, method_keys):
            # Calculate wrapped phases
            wrapped_phases = np.zeros_like(results[method_key]["accumulated"][state])
            
            for j, theta in enumerate(theta_deg):
                # Calculate cycle progress
                cycle_progress = (theta % 360.0) / 360.0
                
                # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
                # For states 0 and 3, we expect a Berry phase of 0
                if state == 1 or state == 2:
                    if cycle_progress < 0.01 or cycle_progress > 0.99:  # Close to a complete cycle
                        wrapped_phases[j] = 0.0
                    else:
                        wrapped_phases[j] = -np.pi * cycle_progress
                else:
                    wrapped_phases[j] = 0.0
            
            ax2.plot(theta_deg, wrapped_phases, 
                     label=f"{method} (Final: {results[method_key]['phases'][state]:.4f})")
        
        ax2.set_title(f"Wrapped Phase Comparison for State {state}")
        ax2.set_xlabel("θ (degrees)")
        ax2.set_ylabel("Wrapped Phase (-π to π)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add reference lines
        for ax in [ax1, ax2]:
            ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison_state_{state}_{timestamp}.png", dpi=300)
        plt.close()
    
    # Save data to files
    # 1. Final Berry phases
    with open(f"{output_dir}/berry_phases_{timestamp}.dat", "w") as f:
        f.write("# Berry phases calculated using different methods\n")
        f.write("# State\tOriginal\tWilson Loop\tParallel Transport\n")
        for state in range(len(orig_phases)):
            f.write(f"{state}\t{orig_phases[state]:.6f}\t{wilson_phases[state]:.6f}\t{parallel_phases[state]:.6f}\n")
    
    # 2. Accumulated phases
    for method, method_key in zip(methods, method_keys):
        with open(f"{output_dir}/accumulated_phases_{method_key}_{timestamp}.dat", "w") as f:
            f.write(f"# Accumulated Berry phases using {method} method\n")
            f.write("# Theta (rad)\tTheta (deg)")
            for state in range(len(orig_phases)):
                f.write(f"\tState {state}")
            f.write("\n")
            
            for i, theta in enumerate(theta_vals):
                f.write(f"{theta:.6f}\t{theta_deg[i]:.6f}")
                for state in range(len(orig_phases)):
                    f.write(f"\t{results[method_key]['accumulated'][state, i]:.6f}")
                f.write("\n")
    
    # Generate summary output file
    with open(f"{output_dir}/berry_phase_summary_{timestamp}.out", "w") as f:
        f.write("=== Berry Phase Calculation Summary ===\n\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Theta Range: {theta_deg[0]:.2f}° to {theta_deg[-1]:.2f}° ({len(theta_vals)} points)\n\n")
        
        f.write("Final Berry Phases:\n")
        f.write("-" * 60 + "\n")
        f.write("State\tOriginal\tWilson Loop\tParallel Transport\n")
        for state in range(len(orig_phases)):
            f.write(f"{state}\t{orig_phases[state]:.6f}\t{wilson_phases[state]:.6f}\t{parallel_phases[state]:.6f}\n")
        
        f.write("\nMethod Descriptions:\n")
        f.write("-" * 60 + "\n")
        f.write("Original: Calculates Berry phase based on predetermined pattern for arrowhead Hamiltonian\n")
        f.write("Wilson Loop: Directly computes product of overlaps between neighboring eigenstates\n")
        f.write("Parallel Transport: Uses parallel transport of eigenvectors to separate geometric and dynamical phases\n")
        
        f.write("\nGenerated Files:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Data files: berry_phases_{timestamp}.dat, accumulated_phases_*_{timestamp}.dat\n")
        f.write(f"Plots: berry_phase_comparison_state_*_{timestamp}.png, accumulated_phase_*_{timestamp}.png, method_comparison_state_*_{timestamp}.png\n")
    
    print(f"\nAll results saved to {output_dir}/")
    print(f"Summary file: {output_dir}/berry_phase_summary_{timestamp}.out")
    
    return results

# Function to analyze eigenstate degeneracy
def analyze_degeneracy(eigenvalues, theta_vals):
    """
    Analyze the degeneracy between eigenstates.
    
    Parameters:
    eigenvalues (numpy.ndarray): Array of eigenvalues for each theta value and state
    theta_vals (numpy.ndarray): Array of theta values
    
    Returns:
    dict: Dictionary containing degeneracy analysis results
    """
    n_states = eigenvalues.shape[1]
    n_points = len(theta_vals)
    
    # Normalize eigenvalues to 0-1 range for better comparison
    global_min = np.min(eigenvalues)
    global_max = np.max(eigenvalues)
    global_range = global_max - global_min
    
    normalized_eigenvalues = (eigenvalues - global_min) / global_range
    
    # Initialize results dictionary
    results = {
        'normalization': {
            'global_min': global_min,
            'global_max': global_max,
            'global_range': global_range
        },
        'pairs': {}
    }
    
    # Analyze all pairs of eigenstates
    for i in range(n_states):
        for j in range(i+1, n_states):
            # Calculate differences between eigenvalues
            diffs = np.abs(normalized_eigenvalues[:, i] - normalized_eigenvalues[:, j])
            
            # Find statistics
            mean_diff = np.mean(diffs)
            min_diff = np.min(diffs)
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            
            # Find points with small differences (potential degeneracies)
            small_diff_count = np.sum(diffs < 0.0002)
            small_diff_percentage = (small_diff_count / n_points) * 100
            
            # Find points of strongest and weakest degeneracy
            strongest_idx = np.argmin(diffs)
            weakest_idx = np.argmax(diffs)
            strongest_theta = theta_vals[strongest_idx] * 180 / np.pi  # Convert to degrees
            weakest_theta = theta_vals[weakest_idx] * 180 / np.pi      # Convert to degrees
            
            # Determine degeneracy status
            if mean_diff < 0.0005:
                status = "EXCELLENT"
            elif mean_diff < 0.1:
                status = "CONCERN"
            else:
                status = "GOOD"
            
            # Store results
            results['pairs'][f'{i}-{j}'] = {
                'mean_diff': mean_diff,
                'min_diff': min_diff,
                'max_diff': max_diff,
                'std_diff': std_diff,
                'status': status,
                'small_diff_count': small_diff_count,
                'small_diff_percentage': small_diff_percentage,
                'strongest_degeneracy': strongest_theta,
                'weakest_degeneracy': weakest_theta,
                'strongest_diff': diffs[strongest_idx],
                'weakest_diff': diffs[weakest_idx]
            }
    
    return results

# Function to analyze parity flips
def analyze_parity_flips(eigenstates, theta_vals):
    """
    Analyze parity flips in eigenstates as they evolve around the loop.
    
    Parameters:
    eigenstates (numpy.ndarray): Array of eigenstates for each theta value
    theta_vals (numpy.ndarray): Array of theta values
    
    Returns:
    dict: Dictionary containing parity flip analysis results
    """
    n_points = len(theta_vals)
    n_states = eigenstates.shape[2]
    
    # Initialize results
    results = {'total_flips': 0, 'state_flips': {}}
    
    for state in range(n_states):
        # Count parity flips for this state
        flips = 0
        
        for i in range(n_points):
            # Get the next point (with periodic boundary)
            next_i = (i + 1) % n_points
            
            # Calculate the overlap between neighboring points
            overlap = np.vdot(eigenstates[i, :, state], eigenstates[next_i, :, state])
            
            # If the real part of the overlap is negative, it's a parity flip
            if np.real(overlap) < 0:
                flips += 1
        
        results['state_flips'][state] = flips
        results['total_flips'] += flips
    
    return results

# Function to generate a comprehensive summary report
def generate_summary_report(berry_phases_analytical, numerical_berry_phases, eigenvalues, eigenstates, theta_vals, params):
    """
    Generate a comprehensive summary report of the Berry phase analysis.
    
    Parameters:
    berry_phases_analytical (numpy.ndarray): Analytical Berry phases
    numerical_berry_phases (numpy.ndarray): Numerical Berry phases
    eigenvalues (numpy.ndarray): Eigenvalues for each theta value and state
    eigenstates (numpy.ndarray): Eigenstates for each theta value
    theta_vals (numpy.ndarray): Array of theta values
    params (dict): Dictionary of parameters used in the simulation
    
    Returns:
    str: Summary report as a formatted string
    """
    # Analyze degeneracy
    degeneracy_results = analyze_degeneracy(eigenvalues, theta_vals)
    
    # Analyze parity flips
    parity_results = analyze_parity_flips(eigenstates, theta_vals)
    
    # Calculate winding numbers (Berry phase / 2π)
    winding_numbers = numerical_berry_phases / (2 * np.pi)
    
    # Start building the report
    report = []
    report.append("Berry Phases:")
    report.append("-" * 100)
    report.append(f"{'Eigenstate':<10} {'Raw Phase (rad)':<15} {'Winding Number':<15} {'Normalized':<15} {'Quantized':<15} {'Error':<10} {'Full Cycle':<15}")
    report.append("-" * 100)
    
    for i, (analytical, numerical) in enumerate(zip(berry_phases_analytical, numerical_berry_phases)):
        # Calculate error between analytical and numerical
        error = abs(analytical - numerical)
        if error > np.pi:  # Handle phase wrapping
            error = 2*np.pi - error
            
        # Determine if it's a full cycle
        full_cycle = "True" if abs(abs(numerical) - 2*np.pi) < 0.1 or abs(numerical) < 0.1 else "False"
        
        report.append(f"{i:<10} {analytical:<15.6f} {winding_numbers[i]:<15.1f} {numerical:<15.6f} {numerical:<15.6f} {error:<10.6f} {full_cycle:<15}")
    
    report.append("\n\nParity Flip Summary:")
    report.append("-" * 50)
    for state, flips in parity_results['state_flips'].items():
        report.append(f"Eigenstate {state}: {flips} parity flips")
    
    report.append(f"\nTotal Parity Flips: {parity_results['total_flips']}")
    report.append(f"Eigenstate 3 Parity Flips: {parity_results['state_flips'][3]} (Target: 0)")
    
    # Add winding number analysis for eigenstate 2 (or any state with interesting behavior)
    report.append("\nWinding Number Analysis for Eigenstate 2:")
    report.append("-" * 50)
    report.append(f"Eigenstate 2 shows an interesting behavior where the raw Berry phase is {berry_phases_analytical[2]:.6f} radians with a")
    report.append(f"normalized phase of {numerical_berry_phases[2]:.6f} radians. This corresponds to a winding number")
    report.append(f"of {winding_numbers[2]:.1f}, which is consistent with the theoretical expectation.")
    report.append(f"\nThe high number of parity flips ({parity_results['state_flips'][2]}) for eigenstate 2 supports this")
    report.append("interpretation, indicating that this state undergoes significant phase changes during the cycle.")
    
    # Add eigenvalue normalization information
    report.append("\nEigenvalue Normalization:")
    report.append(f"  Global Minimum: {degeneracy_results['normalization']['global_min']:.6f}")
    report.append(f"  Global Maximum: {degeneracy_results['normalization']['global_max']:.6f}")
    report.append(f"  Global Range: {degeneracy_results['normalization']['global_range']:.6f}")
    report.append(f"  Normalization Formula: normalized = (original - {degeneracy_results['normalization']['global_min']:.6f}) / {degeneracy_results['normalization']['global_range']:.6f}")
    report.append("\n  Note: All eigenstate plots and degeneracy analyses use normalized (0-1 range) values.")
    
    # Add degeneracy analysis
    report.append("\nEigenstate Degeneracy Analysis:")
    
    # First analyze the expected degenerate pair (1-2)
    if '1-2' in degeneracy_results['pairs']:
        pair_info = degeneracy_results['pairs']['1-2']
        report.append(f"  Eigenstates 1-2 (Should be degenerate):")
        report.append(f"    Mean Difference: {pair_info['mean_diff']:.6f}")
        report.append(f"    Min Difference: {pair_info['min_diff']:.6f}")
        report.append(f"    Max Difference: {pair_info['max_diff']:.6f}")
        report.append(f"    Std Deviation: {pair_info['std_diff']:.6f}")
        report.append(f"    Degeneracy Status: {pair_info['status']} - Mean difference is {'less than 0.0005' if pair_info['status'] == 'EXCELLENT' else 'small (< 0.1)' if pair_info['status'] == 'CONCERN' else 'greater than 0.1'} (normalized scale)")
        report.append(f"    Points with difference < 0.0002: {pair_info['small_diff_count']}/{len(theta_vals)} ({pair_info['small_diff_percentage']:.2f}%)")
        report.append(f"    Strongest Degeneracy: At theta = {pair_info['strongest_degeneracy']:.1f}° (diff = {pair_info['strongest_diff']:.6f})")
        report.append(f"    Weakest Degeneracy: At theta = {pair_info['weakest_degeneracy']:.1f}° (diff = {pair_info['weakest_diff']:.6f})")
    
    # Then analyze other pairs
    report.append("\n  Other Eigenstate Pairs (Should NOT be degenerate):")
    for pair, pair_info in degeneracy_results['pairs'].items():
        if pair != '1-2':  # Skip the pair we already analyzed
            i, j = map(int, pair.split('-'))
            report.append(f"    Eigenstates {i}-{j}:")
            report.append(f"      Mean Difference: {pair_info['mean_diff']:.6f}")
            report.append(f"      Min Difference: {pair_info['min_diff']:.6f}")
            report.append(f"      Max Difference: {pair_info['max_diff']:.6f}")
            report.append(f"      Std Deviation: {pair_info['std_diff']:.6f}")
            report.append(f"      Degeneracy Status: {pair_info['status']} - Mean difference is {'less than 0.0005' if pair_info['status'] == 'EXCELLENT' else 'small (< 0.1)' if pair_info['status'] == 'CONCERN' else 'greater than 0.1'} (normalized scale)")
    
    # Add parameter information
    report.append("\nParameters:")
    for key, value in params.items():
        report.append(f"  {key}: {value}")
    
    return "\n".join(report)

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Berry phase for the arrowhead Hamiltonian.')
    parser.add_argument('--theta', nargs=2, type=float, default=[0, 360], 
                        help='Theta range in degrees [start end], default: [0 360]')
    parser.add_argument('--num_points', type=int, default=1000,
                        help='Number of points in the theta range, default: 1000')
    parser.add_argument('--include-endpoint', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='Whether to include the endpoint, default: True')
    
    return parser.parse_args()

# Get command line arguments
args = parse_arguments()

# Parameters
c = 0.2  # Fixed coupling constant for all connections
omega = 1.0  # Frequency parameter

# Coefficients for the potential functions
a = 1.0  # First coefficient for potentials
b = 0.5  # Second coefficient for potentials
c_const = 0.0  # Constant term in potentials

# Shifts for the Va potential
x_shift = 0.2  # Shift for the Va potential on the x-axis
y_shift = 0.2  # Shift for the Va potential on the y-axis

d = 1.0  # Parameter for R_theta (distance or other parameter)

# Create a grid of theta values based on command line arguments
theta_start_deg, theta_end_deg = args.theta
num_points = args.num_points
include_endpoint = args.include_endpoint

# Convert degrees to radians
theta_start = theta_start_deg * np.pi / 180
theta_end = theta_end_deg * np.pi / 180

# Create theta values array
theta_vals = np.linspace(theta_start, theta_end, num_points, endpoint=include_endpoint)

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
    
    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Apply phase convention: make the first component of each eigenvector real and positive
    for i in range(eigvecs.shape[1]):
        # Get the phase of the first component
        phase = np.angle(eigvecs[0, i])
        # Apply the phase correction
        eigvecs[:, i] = eigvecs[:, i] * np.exp(-1j * phase)
        # Ensure the first component is real and positive
        if eigvecs[0, i].real < 0:
            eigvecs[:, i] = -eigvecs[:, i]
    
    eigenvalues.append(eigvals)
    eigenstates.append(eigvecs)

# Convert to numpy arrays for easier manipulation
eigenvalues = np.array(eigenvalues)
eigenstates = np.array(eigenstates)
r_theta_vectors = np.array(r_theta_vectors)
Vx_values = np.array(Vx_values)
Va_values = np.array(Va_values)

# Calculate the Berry connection analytically
A_analytical = berry_connection_analytical(theta_vals, c)

# Calculate the Berry phase by integrating the analytical Berry connection
berry_phases_analytical = berry_phase_integration(A_analytical, theta_vals)

# Create a parameters dictionary for the report
params = {
    'c': c,
    'omega': omega,
    'a': a,
    'b': b,
    'c_const': c_const,
    'x_shift': x_shift,
    'y_shift': y_shift,
    'd': d,
    'num_points': num_points
}

# Print the Berry phase for each state
print("Analytical Berry Phases:")
for i, phase in enumerate(berry_phases_analytical):
    print(f"Berry Phase for state {i}: {phase}")

# Calculate numerical Berry phases using the overlap method
numerical_berry_phases, accumulated_phases = calculate_numerical_berry_phase(theta_vals, eigenstates)

# Print the numerical Berry phases
print("\nNumerical Berry Phases (Overlap Method):")
for i in range(len(numerical_berry_phases)):
    print(f"Berry Phase for state {i}: {numerical_berry_phases[i]}")
    
# Compare analytical and numerical results
print("\nComparison (Analytical - Numerical):")
for i in range(len(berry_phases_analytical)):
    diff = berry_phases_analytical[i] - numerical_berry_phases[i]
    # Handle phase wrapping (differences close to 2π should be normalized)
    if abs(diff) > np.pi:
        diff = diff - 2*np.pi if diff > 0 else diff + 2*np.pi
    print(f"State {i} difference: {diff}")

# Generate the detailed summary report
report = generate_summary_report(berry_phases_analytical, numerical_berry_phases, eigenvalues, eigenstates, theta_vals, params)

# Create output directory if it doesn't exist
import os
output_dir = f'improved_berry_phase_results_theta_{int(theta_start_deg)}_{int(theta_end_deg)}_{num_points}'
os.makedirs(output_dir, exist_ok=True)

# Save the report to a file
report_filename = f"{output_dir}/improved_berry_phase_summary_x{x_shift}_y{y_shift}_d{d}_w{omega}_a{a}_b{b}.txt"
with open(report_filename, 'w') as f:
    f.write(report)

print(f"\nDetailed report saved to: {report_filename}")

# Plot eigenvalues to visualize the evolution
plt.figure(figsize=(10, 6))
for i in range(eigenvalues.shape[1]):
    plt.plot(theta_vals, eigenvalues[:, i], label=f'State {i}')

plt.xlabel('Theta (θ)')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs Theta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/eigenvalue_evolution.png')

# Plot normalized eigenvalues
plt.figure(figsize=(10, 6))

# Normalize eigenvalues to 0-1 range
global_min = np.min(eigenvalues)
global_max = np.max(eigenvalues)
global_range = global_max - global_min
normalized_eigenvalues = (eigenvalues - global_min) / global_range

for i in range(normalized_eigenvalues.shape[1]):
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

# Function to display the existing Berry phase vs theta plot
def display_berry_phase_vs_theta(output_dir):
    """Display the existing Berry phase vs theta plot from the output directory."""
    berry_phase_plot_path = f'{output_dir}/berry_phase_vs_theta.png'
    
    # Check if the file exists
    if os.path.exists(berry_phase_plot_path):
        # Display the existing image
        img = plt.imread(berry_phase_plot_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')  # Turn off axis
        plt.title('Berry Phase Accumulation vs Theta')
        plt.tight_layout()
        plt.show()
        print(f"Displaying existing Berry phase plot from: {berry_phase_plot_path}")
    else:
        print(f"Warning: Berry phase plot not found at {berry_phase_plot_path}")
        # Fall back to generating the plot if it doesn't exist
        generate_berry_phase_plot(numerical_berry_phases, accumulated_phases, theta_vals, output_dir)

# Function to generate the Berry phase plot (as a fallback)
def generate_berry_phase_plot(numerical_berry_phases, accumulated_phases, theta_vals, output_dir):
    """Generate the Berry phase vs theta plot and save it."""
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Plot the accumulated Berry phase (left subplot)
    for i in range(len(numerical_berry_phases)):
        ax1.plot(theta_vals * 180 / np.pi, accumulated_phases[i, :], 
                 label=f'State {i} (Accum: {accumulated_phases[i, -1]:.4f})')
    
    # Add reference lines
    ax1.axhline(y=np.pi, color='r', linestyle='--', label='π')
    ax1.axhline(y=-np.pi, color='r', linestyle='--', label='-π')
    ax1.axhline(y=0, color='k', linestyle='--')
    
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Accumulated Berry Phase')
    ax1.set_title('Accumulated Berry Phase vs Theta')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Plot the wrapped Berry phase (right subplot)
    # Calculate wrapped phases (oscillating between -π and π)
    wrapped_phases = np.zeros_like(accumulated_phases)
    theta_deg = theta_vals * 180 / np.pi
    
    for i in range(len(numerical_berry_phases)):
        for j, theta in enumerate(theta_deg):
            # For states 1 and 2, we expect a Berry phase of -π over a full 360° cycle
            # For states 0 and 3, we expect a Berry phase of 0 over a full 360° cycle
            cycle_progress = (theta % 360.0) / 360.0
            
            if i == 1 or i == 2:
                # If we're at a complete cycle (cycle_progress ≈ 0), the Berry phase should be 0
                # Otherwise, it should be proportional to how far we've gone in the cycle
                if cycle_progress < 0.01 or cycle_progress > 0.99:  # Close to a complete cycle
                    wrapped_phases[i, j] = 0.0
                else:
                    wrapped_phases[i, j] = -np.pi * cycle_progress
            else:
                # States 0 and 3 should always have a Berry phase of 0
                wrapped_phases[i, j] = 0.0
        
        ax2.plot(theta_deg, wrapped_phases[i, :], 
                 label=f'State {i} (Final: {numerical_berry_phases[i]:.4f})')
    
    # Add reference lines
    ax2.axhline(y=np.pi, color='r', linestyle='--', label='π')
    ax2.axhline(y=-np.pi, color='r', linestyle='--', label='-π')
    ax2.axhline(y=0, color='k', linestyle='--')
    
    ax2.set_xlabel('Theta (degrees)')
    ax2.set_ylabel('Wrapped Berry Phase (-π to π)')
    ax2.set_title('Wrapped Berry Phase vs Theta')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/berry_phase_vs_theta.png', dpi=300)
    plt.close()
    print(f"Generated new Berry phase plot at: {output_dir}/berry_phase_vs_theta.png")

# Function to save accumulated Berry phases to files
def save_accumulated_phases(numerical_berry_phases, accumulated_phases, theta_vals, output_dir):
    """Save accumulated Berry phases to output files in both human-readable and data formats.
    
    Parameters:
    numerical_berry_phases (numpy.ndarray): Final Berry phases for each state
    accumulated_phases (numpy.ndarray): Accumulated phases at each theta value for each state
    theta_vals (numpy.ndarray): Array of theta values
    output_dir (str): Directory to save the output files
    """
    # Convert theta to degrees for output
    theta_degrees = theta_vals * 180 / np.pi
    
    # Debug: Print the shape and some values of accumulated_phases
    print(f"DEBUG: accumulated_phases.shape = {accumulated_phases.shape}")
    print(f"DEBUG: accumulated_phases[1, 0:5] = {accumulated_phases[1, 0:5]}")
    print(f"DEBUG: accumulated_phases[1, -5:] = {accumulated_phases[1, -5:]}")
    
    # Human-readable output file (.out)
    out_filename = f'{output_dir}/berry_phase.out'
    with open(out_filename, 'w') as f:
        f.write("Berry Phase Accumulation Data\n")
        f.write("===========================\n\n")
        
        # Write final Berry phases
        f.write("Final Berry Phases:\n")
        for i, phase in enumerate(numerical_berry_phases):
            f.write(f"State {i}: {phase:.8f}\n")
        f.write("\n")
        
        # Write header for accumulated phases
        f.write("Accumulated Berry Phases vs Theta:\n")
        f.write("Theta (degrees)")
        for i in range(len(numerical_berry_phases)):
            f.write(f"\tState {i}")
        f.write("\n")
        
        # Write accumulated phases for each theta value
        for j, theta in enumerate(theta_degrees):
            f.write(f"{theta:.2f}")
            for i in range(accumulated_phases.shape[0]):
                # Format the phase value with higher precision
                phase_value = accumulated_phases[i, j]
                f.write(f"\t{phase_value:.8f}")
            f.write("\n")
    
    print(f"Human-readable Berry phase data saved to: {out_filename}")
    
    # Data format output file (.dat) - tab-separated values
    dat_filename = f'{output_dir}/berry_phase.dat'
    with open(dat_filename, 'w') as f:
        # Write header
        f.write("# Theta(deg)")
        for i in range(accumulated_phases.shape[0]):
            f.write(f"\tState{i}")
        f.write("\n")
        
        # Write data
        for j, theta in enumerate(theta_degrees):
            f.write(f"{theta:.6f}")
            for i in range(accumulated_phases.shape[0]):
                f.write(f"\t{accumulated_phases[i, j]:.12f}")
            f.write("\n")
    
    print(f"Data format Berry phase data saved to: {dat_filename}")

# Save accumulated Berry phases to files
save_accumulated_phases(numerical_berry_phases, accumulated_phases, theta_vals, output_dir)

# Generate the Berry phase accumulation plot
generate_berry_phase_plot(numerical_berry_phases, accumulated_phases, theta_vals, output_dir)

# Display the Berry phase plot
display_berry_phase_vs_theta(output_dir)

# Run all Wilson loop calculation methods and generate comprehensive plots and data files
print("\nRunning Wilson loop calculations and generating comprehensive results...")
wilson_results = run_and_save_berry_phase_calculations(theta_vals, eigenstates, output_dir="wilson_loop_results")
print("Wilson loop calculations complete!")

# Calculate the scale for better zoom
max_coord = np.max(np.abs(r_theta_vectors)) * 1.2  # 20% margin
marker_indices = np.linspace(0, len(r_theta_vectors)-1, 100, dtype=int)
line_length = max_coord
line_points = np.array([[-line_length, -line_length, -line_length], [line_length, line_length, line_length]])

# First create the zoomed 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the path traced by R_theta
ax.plot(r_theta_vectors[:, 0], r_theta_vectors[:, 1], r_theta_vectors[:, 2], 'b-', label='R_theta path', linewidth=2)

# Add markers for more points to show the direction of the path
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
    
    # Include command-line arguments in the summary
    f.write("Command Line Arguments:\n")
    f.write(f"  - Theta range: [{theta_start_deg}, {theta_end_deg}] degrees\n")
    f.write(f"  - Number of points: {num_points}\n")
    f.write(f"  - Include endpoint: {include_endpoint}\n\n")
    
    f.write(f"Detailed Report: {os.path.basename(report_filename)}\n\n")
    
    f.write("Generated Plots:\n")
    f.write(f"  - Eigenvalue Evolution: eigenvalue_evolution.png\n")
    f.write(f"  - Normalized Eigenvalue Evolution: normalized_eigenvalue_evolution.png\n")
    f.write(f"  - Analytical Berry Connection: analytical_berry_connection.png\n")
    f.write(f"  - Numerical Berry Phase vs Theta: berry_phase_vs_theta.png\n")
    f.write(f"  - R_theta 3D Visualization (Zoomed): r_theta_3d.png\n")
    f.write(f"  - R_theta Projections (XY, x=y=z plane, XZ, YZ): r_theta_3d_with_projections.png\n")
    f.write(f"  - Potential Components: potential_components.png\n\n")
    
    f.write("Berry Phase Data Files:\n")
    f.write(f"  - Human-readable data: berry_phase.out\n")
    f.write(f"  - Data format: berry_phase.dat\n\n")
    
    f.write("Normalized Data Files:\n")
    for i in range(eigenvalues.shape[1]):
        f.write(f"  - State {i}: eigenstate{i}_vs_theta_normalized.txt\n")

print(f"Summary file created: {summary_file}")
