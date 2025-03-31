"""
Berry Phase Analysis Script

This script uses functions from improved_berry_phase.py to generate Va, Vx, and arrowhead matrices,
and calculates the Berry phase using the Wilson loop method.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import datetime
from scipy.constants import hbar
import multiprocessing

# Import the perfect orthogonal circle generation function from the Arrowhead/generalized package
import sys
import os
sys.path.append('/home/zoltan/arrowhead_new_new/arrowhead_new/generalized')
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle
from main import *
print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")

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
def V_x(R_theta, aVx):
    # Calculate individual V_x components for each R_theta component
    Vx = [aVx * (R_theta[i] ** 2) for i in range(len(R_theta))]
    return Vx

def V_a(R_theta, aVa, c, x_shift):
    # Calculate individual V_a components with shifts applied for each R_theta component
    Va = [aVa * ((R_theta[i] - x_shift) ** 2) + c for i in range(len(R_theta))]
    return Va

# Define the Hamiltonian matrix with explicit Berry phase terms
def hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d):
    # Calculate R_theta for the current theta and parameters
    R_theta_val = R_theta(d, theta)
    
    # Calculate the potentials V_x and V_a (each returns a list of 3 components)
    Vx = V_x(R_theta_val, aVx)  # [Vx0, Vx1, Vx2]
    Va = V_a(R_theta_val, aVa, c_const, x_shift)  # [Va0, Va1, Va2]
    
    # Create a 4x4 Hamiltonian with an arrowhead structure
    H = np.zeros((4, 4), dtype=complex)
    
    # Set the diagonal elements
    #H[0, 0] = Vx[0] + Vx[1] + Vx[2] + hbar * omega
    #H[1, 1] = Va[0] + Vx[1] + Vx[2]
    #H[2, 2] = Vx[0] + Va[1] + Vx[2]
    #H[3, 3] = Vx[0] + Vx[1] + Va[2]
    
    #or we can do it like this:
    #H[0, 0] = hbar * omega + [sum of all V
    sumVx = sum(Vx)
    H[0, 0] = hbar * omega + sumVx
    for i in range(1, len(H)):
        H[i, i] = H[0, 0] + Va[i-1] - Vx[i-1]
        

    # Set the off-diagonal elements with explicit theta dependence
    # These terms will create a non-zero Berry phase
    """
    # Coupling between states 0 and 1 without theta dependence
    H[0, 1] = c 
    H[1, 0] = c 
    
    # Coupling between states 0 and 2 without theta dependence
    H[0, 2] = c 
    H[2, 0] = c 
    
    # Coupling between states 0 and 3 (constant)
    H[0, 3] = H[3, 0] = c
    """
    for i in range(1, len(H)):
        H[i, 0] = H[0, i] = c

    return H, R_theta_val, Vx, Va



def calculate_berry_curvature(eigenvectors, theta_vals, output_dir):
    """
    Calculate the Berry curvature with central difference and boundary handling.
    """
    num_theta = len(theta_vals)
    num_bands = eigenvectors.shape[2]
    #eigenvectors = fix_gauge(eigenvectors)
    curvature = np.zeros((num_theta, num_bands))  # Now includes boundary points

    with open(f'{output_dir}/curvature.out', "w") as log_file:
        log_file.write("#Theta " + " ".join([f"Curv_{j}" for j in range(num_bands)]) + "\n")

        for i in range(num_theta):
            for j in range(num_bands):
                if i == 0:  # Forward difference at the boundary
                    dtheta = theta_vals[1] - theta_vals[0]
                    curvature[i, j] = np.imag(np.conj(eigenvectors[i, :, j]).T @ eigenvectors[i + 1, :, j]) / dtheta
                elif i == num_theta - 1:  # Backward difference at the boundary
                    dtheta = theta_vals[-1] - theta_vals[-2]
                    curvature[i, j] = np.imag(np.conj(eigenvectors[i - 1, :, j]).T @ eigenvectors[i, :, j]) / dtheta
                else:  # Central difference
                    dtheta = theta_vals[i + 1] - theta_vals[i - 1]
                    curvature[i, j] = np.imag(np.conj(eigenvectors[i - 1, :, j]).T @ eigenvectors[i + 1, :, j]) / dtheta

            log_file.write(f"{theta_vals[i]:.6f} " + " ".join([f"{curvature[i, j]:.6f}" for j in range(num_bands)]) + "\n")

    return curvature

#lets log the simplified method in the test.py
def calculate_berry_phase_with_berry_curvature_simplified(theta_vals, eigenvectors, output_dir):
    curvature = calculate_berry_curvature(eigenvectors, theta_vals, output_dir)
    num_bands = eigenvectors.shape[2]
    num_theta = len(theta_vals)
    berry_phases = np.zeros(num_bands)
    accumulated_phases = np.zeros((num_bands, num_theta))

    with open(f'{output_dir}/phase_log_berry_curvature_simplified.out', "w") as log_file:
        log_file.write("#Theta " + " ".join([f"Phase_{j}" for j in range(num_bands)]) + "\n")

        for j in range(num_bands):
            berry_phase = np.zeros(num_theta)
            berry_phase[0] = 0
            for i in range(1, num_theta):
                berry_phase[i] = berry_phase[i - 1] + curvature[i, j] * (theta_vals[i] - theta_vals[i - 1]) #simple rectangular rule no wrapping.
            berry_phases[j] = berry_phase[-1]
            print(f"Berry phase for state {j}: {berry_phases[j]:.15f}")
            accumulated_phases[j] = berry_phase

        for i in range(num_theta):
            log_file.write(f"{theta_vals[i]:.15f} " + " ".join([f"{accumulated_phases[j, i]:.15f}" for j in range(num_bands)]) + "\n")

    return berry_phases, accumulated_phases

#multiprocess the calculate_berry_phase_with_berry_curvature_simplified function and ensure that we are using the num_cpus -1
def calculate_berry_phase_for_band(j, theta_vals, eigenvectors, output_dir):
    num_bands = eigenvectors.shape[2]
    num_theta = len(theta_vals)
    curvature = calculate_berry_curvature(eigenvectors, theta_vals, output_dir)
    
    # Calculate berry phase
    berry_phase = np.zeros(num_theta)
    berry_phase[0] = 0
    for i in range(1, num_theta):
        berry_phase[i] = berry_phase[i - 1] + curvature[i, j] * (theta_vals[i] - theta_vals[i - 1]) #simple rectangular rule no wrapping.
    
    return berry_phase[-1], berry_phase

def calculate_berry_phase_with_berry_curvature_simplified_multiprocessing(theta_vals, eigenvectors, output_dir):
    num_bands = eigenvectors.shape[2]
    num_theta = len(theta_vals)
    berry_phases = np.zeros(num_bands)
    accumulated_phases = np.zeros((num_bands, num_theta))
    
    with open(f'{output_dir}/phase_log_berry_curvature_simplified.out', "w") as log_file:
        log_file.write("#Theta " + " ".join([f"Phase_{j}" for j in range(num_bands)]) + "\n")
        
        # Use multiprocessing to calculate berry phases for each band
        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                calculate_berry_phase_for_band,
                [(j, theta_vals, eigenvectors, output_dir) for j in range(num_bands)]
            )
            
        # Unpack results
        for j, (berry_phase, accumulated_phase) in enumerate(results):
            berry_phases[j] = berry_phase
            accumulated_phases[j] = accumulated_phase
            print(f"Berry phase for state {j}: {berry_phases[j]:.15f}")

        for i in range(num_theta):
            log_file.write(f"{theta_vals[i]:.15f} " + " ".join([f"{accumulated_phases[j, i]:.15f}" for j in range(num_bands)]) + "\n")

    return berry_phases, accumulated_phases

def calculate_berry_phase_with_berry_curvature(theta_vals, eigenvectors, output_dir):
    curvature = calculate_berry_curvature(eigenvectors, theta_vals, output_dir)
    num_bands = eigenvectors.shape[2]
    num_theta = len(theta_vals)
    berry_phases = np.zeros(num_bands)
    accumulated_phases = np.zeros((num_bands, num_theta))

    with open(f'{output_dir}/phase_log_berry_curvature.out', "w") as log_file:
        log_file.write("#Theta " + " ".join([f"Phase_{j}" for j in range(num_bands)]) + "\n")

        for j in range(num_bands):
            berry_phase = np.zeros(num_theta, dtype=complex)
            berry_phase[0] = 0
            for i in range(1, num_theta):
                berry_phase[i] = berry_phase[i - 1] + np.trapezoid(curvature[:i+1, j], theta_vals[:i+1])
                berry_phase[i] = (np.angle(berry_phase[i]) + np.pi) % (2 * np.pi) - np.pi
            berry_phases[j] = berry_phase[-1].real
            accumulated_phases[j] = berry_phase.real

        for i in range(num_theta):
            log_file.write(f"{theta_vals[i]:.15f} " + " ".join([f"{accumulated_phases[j, i]:.15f}" for j in range(num_bands)]) + "\n")

    return berry_phases, accumulated_phases

if __name__ == '__main__':
    # Parameters for the arrowhead matrix
    c = 0.2  # Coupling constant
    omega = 0.1  # Frequency
    #a = 1.0  # Potential parameter
    #let a be an aVx and an aVa parameter
    aVx = 1.0
    aVa = 5.0
    b = 1.0  # Potential parameter
    c_const = 1.0  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 1.0  # Shift in x direction
    y_shift = 0.0  # Shift in y direction --> turns out that this is not a y axis shift like I wanted it!!!!!
    d = 0.001  # Radius of the circle
    theta_min = 0
    theta_max = 2 * np.pi
    num_points = 50
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    # Calculate eigenvectors at each theta value, explicitly including endpoint
    eigenvectors = []
    for i, theta in enumerate(theta_vals):
        # Diagonalize Hamiltonian
        evals, evecs = np.linalg.eigh(hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d)[0])
        eigenvectors.append(evecs)

    eigenvectors = np.array(eigenvectors)
    output_dir = f'output_berry_phase_results_thetamin_{theta_min:.2f}_thetamax_{theta_max:.2f}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    out_dir = os.path.join(output_dir, 'out')
    #create a new directory for the vectors
    save_dir = os.path.join(output_dir, 'vectors')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    #use the perfect_orthogonal_circle.py script to visualize the R_theta vectors
    from perfect_orthogonal_circle import verify_circle_properties, visualize_perfect_orthogonal_circle, generate_perfect_orthogonal_circle

    #visualize the R_theta vectors
    points = multiprocessing_create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max) #we already have a method for this
    #points = create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max)
    print(points.shape)
    visualize_perfect_orthogonal_circle(points, save_dir)
    verify_circle_properties(d, num_points, points, save_dir)

    with open(f'{out_dir}/eigenvector_diff.out', "a") as log_file:
        log_file.write('#State Theta Norm_Diff\n')
        for i in range(1, len(theta_vals)):
            for j in range(eigenvectors.shape[2]):
                log_file.write(f"State {j}, Theta {theta_vals[i]:.2f}: {np.linalg.norm(eigenvectors[i, j] - eigenvectors[i-1, j]):.6f}\n")
        log_file.close()
    """
    def fix_gauge(eigenvectors):
        for i in range(1, len(eigenvectors)):
            for j in range(eigenvectors.shape[2]):
                overlap = np.dot(np.conj(eigenvectors[i - 1, :, j]), eigenvectors[i, :, j])
                if np.real(overlap) < 0:
                    eigenvectors[i, :, j] *= -1
        return eigenvectors
    """
    
    # Calculate and plot eigenstate overlaps
    overlaps = np.zeros((eigenvectors.shape[2], len(theta_vals)), dtype=complex)

    plt.figure(figsize=(12, 6))

    for state in range(eigenvectors.shape[2]):
        for i in range(len(theta_vals)):
            current_eigenvector = eigenvectors[i, :, state]
            next_eigenvector = eigenvectors[(i + 1) % len(theta_vals), :, state]
            # Include endpoint by using the first eigenvector for the last point
            if i == len(theta_vals) - 1:
                next_eigenvector = eigenvectors[0, :, state]
            overlaps[state, i] = np.conj(current_eigenvector).T @ next_eigenvector
            if overlaps[state, i] < 0:
                overlaps[state, i] *= -1 # Ensure positive overlap
    
        plt.plot(theta_vals, np.real(overlaps[state]), label=f'State {state}')

    plt.xlabel('Theta')
    plt.ylabel('Overlap')
    plt.title('Eigenstate Overlaps')
    plt.legend()
    plt.grid()
    plt.savefig(f'{figures_dir}/eigenstate_overlaps.png')


    #plots
    #plot the H*v aka Hamiltonian times eigenvectors weighted by the eigenvalues
    plt.figure(figsize=(12, 6))
    for state in range(eigenvectors.shape[2]):
        # Calculate H*v for each theta value
        Hv_results = np.zeros((len(theta_vals), eigenvectors.shape[1]), dtype=complex)
        #get eigenvaluesof each H_theta, it is not theta vals
        #calculate H_thetas array by calculating H_theta, it should be a (num_points, 4, 4) array, like (theta_value, 4, 4)
        H_thetas = np.array([hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d)[0] for theta in theta_vals])
        print(H_thetas.shape)
        # Get all the eigenvalues
        eigenvalues = np.array([np.linalg.eigvalsh(H) for H in H_thetas])
    
        # Get all eigenvalues and eigenvectors separately
        eigenvals_eigvecs = [np.linalg.eigh(H) for H in H_thetas]
        eigenvalues_full = np.array([ev[0] for ev in eigenvals_eigvecs])
        # Extract the eigenvalues and eigenvectors
        eigenvalues = np.array([ev[0] for ev in eigenvals_eigvecs])
        eigenstates = np.array([ev[1] for ev in eigenvals_eigvecs])
    
        # For reference, H*v = λ*v
        #calculate H*v
        #use H_thetas and eigenstates
        for i, theta in enumerate(theta_vals):
            Hv_results[i] = H_thetas[i] @ eigenstates[i, :, state]
    
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        for j in range(4):  # Just plot components for one state at a time
            axs[j].plot(theta_vals, np.real(Hv_results[:, j]), 'b-', label='H*v (Real)')
            axs[j].plot(theta_vals, np.imag(Hv_results[:, j]), 'b--', label='H*v (Imag)')
            #plot the magnitude of H*v
            axs[j].plot(theta_vals, np.abs(Hv_results[:, j]), 'r-', label='|H*v|')
            
            axs[j].set_title(f'Component {j}')
            axs[j].set_xlabel('Theta')
            axs[j].set_ylabel('Value')
            axs[j].grid(True)
            axs[j].legend()
        
        plt.tight_layout()
        plt.suptitle(f'H*v for State {state}')
        plt.subplots_adjust(top=0.92)
        
        plt.savefig(f'{figures_dir}/H_times_v_state_{state}.png')
        plt.close()

        #plot the lambda*v as H*v
        for j in range(4):
            lambda_v = eigenvalues_full[:, j] * eigenstates[:, j, state]

            # Ensure lambda_v is 2-dimensional
            lambda_v = lambda_v.reshape(-1, 1)  # Reshape to (num_theta_vals, 1)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.flatten()
            
            for k in range(4):
                axs[k].plot(theta_vals, np.real(lambda_v[:, 0]), 'b-', label='λ*v (Real)')
                axs[k].plot(theta_vals, np.imag(lambda_v[:, 0]), 'b--', label='λ*v (Imag)')
                #plot the magnitude of lambda*v
                axs[k].plot(theta_vals, np.abs(lambda_v[:, 0]), 'r-', label='|λ*v|')
            
                axs[k].set_title(f'Component {k}')
                axs[k].set_xlabel('Theta')
                axs[k].set_ylabel('Value')
                axs[k].grid(True)
                axs[k].legend()
        
            plt.tight_layout()
            plt.suptitle(f'λ*v for State {state}')
            plt.subplots_adjust(top=0.92)
            
            plt.savefig(f'{figures_dir}/lambda_times_v_state_{state}.png')
            plt.close()
        
    

    #berry phases using berry curvature simplified
    berry_phases, accumulated_phases = calculate_berry_phase_with_berry_curvature_simplified_multiprocessing(theta_vals, eigenvectors, output_dir)
    #berry phases using wilson loop
    #berry_phases, accumulated_phases = calculate_wilson_loop_berry_phase_new(theta_vals, eigenvectors)


    # Write berry_phases to a .out file
    with open(f'{output_dir}/berry_phases.out', 'w') as f:
        f.write('Berry Phase Accumulation Data\n')
        f.write('===========================\n\n')
        f.write('Final Berry Phases:\n')
        for state, phase in enumerate(berry_phases):
            f.write(f'State {state}: {phase:.8f}\n')
        f.write('\nAccumulated Berry Phases vs Theta:\n')
        f.write('Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
        for i, theta in enumerate(theta_vals):
            theta_deg = np.degrees(theta)
            f.write(f'{theta_deg:.2f}\t')
            for state in range(len(berry_phases)):
                f.write(f'{accumulated_phases[state][i]:.8f}\t')
            f.write('\n')

    # Write berry_phases to a .dat file
    np.savetxt(f'{out_dir}/berry_phases.dat', berry_phases, header='Berry phases for each state')

    # Write accumulated_phases to a .dat file with theta values
    with open(f'{out_dir}/accumulated_phases.dat', 'w') as f:
        f.write('# Theta (radians)\tState 0\tState 1\tState 2\tState 3\n')
        for i, theta in enumerate(theta_vals):
            f.write(f'{theta:.8f}\t')
            np.savetxt(f, accumulated_phases[:, i].reshape(1, -1), fmt='%.8f', delimiter='\t')

    # Write theta values to a .dat file
    np.savetxt(f'{out_dir}/theta_values.dat', theta_vals, header='Theta values used in calculation')

    # Write eigenstate overlaps to file
    with open(f'{out_dir}/eigenstate_overlaps.out', 'w') as f:
        f.write('# Eigenstate Overlaps vs Theta\n')
        f.write('# Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
        for i, theta in enumerate(theta_vals):
            theta_deg = np.degrees(theta)
            f.write(f'{theta_deg:.2f}\t')
            for state in range(eigenvectors.shape[2]):
                f.write(f'{overlaps[state, i]:.8f}\t')
            f.write('\n')

    # Write eigenvectors to file
    with open(f'{out_dir}/eigenvectors.out', 'w') as f:
        f.write('Eigenvectors vs Theta\n')
        for i, theta in enumerate(theta_vals):
            theta_deg = np.degrees(theta)
            f.write(f'Theta = {theta_deg:.2f} degrees\n')
            for state in range(eigenvectors.shape[2]):
                f.write(f'State {state}:\n')
                np.savetxt(f, eigenvectors[i, :, state].reshape(1, -1), fmt='%.8f')
            f.write('\n')
    
    #plot accumulated phases
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for state in range(len(berry_phases)):
        plt.plot(theta_vals, accumulated_phases[state], label=f'State {state}')
    plt.xlabel('Theta')
    plt.ylabel('Berry Phase')
    plt.title('Berry Phase vs Theta')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(theta_vals, accumulated_phases.T)
    plt.xlabel('Theta')
    plt.ylabel('Accumulated Phase')
    plt.title('Accumulated Phase vs Theta')

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/accumulated_phases.png')

    #calculate and save the Hamiltonians, Va and Vx into .npy files
    # Assuming you have defined the Hamiltonian function and potential functions
    Hamiltonians = []
    Va_values = []
    Vx_values = []

    for theta in theta_vals:
        H, R_theta_val, Vx, Va = hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d)
        Hamiltonians.append(H)
        Va_values.append(Va)
        Vx_values.append(Vx)

    # Convert lists to numpy arrays
    Hamiltonians = np.array(Hamiltonians)
    Va_values = np.array(Va_values)
    Vx_values = np.array(Vx_values)

    #create a directory in the output directory for npy files
    #output_dir = os.path.join(output_dir, 'output_berry_phase_results_thetamin_0.00_thetamax_6.28_20250324150750')
    npy_dir = os.path.join(output_dir, 'npy')

    # Create the directory if it doesn't exist
    os.makedirs(npy_dir, exist_ok=True)

    # Save the Hamiltonians, Va and Vx into .npy files
    np.save(f'{npy_dir}/Hamiltonians.npy', Hamiltonians)
    np.save(f'{npy_dir}/Va_values.npy', Va_values)
    np.save(f'{npy_dir}/Vx_values.npy', Vx_values)

    #plot Va potential components
    plt.figure(figsize=(12, 6))
    Va_values = np.load(f'{npy_dir}/Va_values.npy')
    for i in range(3):
        plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Va Components')
    plt.title('Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/Va_components.png')

    #plot Vx potential components
    plt.figure(figsize=(12, 6))
    Vx_values = np.load(f'{npy_dir}/Vx_values.npy')
    for i in range(3):
        plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Vx Components')
    plt.title('Vx Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/Vx_components.png')

    # Initialize lists to store eigenvalues and eigenvectors
    eigenvalues_list = []
    eigenvectors_list = []

    for theta in theta_vals:
        H, R_theta_val, Vx, Va = hamiltonian(theta, c, omega, aVx, aVa, b, c_const, x_shift, y_shift, d)
    
        # Calculate eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(H)  # Use np.linalg.eigh for Hermitian matrices
        eigenvalues_list.append(evals)
        eigenvectors_list.append(evecs)

    # Convert lists to numpy arrays
    eigenvalues_array = np.array(eigenvalues_list)
    eigenvectors_array = np.array(eigenvectors_list)

    # Save the eigenvalues and eigenvectors into .npy files
    np.save(f'{npy_dir}/eigenvalues.npy', eigenvalues_array)
    np.save(f'{npy_dir}/eigenvectors.npy', eigenvectors_array)

    # Plot the eigenvalues
    plt.figure(figsize=(12, 6))

    # Transpose eigenvalues_array for correct plotting
    for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
        plt.plot(theta_vals, eigenvalues_array[:, i], label=f'Eigenvalue {i+1}')

    plt.xlabel('Theta (θ)')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues vs Theta')
    plt.grid(True)
    plt.legend()  # Add legend to identify each eigenvalue
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/eigenvalues.png')

    ## Check for degeneracy
    tolerance = 1e-3  # Define a tolerance level for degeneracy
    small_difference_threshold = 0.1  # Define a threshold for small differences
    degeneracy_list = []
    near_degeneracy_list = []
    difference_list = []
    report = []

    for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
        for j in range(i + 1, eigenvalues_array.shape[1]):
            difference = np.abs(eigenvalues_array[:, i] - eigenvalues_array[:, j])
            difference_list.append(difference)
            if np.all(difference < tolerance):
                degeneracy_list.append((i, j))  # Store the indices of degenerate states
            elif np.all(difference < small_difference_threshold):
                near_degeneracy_list.append((i, j))  # Store the indices of near-degenerate states

    # Log the degeneracy results
    with open(f'{out_dir}/degeneracy_check.out', 'w') as log_file:
        log_file.write("# Degeneracy Check\n")
        log_file.write("======================================\n\n")
        log_file.write(f"Parameters:\n")
        log_file.write(f"Tolerance level: {tolerance}\n")
        log_file.write(f"Small difference threshold: {small_difference_threshold}\n\n")
        log_file.write("======================================\n\n")
        if not degeneracy_list and not near_degeneracy_list:
            log_file.write("No degeneracies or near degeneracies found.\n")
        if not degeneracy_list and near_degeneracy_list:
            log_file.write("No degenerate eigenstates found.\n")
        for state1, state2 in near_degeneracy_list:
            log_file.write(f"Eigenstates {state1} and {state2} are near degenerate.\n")
        if not near_degeneracy_list and degeneracy_list:
            log_file.write("No near-degenerate eigenstates found.\n")
            for state1, state2 in degeneracy_list:
                log_file.write(f"Eigenstates {state1} and {state2} are degenerate.\n")
        log_file.write('\n')
        # Log the differences between each eigenstate
        log_file.write('======================================\nDifferences between eigenstates:\n')
        for i in range(eigenvalues_array.shape[1]):  # Loop through each eigenstate
            for j in range(i + 1, eigenvalues_array.shape[1]):
                difference = np.abs(eigenvalues_array[:, i] - eigenvalues_array[:, j])
                log_file.write(f"Eigenstates {i} and {j}: {difference}\n")  # Convert to string
    
    # Write detailed text report
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write(f'Berry Phase Analysis Report\n')
        f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'Parameters:\n')
        f.write(f'  c = {c}\n')
        f.write(f'  omega = {omega}\n')
        f.write(f'  aVx = {aVx}\n')
        f.write(f'  aVa = {aVa}\n')
        f.write(f'  b = {b}\n')
        f.write(f'  c_const = {c_const}\n')
        f.write(f'  x_shift = {x_shift}\n')
        f.write(f'  y_shift = {y_shift}\n')
        f.write(f'  d = {d}\n\n')
        f.write(f'Calculation Parameters:\n')
        f.write(f'  Theta range: [{theta_min}, {theta_max}]\n')
        f.write(f'  Number of points: {len(theta_vals)}\n')
        f.write(f'  Number of states: {len(berry_phases)}\n\n')
        f.write(f'Results:\n')
        for state, phase in enumerate(berry_phases):
            f.write(f'  State {state}: Berry phase = {phase:.6f}\n')
        f.write(f'\nDegenerate Eigenstates:\n')
        if degeneracy_list:
            for state1, state2 in degeneracy_list:
                f.write(f'  Eigenstates {state1} and {state2} are degenerate.\n')
        else:
            f.write('  No degeneracies found.\n')
        f.write('Detailed degeneracy check logged in f"{out_dir}/degeneracy_check.out"\n')
    
        f.write('\n')
        f.write('Berry curvature logged in f"{out_dir}/phase_log_berry_curvature.out"\n')
        f.write('Berry phases logged in f"{output_dir}/berry_phases.out"\n')
        f.write('\n')
        f.write('Eigenvalue plot saved as f"{figures_dir}/eigenvalues.png"\n')

        f.write('Eigenvector differences logged in f"{out_dir}/eigenvector_diff.out"\n')
        f.write('\nEigenvalues and Eigenvectors:\n')
        f.write('To load the eigenvalues and eigenvectors, use:\n')
        f.write('eigenvalues = np.load(f"{npy_dir}/eigenvalues.npy")\n')
        f.write('eigenvectors = np.load(f"{npy_dir}/eigenvectors.npy")\n')
    
        f.write('\nFor Hamiltonians, Va, Vx:\n')
        f.write('Use np.load() to load the data as a numpy array. Example usage:\n')
        f.write('H = np.load(f"{npy_dir}/Hamiltonians.npy")\n')
        f.write('Va = np.load(f"{npy_dir}/Va_values.npy")\n')
        f.write('Vx = np.load(f"{npy_dir}/Vx_values.npy")\n')
    
        if report:
            f.write('\nNear Degenerate Eigenstates:\n')
            for line in report:
                f.write(line + '\n')

        f.write('\n')
        f.write('Summary logged in f"{output_dir}/summary.txt"\n')
        f.write('\n')
        f.write('Done.\n')
