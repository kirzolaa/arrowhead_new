from new_bph import Hamiltonian

import multiprocessing as mp
from functools import partial
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def fix_sign(eigvecs, printout):
    # Ensure positive real part of eigenvectors
    #with open(f'{output_dir}/eigvecs_sign_flips_{printout}.out', "a") as log_file:
    for i in range(eigvecs.shape[0]): #for every theta
        for j in range(eigvecs.shape[2]): #for every eigvec
            s = 0.0
            for k in range(eigvecs.shape[1]): #for every component
                s += np.real(eigvecs[i, k, j]) * np.real(eigvecs[i-1, k, j]) #dot product of current and previous eigvec
                """
                if s < 0:
                    log_file.write(f"Flipping sign of state {j} at theta {i} (s={s})\n")
                    log_file.write(f"Pervious eigvec: {eigvecs[i-1, :, j]}\n")
                    log_file.write(f"Current eigvec: {eigvecs[i, :, j]}\n")
                    eigvecs[i, :, j] *= -1
                """
    return eigvecs

def process_c_x_shift_gpu(c, x_shift, omega, aVx, aVa, R_0, d, theta_vals, output_dir):
    # Log current parameters on the console, use clear console befor printing
    os.system('clear')
    print(f'Processing c={c}, x_shift={x_shift}')
    
    # Create Hamiltonian with NumPy arrays since vector_utils expects NumPy
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c, R_0, d, theta_vals)
    H_thetas = hamiltonian.H_thetas()
    
    # Convert Hamiltonians to cupy arrays
    H_thetas_gpu = cp.array(H_thetas)
    
    # Calculate eigenvectors using cupy
    eigenvectors_gpu = cp.array([cp.linalg.eigh(H)[1] for H in H_thetas_gpu])
    eigenvectors_gpu = fix_sign(cp.asnumpy(eigenvectors_gpu), printout=0)
    eigenvectors_gpu = fix_sign(cp.asnumpy(eigenvectors_gpu), printout=0)
    
    # Convert back to numpy for plotting
    eigenvectors = cp.asnumpy(eigenvectors_gpu)
    """
    # Plotting remains the same as before
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Eigenvector Components - All eigenvectors\n(c={c}, x_shift={x_shift})', fontsize=16)
    
    for state in range(eigenvectors.shape[2]):
        for vect_comp in range(4):
            plt.subplot(2, 2, vect_comp + 1)
            plt.plot(theta_vals, np.real(eigenvectors[:, state, vect_comp]), label=f'Re(State {state})')
            plt.xlabel('Theta')
            plt.ylabel(f'Component {vect_comp}')
            plt.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/2D_figures/c_{c}_x_shift_{x_shift}.png')
    plt.close()
    """
    
    return np.real(eigenvectors)


def process_c_x_shift(c, x_shift, omega, aVx, aVa, R_0, d, theta_vals, output_dir):
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c, R_0, d, theta_vals)
    H_thetas = hamiltonian.H_thetas()
    r_theta = hamiltonian.R_thetas()
    
    # Calculate eigenvectors
    eigenvectors = fix_sign(np.array([np.linalg.eigh(H)[1] for H in H_thetas]), printout=0)
    eigenvectors = fix_sign(eigenvectors, printout=1)

    eigvals_all = np.array([np.linalg.eigh(H)[0] for H in H_thetas])
    
    # Plot eigenvector components
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Eigenvector Components - All eigenvectors\n(c={c}, x_shift={x_shift})', fontsize=16)
    
    for state in range(eigenvectors.shape[2]):
        for vect_comp in range(4):
            plt.subplot(2, 2, vect_comp + 1)
            plt.plot(theta_vals, np.real(eigenvectors[:, state, vect_comp]), label=f'Re(State {state})')
            plt.xlabel('Theta')
            plt.ylabel(f'Component {vect_comp}')
            plt.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/2D_figures/c_{c}_x_shift_{x_shift}.png')
    plt.close()
    
    return np.real(eigenvectors)

if __name__ == "__main__":
    
    #let a be an aVx and an aVa parameter
    aVx = 1.0
    aVa = 5.0
    c_const = 0.1  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 0.1  # Shift in x direction
    d = 0.2  # Radius of the circle, use unit circle for bigger radius
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 5000
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)


    #add a c and a x_shift parameter range
    c_range = np.linspace(0.001, 1, 25)
    x_shift_range = np.linspace(0.001, 1, 25)

    #create a directory for the output
    output_dir = '3D_figures_multiprocessed'
    os.makedirs(output_dir, exist_ok=True)
    ketD_plot_dir = '2D_figures'
    os.makedirs(os.path.join(output_dir, ketD_plot_dir), exist_ok=True)
    haromD_plot_dir = '3D_figures'
    os.makedirs(os.path.join(output_dir, haromD_plot_dir), exist_ok=True)
    state0_dir = 'state0'
    os.makedirs(os.path.join(output_dir, haromD_plot_dir, state0_dir), exist_ok=True)
    state1_dir = 'state1'
    os.makedirs(os.path.join(output_dir, haromD_plot_dir, state1_dir), exist_ok=True)
    state2_dir = 'state2'
    os.makedirs(os.path.join(output_dir, haromD_plot_dir, state2_dir), exist_ok=True)
    state3_dir = 'state3'
    os.makedirs(os.path.join(output_dir, haromD_plot_dir, state3_dir), exist_ok=True)

    def fix_sign(eigvecs, printout):
        # Ensure positive real part of eigenvectors
        with open(f'{output_dir}/eigvecs_sign_flips_{printout}.out', "a") as log_file:
            for i in range(eigvecs.shape[0]): #for every theta
                for j in range(eigvecs.shape[2]): #for every eigvec
                    s = 0.0
                    for k in range(eigvecs.shape[1]): #for every component
                        s += np.real(eigvecs[i, k, j]) * np.real(eigvecs[i-1, k, j]) #dot product of current and previous eigvec
                    if s < 0:
                        log_file.write(f"Flipping sign of state {j} at theta {i} (s={s})\n")
                        log_file.write(f"Pervious eigvec: {eigvecs[i-1, :, j]}\n")
                        log_file.write(f"Current eigvec: {eigvecs[i, :, j]}\n")
                        eigvecs[i, :, j] *= -1
        return eigvecs
    """
    #run throgh the c and x_shift ranges
    eigvecs_c_shiftre = []
    for c in c_range:
        for x_shift in x_shift_range:
            hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c, R_0, d, theta_vals)
            H_thetas = H_theta = hamiltonian.H_thetas()
            r_theta = R_thetas = hamiltonian.R_thetas()
            # Calculate eigenvectors
            eigenvectors = eigvecs_all = fix_sign(np.array([np.linalg.eigh(H)[1] for H in H_theta]), printout=0)
            eigenvectors = fix_sign(eigenvectors, printout=1)

            eigvals_all = np.array([np.linalg.eigh(H)[0] for H in H_theta]) #keruljuk a diagonalizalast megegyszer
            
            # Plot eigenvector components (4 subplots in a 2x2 grid for each eigenstate)
            plt.figure(figsize=(12, 12))
            plt.suptitle(f'Eigenvector Components - All eigenvectors\n(c={c}, x_shift={x_shift})', fontsize=16)  # Overall title
            
            for state in range(eigenvectors.shape[2]):
                #nest a for loop for vec_comp and use it like: :, state, vect_comp
                for vect_comp in range(4):
                    plt.subplot(2, 2, vect_comp + 1)  # Top left subplot
                    plt.plot(theta_vals, np.real(eigenvectors[:, state, vect_comp]), label=f'Re(State {state})')
                    #plt.plot(theta_vals, np.imag(eigenvectors[:, state, vect_comp]), label=f'Im(Comp {vect_comp})')
                    #plt.plot(theta_vals, np.abs(eigenvectors[:, state, vect_comp]), label=f'Abs(Comp {vect_comp})')
                    plt.xlabel('Theta')
                    plt.ylabel(f'Component {vect_comp}')
                    plt.legend()
                    
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for overall title
            plt.savefig(f'{output_dir}/2D_figures/c_{c}_x_shift_{x_shift}.png')
            plt.close()

            eigvecs_c_shiftre.append(np.real(eigenvectors))
    """
    """
    # Create a pool of processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Create partial function with fixed parameters
        func = partial(process_c_x_shift, 
                      omega=omega, 
                      aVx=aVx, 
                      aVa=aVa, 
                      R_0=R_0, 
                      d=d, 
                      theta_vals=theta_vals, 
                      output_dir=output_dir)
        
        # Create list of arguments
        args = [(c, x_shift) for c in c_range for x_shift in x_shift_range]
        
        # Process in parallel
        results = pool.starmap(func, args)
        
        # Combine results
        eigvecs_c_shiftre = np.array(results)
        print(f"Final shape: {eigvecs_c_shiftre.shape}")
    """
    # Create a pool of GPU processes
    with mp.Pool(processes=1) as pool:  # Use single process for GPU
        # Create partial function with fixed parameters
        func = partial(process_c_x_shift_gpu,
                      omega=omega, 
                      aVx=aVx, 
                      aVa=aVa, 
                      R_0=R_0, 
                      d=d, 
                      theta_vals=theta_vals, 
                      output_dir=output_dir)
        
        # Create list of arguments
        args = [(c, x_shift) for c in c_range for x_shift in x_shift_range]
        
        # Process in parallel
        results = pool.starmap(func, args)
        
        # Convert results to NumPy arrays
        eigvecs_c_shiftre_gpu = [cp.asnumpy(result) for result in results]

        for (c_val, x_shift_val), eigenvectors in zip(args, eigvecs_c_shiftre_gpu):
            plt.figure(figsize=(12, 12))
            plt.suptitle(f'Eigenvector Components - All eigenvectors\n(c={c_val}, x_shift={x_shift_val})', fontsize=16)
            for state in range(eigenvectors.shape[2]):
                for vect_comp in range(4):
                    plt.subplot(2, 2, vect_comp + 1)
                    plt.plot(theta_vals, np.real(eigenvectors[:, state, vect_comp]), label=f'Re(State {state})')
                    plt.xlabel('Theta')
                    plt.ylabel(f'Component {vect_comp}')
                    plt.legend()
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{output_dir}/2D_figures/c_{c_val}_x_shift_{x_shift_val}.png')
            plt.close()
        
        # Combine results
        eigvecs_c_shiftre_gpu = np.array(eigvecs_c_shiftre_gpu)
        print(f"Final shape: {eigvecs_c_shiftre_gpu.shape}")

    # Reshape for plotting
    eigvecs_c_shiftre_gpu = eigvecs_c_shiftre_gpu.reshape(len(c_range), len(x_shift_range), len(theta_vals), 4, 4)
    print(eigvecs_c_shiftre_gpu.shape)
    # Now shape: (5, 5, 5000, 4, 4)

    # After calculating eigenvectors
    # Create meshgrid for 3D plotting (transpose indexing to match dimensions)
    C, X, T = np.meshgrid(c_range, x_shift_range, theta_vals, indexing='ij')

    # Loop over states and components
    for state in range(4):
        state_dir = f'state{state}'
        os.makedirs(os.path.join(output_dir, haromD_plot_dir, state_dir), exist_ok=True)
    
        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvector Components - State {state}', fontsize=16)
        
        for comp in range(4):
            ax = plt.subplot(2, 2, comp + 1, projection='3d')
            Z = eigvecs_c_shiftre_gpu[:, :, :, state, comp]  # shape: (5, 5, 5000)
        
            # Note: T, C, Z all have shape (5, 5, 5000), as required
            ax.plot_surface(T[:,0,:], C[:,0,:], Z[:,0,:], cmap='viridis')  # Use slices to plot 2D surface
            
            ax.set_xlabel('Theta')
            ax.set_ylabel('c const')
            ax.set_zlabel(f'Component {comp}')
            ax.set_title(f'Component {comp}')
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, haromD_plot_dir, state_dir, f'theta_c_compOf{state}.png'))
        plt.close()

        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvector Components - State {state}', fontsize=16)
        
        fixed_c_idx = 2  # Choose a c value in the middle for a good cross-section

        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvector Components - State {state} (fixed c = {c_range[fixed_c_idx]:.3f})', fontsize=16)

        for comp in range(4):
            ax = plt.subplot(2, 2, comp + 1, projection='3d')
            Z = eigvecs_c_shiftre_gpu[fixed_c_idx, :, :, state, comp]  # shape: (x_shift, theta)
            
            ax.plot_surface(T[fixed_c_idx, :, :], X[fixed_c_idx, :, :], Z, cmap='viridis')
            
            ax.set_xlabel('Theta')
            ax.set_ylabel('x shift')
            ax.set_zlabel(f'Component {comp}')
            ax.set_title(f'Component {comp}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, haromD_plot_dir, state_dir, f'theta_x_shift_compOf{state}_c{fixed_c_idx}.png'))
        plt.close()