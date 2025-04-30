pip install cupy-cuda11x  # or the appropriate version for your CUDA

import cupy as cp

def process_c_x_shift_gpu(c, x_shift, omega, aVx, aVa, R_0, d, theta_vals, output_dir):
    # Convert numpy arrays to cupy arrays
    theta_vals_gpu = cp.array(theta_vals)
    
    # Create Hamiltonian and calculate eigenvectors on GPU
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c, R_0, d, theta_vals_gpu)
    H_thetas = hamiltonian.H_thetas()
    
    # Convert Hamiltonians to cupy arrays
    H_thetas_gpu = cp.array(H_thetas)
    
    # Calculate eigenvectors using cupy
    eigenvectors_gpu = cp.array([cp.linalg.eigh(H)[1] for H in H_thetas_gpu])
    eigenvectors_gpu = fix_sign(eigenvectors_gpu, printout=0)
    eigenvectors_gpu = fix_sign(eigenvectors_gpu, printout=1)
    
    # Convert back to numpy for plotting
    eigenvectors = cp.asnumpy(eigenvectors_gpu)
    
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
    
    return np.real(eigenvectors)

if __name__ == "__main__":
    # ... (rest of your existing code)
    
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
        
        # Combine results
        eigvecs_c_shiftre = np.array(results)
        print(f"Final shape: {eigvecs_c_shiftre.shape}")