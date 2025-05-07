import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from new_bph import Hamiltonian


def fix_sign(eigvecs, printout, output_dir):
    # Ensure positive real part of eigenvectors
    with open(f'{output_dir}/eigvecs_sign_flips_{printout}.out', "a") as log_file:
        sign = +1
        for i in range(eigvecs.shape[0]): #for every theta
            for j in range(eigvecs.shape[2]): #for every eigvec
                s = 0.0
                for k in range(eigvecs.shape[1]): #for every component
                    s += sign * np.real(eigvecs[i, k, j]) * np.real(eigvecs[i-1, k, j]) #dot product of current and previous eigvec
                    if s * sign < 0 and printout == 1:
                        log_file.write(f"Flipping sign of state {j} at theta {i} (s={s}, sign={sign})\n")
                        log_file.write(f"Pervious eigvec: {eigvecs[i-1, :, :]}\n")
                        log_file.write(f"Current eigvec:  {eigvecs[i, :, :]}\n")
                        sign = -sign
                    if sign == -1:
                        eigvecs[i, :, j] *= -1
    return eigvecs


def compute_berry_phase(eigvectors_all, R_thetas, theta_vals):
    """
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - R_thetas: ndarray of shape (N, 3), parameter-space path

    Returns:
    - tau: ndarray of shape (M, M, N), Berry connection for each eigenstate in radians
    - gamma: ndarray of shape (M, M, N), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    
    tau = np.zeros((M, M, N), dtype=np.complex128)
    gamma = np.zeros((M, M, N), dtype=np.float64)

    for n in range(M):
        #total_phase = np.zeros(M, dtype=np.float64), use gamma instead
        for m in range(M):
            for i in range(N):
                psi_prev = eigvectors_all[i - 1, :, n] # (theta_vals, components of the eigvec, eigvec_state)
                psi_curr = eigvectors_all[i, :, m]
                psi_next = eigvectors_all[(i + 1) % N, :, n]

                # Normalize for safety
                psi_prev = psi_prev / np.linalg.norm(psi_prev)
                psi_curr = psi_curr / np.linalg.norm(psi_curr)
                psi_next = psi_next / np.linalg.norm(psi_next)

                # Finite difference approximation of ∇_theta |ψ>
                delta_psi = psi_next - psi_prev
                grad_psi = delta_psi / (theta_vals[i] - theta_vals[i-2])

                # τ = ⟨ψ_i | ∇_theta | ψ_{i-1}⟩
                tau[n, m, i] = np.vdot(np.conj(psi_curr).T, grad_psi)
                # · d_theta to integrate
                gamma[n, m, i] = gamma[n,m,i-1] + tau[n, m, i] * (theta_vals[i] - theta_vals[i-1])

    # Take imaginary part (Berry phase is real-valued in radians)
    return tau, gamma

import numpy as np

def fix_sign_og(eigvecs, printout, output_dir):
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


def compute_berry_phase_wilson(eigvectors_all):
    """
    Compute Berry phases γ_n using the Wilson loop method for each eigenstate.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)

    Returns:
    - berry_phases: ndarray of shape (M,), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    berry_phases_all = np.zeros(M)
    """
    # Ensure loop is closed
    if not np.allclose(eigvectors_all[0], eigvectors_all[-1]):
        eigvectors_all = np.concatenate([eigvectors_all, eigvectors_all[:1]], axis=0)
        N += 1
    """
    for n in range(M):
        product = 1.0 + 0.0j
        for i in range(1, N):
            psi_prev = eigvectors_all[i - 1, :, n]
            psi_curr = eigvectors_all[i, :, n]

            # Normalize
            psi_prev /= np.linalg.norm(psi_prev)
            psi_curr /= np.linalg.norm(psi_curr)

            # Overlap between adjacent eigenstates
            overlap = np.vdot(np.conj(psi_prev).T, psi_curr)
            product *= overlap / np.abs(overlap)

        berry_phases_all[n] = np.angle(product)

    return berry_phases_all


if __name__ == "__main__":
    
    
    aVx = 1.0
    aVa = 5.0
    c_const = 0.1  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 0.1  # Shift in x direction
    d = 0.1  # Radius of the circle, use unit circle for bigger radius
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 50000
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
    
    #create a directory for the output
    output_dir = 'berry_phase_with_tau_and_gamma'
    os.makedirs(output_dir, exist_ok=True)
    
    #create a directory for the npy files
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    #create a directory for the plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    H_thetas = hamiltonian.H_thetas()
    R_thetas = hamiltonian.R_thetas()
    
    # Calculate eigenvectors
    eigenvectors = fix_sign_og(np.array([np.linalg.eigh(H)[1] for H in H_thetas]), printout=1, output_dir=output_dir)
    #eigenvectors = fix_sign(eigenvectors, printout=0)
    
    # Calculate Berry phases
    berry_phases_all = compute_berry_phase_wilson(eigenvectors)
    print(f"Berry phases Wilson-looppal: {berry_phases_all}")
    
    # Save results
    np.save(f'{npy_dir}/berry_phases_all.npy', berry_phases_all)
    
    print("Berry phases computed and saved.")

    # plot the eigenvalues vs theta
    plt.figure()
    for i in range(eigenvectors.shape[2]):
        plt.plot(theta_vals, np.linalg.eigvalsh(H_thetas)[:, i])
    plt.xlabel('Theta')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues vs Theta')
    plt.savefig(f'{plot_dir}/eigenvalues.png')
    plt.close()

    tau, gamma = compute_berry_phase(eigenvectors, R_thetas, theta_vals)
    print(f"Berry phases with tau:\n {gamma[:,:,-1]}")
    print(f"theta[0]: {theta_vals[0]}")
    print(f"theta[-1]: {theta_vals[-1]}")
    # Save results
    np.save(f'{npy_dir}/tau.npy', tau)
    np.save(f'{npy_dir}/gamma.npy', gamma)
    
    print("Berry phases computed and saved.")

    # plot tau_01, tau_12, tau_23
    plt.figure()
    plt.plot(theta_vals, tau[0,1,:])
    plt.plot(theta_vals, tau[1,2,:])
    plt.plot(theta_vals, tau[2,0,:])
    plt.xlabel('Theta')
    plt.ylabel('tau')
    plt.title('tau vs Theta')
    plt.savefig(f'{plot_dir}/tau.png')
    plt.close()

    # plot gamma_01, gamma_12, gamma_23
    plt.figure()
    plt.plot(theta_vals, gamma[0,1,:])
    plt.plot(theta_vals, gamma[1,2,:])
    plt.plot(theta_vals, gamma[2,0,:])
    plt.xlabel('Theta')
    plt.ylabel('gamma')
    plt.title('gamma vs Theta')
    plt.savefig(f'{plot_dir}/gamma.png')
    plt.close()

    # Plot eigenvector components (4 subplots in a 2x2 grid for each eigenstate)
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Eigenvector Components - All eigenvectors', fontsize=16)  # Overall title
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
    plt.savefig(f'{plot_dir}/eigenvector_components_for_eigvec_2x2.png')
    plt.close()

    berry_phases = np.zeros(4)
    for n in range(4):
        berry_phases[n] = np.sum((tau[n, n, :]) * np.diff(np.append(theta_vals[-1]-2*np.pi, theta_vals)))
    print(f"Berry phases with tau: {berry_phases}")
    # Save results
    np.save(f'{npy_dir}/berry_phases_tau.npy', berry_phases)
    
    print("Berry phases with and from tau computed and saved.")
