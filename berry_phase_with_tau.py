import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from new_bph import Hamiltonian
from eigenvals_vs_basis1_basis2_for_d import fix_sign

"""
def compute_berry_phase(eigvectors_all, R_thetas):
    #""#
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - R_thetas: ndarray of shape (N, 3), parameter-space path

    Returns:
    - berry_phases: ndarray of shape (M,), Berry phase for each eigenstate in radians
    #""#
    N, M, _ = eigvectors_all.shape
    berry_phases = np.zeros(M, dtype=np.float64)

    # Make sure path is closed
    if not np.allclose(R_thetas[0], R_thetas[-1]):
        eigvectors_all = np.concatenate([eigvectors_all, eigvectors_all[:1]], axis=0)
        R_thetas = np.concatenate([R_thetas, R_thetas[:1]], axis=0)
        N += 1

    # Compute gradient dR
    dR = np.gradient(R_thetas, axis=0)  # shape (N, 3)

    for n in range(M):
        total_phase = 0.0
        for i in range(1, N):
            psi_prev = eigvectors_all[i - 1, n, :]
            psi_curr = eigvectors_all[i, n, :]

            # Normalize for safety
            psi_prev = psi_prev / np.linalg.norm(psi_prev)
            psi_curr = psi_curr / np.linalg.norm(psi_curr)

            # Finite difference approximation of ∇_R |ψ>
            delta_psi = psi_curr - psi_prev
            grad_psi = delta_psi / np.linalg.norm(R_thetas[i] - R_thetas[i - 1])

            # τ = ⟨ψ_i | ∇_R | ψ_{i-1}⟩ · dR
            tau = np.vdot((psi_curr), grad_psi) * np.linalg.norm(dR[i])
            total_phase += tau

        berry_phases[n] = total_phase

    # Take imaginary part (Berry phase is real-valued in radians)
    return np.real(berry_phases)
"""

import numpy as np

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
    num_points = 5000
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
    
    #create a directory for the output
    output_dir = 'berry_phase_with_tau'
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
    eigenvectors = fix_sign(np.array([np.linalg.eigh(H)[1] for H in H_thetas]), printout=0)
    eigenvectors = fix_sign(eigenvectors, printout=0)
    
    # Calculate Berry phases
    berry_phases_all = compute_berry_phase_wilson(eigenvectors)
    print(f"Berry phases: {berry_phases_all}")
    
    # Save results
    np.save(f'{npy_dir}/berry_phases_all.npy', berry_phases_all)
    
    print("Berry phases computed and saved.")