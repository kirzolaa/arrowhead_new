import numpy as np

def compute_berry_phase_overlap(eigvectors_all):
    """
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space,
    using the overlap (Wilson loop) method.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors at each θ

    Returns:
    - berry_phases: ndarray of shape (M,), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    berry_phases = np.zeros(M)

    for n in range(M):
        phase_sum = 0.0
        for i in range(N):
            psi_i = eigvectors_all[i, :, n]
            psi_next = eigvectors_all[(i + 1) % N, :, n]

            # Normalize (defensive)
            psi_i /= np.linalg.norm(psi_i)
            psi_next /= np.linalg.norm(psi_next)

            # Overlap gives phase evolution
            overlap = np.vdot(psi_i, psi_next)
            phase_diff = np.angle(overlap)
            phase_diff = np.unwrap(np.array([phase_diff]))  # Ensure phase is continuous

            # Correct for sign flips if necessary, update: this ids implemented implicitly
            #if np.vdot(psi_i, psi_next) < 0:
            #    phase_diff += np.pi

            phase_sum += phase_diff[0]

        berry_phases[n] = phase_sum

    return berry_phases


def compute_berry_phase_overlap_og(eigvectors_all):
    """
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space,
    using the overlap (Wilson loop) method.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors at each θ

    Returns:
    - berry_phases: ndarray of shape (M,), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    berry_phases = np.zeros(M)

    for n in range(M):
        phase_sum = 0.0
        for i in range(N):
            psi_i = eigvectors_all[i, :, n]
            psi_next = eigvectors_all[(i + 1) % N, :, n]

            # Normalize (defensive)
            psi_i /= np.linalg.norm(psi_i)
            psi_next /= np.linalg.norm(psi_next)

            # Overlap gives phase evolution
            overlap = np.vdot(psi_i, psi_next)
            phase = np.angle(overlap)
            phase = np.unwrap(np.array([phase]))  # Ensure phase is continuous
            phase_sum += phase

        berry_phases[n] = phase_sum

    return berry_phases

import numpy as np
import matplotlib.pyplot as plt

def compute_cumulative_berry_phases(eigvectors_all):
    N, M, _ = eigvectors_all.shape
    cumulative_phases = np.zeros((M, N))

    for n in range(M):
        phase_sum = 0.0
        for i in range(N):
            psi_i = eigvectors_all[i, :, n]
            psi_next = eigvectors_all[(i + 1) % N, :, n]

            psi_i /= np.linalg.norm(psi_i)
            psi_next /= np.linalg.norm(psi_next)

            overlap = np.vdot(psi_i, psi_next)
            phase = np.angle(overlap)
            phase_sum += phase
            cumulative_phases[n, i] = phase_sum

    return cumulative_phases

def compute_berry_phase_other(eigvectors_all, theta_vals):
    """
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - theta_vals: ndarray of shape (N,), angle values in radians

    Returns:
    - tau: ndarray of shape (M, N), Berry connection ⟨ψ | ∇ψ⟩
    - berry_phases: ndarray of shape (M,), Berry phase for each eigenstate
    """
    N, M, _ = eigvectors_all.shape
    tau = np.zeros((M, N), dtype=np.complex128)

    for n in range(M):
        for i in range(N):
            psi_prev = eigvectors_all[i - 1, :, n]
            psi_curr = eigvectors_all[i, :, n]
            psi_next = eigvectors_all[(i + 1) % N, :, n]

            psi_prev /= np.linalg.norm(psi_prev)
            psi_curr /= np.linalg.norm(psi_curr)
            psi_next /= np.linalg.norm(psi_next)

            delta_psi = psi_next - psi_prev
            delta_theta = theta_vals[(i + 1) % N] - theta_vals[i - 1]
            tau[n, i] = np.vdot(psi_curr, delta_psi) / delta_theta

    # Trapezoidal integration of Im[τ]
    berry_phases = np.zeros(M)
    for n in range(M):
        integrand = np.imag(tau[n, :])
        berry_phases[n] = np.trapz(integrand, x=theta_vals)

    return tau, berry_phases


if __name__ == "__main__":
    
    #load the tau.npy file:
    tau = np.load("/run/user/1000/gvfs/sftp:host=rick.phys.unideb.hu,user=kirzolaa/data/kirzolaa/arrowhead_new/berry_phase_corrected_run/npy/tau.npy")
    print("Tau diagonal:", tau[0,0,:])
    print("Tau diagonal imaginary part:", np.imag(tau[0,0,:]))
    print("Tau diagonal real part:", np.real(tau[0,0,:]))

    try:
        #load the theta_vals.npy file:
        theta_vals = np.load("/run/user/1000/gvfs/sftp:host=rick.phys.unideb.hu,user=kirzolaa/data/kirzolaa/arrowhead_new/berry_phase_corrected_run/npy/theta_vals.npy")
        print("Theta values:", theta_vals.shape)
    except  FileNotFoundError:
        print("Theta values not found")
        theta_min = 0.0
        theta_max = 2.0 * np.pi
        num_points = 50000
        theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
        
    
    #load the eigenvectors.npy file:
    eigenvectors = np.load("/run/user/1000/gvfs/sftp:host=rick.phys.unideb.hu,user=kirzolaa/data/kirzolaa/arrowhead_new/berry_phase_corrected_run/npy/eigvecs.npy")
    print("Eigenvectors:", eigenvectors.shape)
    
    #compute the berry phases
    berry_phases = compute_berry_phase_overlap(eigenvectors)
    print("Berry phases:", berry_phases)
    """
    cumulative_phases = compute_cumulative_berry_phases(eigenvectors)

    plt.figure(figsize=(8, 5))
    for n in range(cumulative_phases.shape[0]):
        plt.plot(theta_vals, cumulative_phases[n], label=f"State {n}")
        
    plt.xlabel("θ (radians)")
    plt.ylabel("Cumulative Berry phase (radians)")
    plt.title("Berry Phase Accumulation Along θ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("berry_plot.png")
    plt.close()

    other_berry_phases = compute_berry_phase_other(eigenvectors, theta_vals)
    print("Other Berry phases:", other_berry_phases)
    """