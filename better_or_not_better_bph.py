import numpy as np

def compute_berry_phase(eigvectors_all, theta_vals):
    """
    Compute Berry phases γ_n for each eigenstate n along a closed path in R-space.

    Parameters:
    - eigvectors_all: ndarray of shape (N, M, M), eigenvectors for each R(θ)
    - R_thetas: ndarray of shape (N, 3), parameter-space path (not directly used in this version)
    - theta_vals: ndarray of shape (N,), parameter values along the path

    Returns:
    - tau: ndarray of shape (M, M, N), Berry connection for each eigenstate in radians
    - gamma: ndarray of shape (M, M, N), Berry phase for each eigenstate in radians
    """
    N, M, _ = eigvectors_all.shape
    
    tau = np.zeros((M, M, N), dtype=np.float64)
    gamma = np.zeros((M, M, N), dtype=np.float64)

    for n in range(M):
        for m in range(M):
            for i in range(N):
                # Handle boundary conditions for the finite difference
                # Inside compute_berry_phase
    # ...
    # N is num_points
    # theta_step = (theta_vals[-1] - theta_vals[0]) / (N - 1) # if endpoint=True
    # Effective N-1 is theta_vals[N-1] if endpoint=True
    # Effective 0 is theta_vals[0]

                if i == 0:
                    psi_prev = eigvectors_all[N - 1, :, n]  # Vector at theta_max (which is theta_0 if path is closed)
                                                            # OR eigvectors_all[N-2,:,n] if using N-1 points to define the distinct loop points (0 to N-2) and N-1 is same as 0
                                                            # Let's assume N points, theta_vals[N-1] is distinct from theta_vals[0] but psi(theta_vals[N-1]) is "before" psi(theta_vals[0])
                    psi_next = eigvectors_all[1, :, n]
                    # For central diff: (psi_next - psi_prev) / (theta_next - theta_prev)
                    # where theta_prev is theta_0 - step, theta_next is theta_0 + step
                    # So effectively (theta_vals[1] - (theta_vals[N-1] - PathLength) ) if N-1 is last point
                    # This depends on how N points define the loop.
                    # If theta_vals[0]...theta_vals[N-1] are N distinct points on the loop:
                    # grad at theta_0: (psi_1 - psi_{N-1}) / ( (theta_1-theta_0) + ( (theta_0 - theta_{N-1})_mod_PathLength ) )
                    # A simpler and often robust way for periodic central difference for point k:
                    # (psi[ (k+1)%N ] - psi[ (k-1+N)%N ]) / (2*theta_step)
                    # Your delta_theta definition for boundaries needs careful review based on your path definition.
                    # However, your current delta_theta (large, negative) makes grad_psi small at boundaries.
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                elif i == N - 1:
                    psi_prev = eigvectors_all[N - 2, :, n]
                    psi_next = eigvectors_all[0, :, n] # Vector at theta_0 (which is theta_N-1 + step if path is closed)
                    delta_theta_for_grad = 2 * (theta_vals[1] - theta_vals[0]) # Assuming constant step
                else:
                    psi_prev = eigvectors_all[i - 1, :, n]
                    psi_next = eigvectors_all[i + 1, :, n]
                    delta_theta_for_grad = theta_vals[i + 1] - theta_vals[i - 1]

                psi_curr = eigvectors_all[i, :, m]
                # Normalize for safety
                psi_prev = psi_prev / np.linalg.norm(psi_prev)
                psi_next = psi_next / np.linalg.norm(psi_next)
                psi_curr = psi_curr / np.linalg.norm(psi_curr)

                # Finite difference approximation of ∇_theta |ψ>
                delta_psi = psi_next - psi_prev
                grad_psi = delta_psi / (delta_theta_for_grad) # Corrected delta_theta

                # τ = ⟨ψ_i | ∇_theta | ψ_{i-1}⟩  (Corrected index for tau)
                tau[n, m, i] = np.vdot(psi_curr, grad_psi)
                tau[n, m, i] /= np.linalg.norm(tau[n, m, i])
                # · d_theta to integrate.  Accumulate the *imaginary* part for Berry phase.
                if i == 0:
                   gamma[n, m, i] = 0.0
                else:
                    delta_theta_integrate = theta_vals[i] - theta_vals[i-1]
                    # Add the area of the segment from theta_vals[i-1] to theta_vals[i]
                    # Option 1: Using tau at the end of the interval (simplest Riemann sum)
                    gamma[n, m, i] = gamma[n, m, i-1] + tau[n, m, i] * delta_theta_integrate

                    # Option 2: Using trapezoidal rule (generally more accurate)
                    # gamma[n, m, i] = gamma[n, m, i-1] + (tau[n, m, i] + tau[n, m, i-1]) / 2.0 * delta_theta_integrate
    
    return tau, gamma

if __name__ == '__main__':
    #load the eigvecs.npy file:
    eigvecs = np.load("/home/zoltan/arrowhead_new/berry_phase_with_tau_and_gamma/npy/eigenvectors.npy")
    theta_min = 0.0
    theta_max = 2.0 * np.pi
    num_points = 50000
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)
    
    tau, gamma = compute_berry_phase(eigvecs, theta_vals)
    #print("Tau:", tau)
    print("Gamma[:,:,-1]:\n", gamma[:,:,-1]) #print the last gamma matrix

    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            print(f"Gamma[{i},{j}]: {gamma[i,j,-1]}")
            print(f"Tau[{i},{j}]: {tau[i,j,-1]}")