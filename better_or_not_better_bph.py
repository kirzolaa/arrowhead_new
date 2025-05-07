import numpy as np
import matplotlib.pyplot as plt
import os


def plot_matrix_elements(tau, gamma, theta_vals, output_dir):
    """
    Plot the evolution of specific matrix elements (01, 12, 13) for both tau and gamma matrices.
    
    Parameters:
    - tau: 3D array of shape (M, M, N) containing tau values over theta
    - gamma: 3D array of shape (M, M, N) containing gamma values over theta
    - theta_vals: 1D array of theta values
    - output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # Elements to plot
    elements = [(0, 1), (1, 2), (1, 3)]
    
    # Plot real and imaginary parts of tau
    plt.subplot(2, 1, 1)
    for i, j in elements:
        plt.plot(theta_vals, np.real(tau[i, j, :]), 
                label=f'Re(τ_{i}{j})', linestyle='-')
        plt.plot(theta_vals, np.imag(tau[i, j, :]), 
                label=f'Im(τ_{i}{j})', linestyle='--')
    plt.xlabel('θ')
    plt.ylabel('τ')
    plt.title('Evolution of τ matrix elements')
    plt.legend()
    plt.grid(True)
    
    # Plot real and imaginary parts of gamma
    plt.subplot(2, 1, 2)
    for i, j in elements:
        plt.plot(theta_vals, np.real(gamma[i, j, :]), 
                label=f'Re(γ_{i}{j})', linestyle='-')
        plt.plot(theta_vals, np.imag(gamma[i, j, :]), 
                label=f'Im(γ_{i}{j})', linestyle='--')
    plt.xlabel('θ')
    plt.ylabel('γ')
    plt.title('Evolution of γ matrix elements')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/matrix_elements_evolution.png')
    plt.close()
    
    # Create separate plots for each element
    for i, j in elements:
        plt.figure(figsize=(10, 6))
        plt.plot(theta_vals, np.real(tau[i, j, :]), label=f'Re(τ_{i}{j})')
        plt.plot(theta_vals, np.imag(tau[i, j, :]), label=f'Im(τ_{i}{j})')
        plt.plot(theta_vals, np.real(gamma[i, j, :]), '--', label=f'Re(γ_{i}{j})')
        plt.plot(theta_vals, np.imag(gamma[i, j, :]), '--', label=f'Im(γ_{i}{j})')
        plt.xlabel('θ')
        plt.ylabel('Value')
        plt.title(f'Evolution of τ and γ_{i}{j} elements')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/element_{i}{j}_evolution.png')
        plt.close()

def format_matrix(matrix, title=None):
    """Format a matrix with box drawing characters"""
    n, m = matrix.shape
    max_len = max(len(f"{x:.4f}") for row in matrix for x in row)
    
    # Calculate width based on max number length and matrix dimensions
    width = (max_len + 3) * m + 1
    
    lines = []
    if title:
        lines.append(f"    |{title:^{width-2}}|")
    
    # Top border
    lines.append("    |‾" + "‾" * (width-2) + "‾|")
    
    # Matrix rows
    for i in range(n):
        row = "    |  "
        for j in range(m):
            if i == j:
                # Diagonal elements (γ_nn)
                row += f"γ_{i}{i} = {matrix[i,j]:.4f}"
            else:
                # Off-diagonal elements (γ_nm)
                row += f"γ_{i}{j} = {matrix[i,j]:.4f}"
            if j < m - 1:
                row += "  "
            else:
                row += "  |"
        lines.append(row)
    
    # Bottom border
    lines.append("    |_" + "_" * (width-2) + "_|")
    
    return "\n".join(lines)



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
    #create a report on the gamma matrix
    with open("gamma_report.txt", "w") as f:
        f.write("Gamma matrix report:\n===========================================\n")
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                f.write(f"Gamma[{i},{j}]: {gamma[i,j,-1]}\n")
                f.write(f"Tau[{i},{j}]: {tau[i,j,-1]}\n")
            f.write("\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(gamma[:,:,-1], "Berry Phase Matrix"))
        f.write("\n\n")
        f.write("===========================================\n")
        f.write("\n")
        f.write(format_matrix(tau[:,:,-1], "Berry Connection Matrix"))
        f.write("\n\n")
        f.write("===========================================\n")

    #print the gamma matrix
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            print(f"Gamma[{i},{j}]: {gamma[i,j,-1]}")
            print(f"Tau[{i},{j}]: {tau[i,j,-1]}")
    
    #create a directory for the output
    output_dir = 'berry_phase_corrected'
    os.makedirs(output_dir, exist_ok=True)
    
    #create a directory for the npy files
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    #create a directory for the plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    #save the tau and gamma matrices
    np.save(os.path.join(npy_dir, 'tau.npy'), tau)
    np.save(os.path.join(npy_dir, 'gamma.npy'), gamma)

    #save the eigvecs
    np.save(os.path.join(npy_dir, 'eigvecs.npy'), eigvecs)

    #save the theta_vals
    np.save(os.path.join(npy_dir, 'theta_vals.npy'), theta_vals)


    #plot the gamma and tau matrices
    plot_matrix_elements(tau, gamma, theta_vals, output_dir)
    