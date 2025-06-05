#drange is 0.001 to 0.01, 20, endpoint true
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from gabor_bph import compute_berry_phase
# Define parameter ranges
d_vals = np.linspace(0.001, 0.01, 20, endpoint=True)
theta_vals = np.linspace(0, 2*np.pi, 100, endpoint=True)  # Adjust N_theta to match your data
n_states = 4  # Adjust based on your system

def load_tau_matrices(base_dir):
    """Load tau matrices and theta values for all d values"""
    tau_all = []
    theta_all = []
    
    for d in d_vals:
        # Format the directory path to match the actual directory names
        dir_name = f"d_{d:.10f}".rstrip('0').rstrip('.')  # Remove trailing zeros and dot
        dir_path = os.path.join(base_dir, "lefutasok", dir_name, "npy")
        tau_path = os.path.join(dir_path, "tau.npy")
        theta_path = os.path.join(dir_path, "theta_vals.npy")
        
        if os.path.exists(tau_path) and os.path.exists(theta_path):
            # Load tau and theta values
            tau = np.load(tau_path)
            theta = np.load(theta_path)
            tau_all.append(tau)
            theta_all.append(theta)
        else:
            print(f"Warning: Missing files in {dir_path}")
            # Append None for missing data
            tau_all.append(None)
            theta_all.append(None)
    
    return tau_all, theta_all

def load_data(base_dir, d_vals):
    """Load all necessary data for all d values"""
    all_data = []
    
    for d in d_vals:
        dir_name = f"d_{d:.10f}".rstrip('0').rstrip('.')
        dir_path = os.path.join(base_dir, dir_name, "npy")
        data = {
            'd': d,
            'tau': None,
            'eigvecs': None,
            'eigvals': None,
            'theta_vals': None
        }
        
        try:
            # Load each file if it exists
            for key in ['tau', 'eigvecs', 'eigvals', 'theta_vals']:
                file_path = os.path.join(dir_path, f"{key}.npy")
                if os.path.exists(file_path):
                    data[key] = np.load(file_path)
                else:
                    print(f"Warning: {file_path} not found")
            
            # Only append if we have the essential data
            if data['eigvecs'] is not None and data['theta_vals'] is not None:
                all_data.append(data)
                print(f"Loaded data for d = {d}")
            else:
                print(f"Warning: Missing essential data for d = {d}")
                
        except Exception as e:
            print(f"Error loading data for d = {d}: {str(e)}")
            continue
            
    print(f"Successfully loaded data for {len(all_data)}/{len(d_vals)} d values")
    return [d for d in all_data if d['tau'] is not None]

def compute_berry_curvature_og(all_data, state_idx=0):
    """Compute Berry curvature using the overlap method"""
    # Get common theta values (should be the same for all d)
    theta_vals = all_data[0]['theta_vals']
    d_vals = [d['d'] for d in all_data]
    
    N_theta = len(theta_vals)
    N_d = len(d_vals)
    
    # Print shapes for debugging
    print(f"Shape of eigvecs: {all_data[0]['eigvecs'].shape}")
    print(f"Number of theta points: {N_theta}")
    print(f"Number of d points: {N_d}")
    print(f"Number of bands: {all_data[0]['eigvecs'].shape[1]}")
    
    # Initialize Berry curvature
    Omega = np.zeros((N_theta, N_d))
    
    # For each point in the grid
    for i in range(N_theta):
        for j in range(N_d):
            if all_data[j]['eigvecs'] is None:
                continue
                
            # Get indices of neighboring points with periodic boundary conditions
            ip = (i + 1) % N_theta
            im = (i - 1) % N_theta
            jp = min(j + 1, N_d - 1)
            jm = max(j - 1, 0)
            
            # Get wavefunctions - FIXED DIMENSION ORDER
            # eigvecs shape is (n_theta, n_bands, n_components)
            eigvecs = all_data[j]['eigvecs']
            psi = eigvecs[i, state_idx, :]  # |ψ(θ_i, d_j)⟩
            psi_theta_p = eigvecs[ip, state_idx, :]  # |ψ(θ_i+1, d_j)⟩
            psi_theta_m = eigvecs[im, state_idx, :]  # |ψ(θ_i-1, d_j)⟩
            
            # d-derivative (handle boundaries carefully)
            if j == 0:
                eigvecs_p = all_data[jp]['eigvecs']
                if eigvecs_p is not None:
                    psi_d_p = eigvecs_p[i, state_idx, :]  # |ψ(θ_i, d_j+1)⟩
                    d_psi_d = (psi_d_p - psi) / (d_vals[jp] - d_vals[j])
                else:
                    d_psi_d = 0
            elif j == N_d - 1:
                eigvecs_m = all_data[jm]['eigvecs']
                if eigvecs_m is not None:
                    psi_d_m = eigvecs_m[i, state_idx, :]  # |ψ(θ_i, d_j-1)⟩
                    d_psi_d = (psi - psi_d_m) / (d_vals[j] - d_vals[jm])
                else:
                    d_psi_d = 0
            else:
                eigvecs_p = all_data[jp]['eigvecs']
                eigvecs_m = all_data[jm]['eigvecs']
                if eigvecs_p is not None and eigvecs_m is not None:
                    psi_d_p = eigvecs_p[i, state_idx, :]
                    psi_d_m = eigvecs_m[i, state_idx, :]
                    d_psi_d = (psi_d_p - psi_d_m) / (d_vals[jp] - d_vals[jm])
                else:
                    d_psi_d = 0
            
            # Compute A_θ and A_d
            dtheta = theta_vals[ip] - theta_vals[im] if ip != im else 2 * np.pi / N_theta
            A_theta = 1j * np.vdot(psi, (psi_theta_p - psi_theta_m) / dtheta)
            A_d = 1j * np.vdot(psi, d_psi_d) if isinstance(d_psi_d, np.ndarray) else 0
            
            # Compute derivatives using central differences
            if i == 0 or i == N_theta - 1:
                # Forward/backward differences at boundaries
                dA_dtheta = (np.vdot(psi_theta_p, d_psi_d) - np.vdot(psi, d_psi_d)) / (theta_vals[ip] - theta_vals[i])
            else:
                dA_dtheta = (np.vdot(psi_theta_p, d_psi_d) - np.vdot(psi_theta_m, d_psi_d)) / (theta_vals[ip] - theta_vals[im])
            
            if j == 0 or j == N_d - 1:
                dA_dd = 0  # Skip boundary terms for d-direction
            else:
                dA_dd = (np.vdot(psi_d_p, d_psi_d) - np.vdot(psi_d_m, d_psi_d)) / (d_vals[jp] - d_vals[jm]) if jp != jm else 0
            
            # Berry curvature
            Omega[i,j] = (dA_dd - dA_dtheta).imag  # Should be real
    
    return theta_vals, d_vals, Omega

def compute_berry_curvature(all_data, state_idx=0):
    """Compute Berry curvature using the overlap method with proper phase handling"""
    # Get common theta values (should be the same for all d)
    theta_vals = all_data[0]['theta_vals']
    d_vals = [d['d'] for d in all_data]
    
    N_theta = len(theta_vals)
    N_d = len(d_vals)
    
    print(f"Shape of eigvecs: {all_data[0]['eigvecs'].shape}")
    print(f"Number of theta points: {N_theta}")
    print(f"Number of d points: {N_d}")
    print(f"Number of bands: {all_data[0]['eigvecs'].shape[1]}")
    
    # Initialize Berry curvature
    Omega = np.zeros((N_theta, N_d))
    
    # For each d value, compute the Berry connection in theta direction
    A_theta = np.zeros((N_theta, N_d), dtype=complex)
    for j in range(N_d):
        if all_data[j]['eigvecs'] is not None:
            # Extract eigenvectors for this d value (shape: n_theta, n_bands, n_components)
            eigvecs = all_data[j]['eigvecs']
            # Transpose to (n_bands, n_theta, n_components) for compute_berry_phase
            eigvecs_reshaped = np.transpose(eigvecs, (1, 0, 2))
            # Compute tau and gamma for this d value
            tau, _ = compute_berry_phase(eigvecs_reshaped, theta_vals, output_dir=None)
            # Get the diagonal elements (n,n) for each theta
            A_theta[:, j] = np.diagonal(tau[:, :, :], axis1=0, axis2=1)[state_idx, :]
    
    # For each theta value, compute the Berry connection in d direction
    A_d = np.zeros((N_theta, N_d), dtype=complex)
    for i in range(N_theta):
        # Collect eigenvectors for this theta across all d values
        eigvecs_theta = []
        valid_d_indices = []
        for j in range(N_d):
            if all_data[j]['eigvecs'] is not None:
                eigvecs_theta.append(all_data[j]['eigvecs'][i, :, :])  # shape: (n_bands, n_components)
                valid_d_indices.append(j)
        
        if len(valid_d_indices) < 2:
            continue
            
        # Stack to get shape (n_d, n_bands, n_components)
        eigvecs_theta = np.array(eigvecs_theta)
        # Transpose to (n_bands, n_d, n_components) for compute_berry_phase
        eigvecs_reshaped = np.transpose(eigvecs_theta, (1, 0, 2))
        # Compute tau and gamma for this theta value
        d_vals_subset = [d_vals[j] for j in valid_d_indices]
        tau, _ = compute_berry_phase(eigvecs_reshaped, d_vals_subset, output_dir=None)
        # Get the diagonal elements (n,n) for each d
        A_d[i, valid_d_indices] = np.diagonal(tau[:, :, :], axis1=0, axis2=1)[state_idx, :]
    
    # Compute derivatives to get the Berry curvature
    for i in range(1, N_theta-1):
        for j in range(1, N_d-1):
            if all_data[j]['eigvecs'] is None:
                continue
                
            # Central differences for derivatives
            dA_d_dtheta = (A_d[i+1, j] - A_d[i-1, j]) / (theta_vals[i+1] - theta_vals[i-1])
            dA_theta_dd = (A_theta[i, j+1] - A_theta[i, j-1]) / (d_vals[j+1] - d_vals[j-1])
            
            # Berry curvature
            Omega[i, j] = (dA_d_dtheta - dA_theta_dd).imag  # Should be real
    
    # Print some debug info
    print(f"A_theta min: {np.min(A_theta.real):.3e}, max: {np.max(A_theta.real):.3e}")
    print(f"A_d min: {np.min(A_d.real):.3e}, max: {np.max(A_d.real):.3e}")
    print(f"Omega min: {np.min(Omega):.3e}, max: {np.max(Omega):.3e}")
    
    return theta_vals, d_vals, Omega

def compute_berry_curvature_hamiltonian(all_data, target_d=0.01, state_idx=0):
    """
    Compute Berry curvature using the Hamiltonian approach.
    
    The Berry curvature for the nth band is given by:
    Ω_n = -2 Im[Σ_{m≠n} ( <∂_θ n|m><m|∂_d n> ) / (E_n - E_m)^2 ]
    
    Parameters:
    - all_data: List of dictionaries containing eigvecs, eigvals, etc.
    - target_d: The d value to use (default: 0.01)
    - state_idx: Which band to compute the curvature for (0-based index)
    """
    # Find the data for the target d value
    data = None
    for d in all_data:
        if np.isclose(d['d'], target_d, atol=1e-8):
            data = d
            break
    
    if data is None:
        raise ValueError(f"Could not find data for d = {target_d}")
    
    # Get the data for this d value
    eigvecs = data['eigvecs']  # shape: (n_theta, n_bands, n_components)
    eigvals = data['eigvals']  # shape: (n_theta, n_bands)
    theta_vals = data['theta_vals']
    n_theta = len(theta_vals)
    n_bands = eigvecs.shape[1]
    
    print(f"Computing Berry curvature for d = {target_d} using Hamiltonian approach")
    print(f"Shape of eigvecs: {eigvecs.shape}")
    print(f"Number of theta points: {n_theta}")
    print(f"Number of bands: {n_bands}")
    
    # Initialize Berry curvature
    Omega = np.zeros(n_theta, dtype=float)
    
    # Compute numerical derivatives using central differences
    for i in range(1, n_theta-1):
        dtheta = theta_vals[i+1] - theta_vals[i-1]
        
        # Get eigenstates at neighboring points
        psi_n = eigvecs[i, state_idx, :]  # |n(θ_i)>
        psi_n_prev = eigvecs[i-1, state_idx, :]  # |n(θ_{i-1})>
        psi_n_next = eigvecs[i+1, state_idx, :]  # |n(θ_{i+1})>
        
        # Normalize states (just to be safe)
        psi_n = psi_n / np.linalg.norm(psi_n)
        psi_n_prev = psi_n_prev / np.linalg.norm(psi_n_prev)
        psi_n_next = psi_n_next / np.linalg.norm(psi_n_next)
        
        # Compute |∂_θ n> using central difference
        dpsi_dtheta = (psi_n_next - psi_n_prev) / (2 * dtheta)
        
        # For each other band m ≠ n
        for m in range(n_bands):
            if m == state_idx:
                continue
                
            # Get the m-th eigenstate and normalize
            psi_m = eigvecs[i, m, :]  # |m(θ_i)>
            psi_m = psi_m / np.linalg.norm(psi_m)
            
            # Compute <m|∂_θ n>
            m_dtheta_n = np.vdot(psi_m, dpsi_dtheta)
            
            # Compute energy difference
            E_n = eigvals[i, state_idx]
            E_m = eigvals[i, m]
            delta_E = E_n - E_m
            
            if abs(delta_E) < 1e-10:  # Avoid division by zero
                continue
                
            # Add contribution to Berry curvature
            Omega[i] += -2 * np.imag(m_dtheta_n * np.conj(m_dtheta_n)) / (delta_E ** 2)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(theta_vals, Omega, 'r-', label='Ω(θ)')
    plt.xlabel('θ')
    plt.ylabel('Berry curvature Ω')
    plt.title(f'Berry curvature for d = {target_d} (band {state_idx})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', f'berry_curvature_d_{target_d}_band_{state_idx}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved as {plot_path}")
    
    return theta_vals, Omega

def compute_berry_curvature_dense_theta(all_data, target_d=0.01, state_idx=0):
    """
    Compute Berry curvature using dense theta sampling for a fixed d value.
    
    Parameters:
    - all_data: List of dictionaries containing eigvecs and theta_vals for each d
    - target_d: The d value to use (default: 0.01)
    - state_idx: Which band to compute the curvature for
    """
    # Find the index of the target d value
    d_vals = [d['d'] for d in all_data]
    d_idx = None
    for i, d in enumerate(d_vals):
        if np.isclose(d, target_d, atol=1e-8):
            d_idx = i
            break
    
    if d_idx is None:
        raise ValueError(f"Could not find data for d = {target_d}")
    
    # Get the data for this d value
    data = all_data[d_idx]
    eigvecs = data['eigvecs']  # shape: (n_theta, n_bands, n_components)
    theta_vals = data['theta_vals']
    n_theta = len(theta_vals)
    n_bands = eigvecs.shape[1]
    
    print(f"Computing Berry curvature for d = {target_d}")
    print(f"Shape of eigvecs: {eigvecs.shape}")
    print(f"Number of theta points: {n_theta}")
    print(f"Number of bands: {n_bands}")
    
    # Initialize Berry connection and curvature
    A_theta = np.zeros(n_theta, dtype=complex)
    Omega = np.zeros(n_theta)
    
    # Compute Berry connection A_theta
    for i in range(n_theta):
        # Get the current and next eigenvector
        psi_n = eigvecs[i, state_idx, :]
        psi_n = psi_n / np.linalg.norm(psi_n)  # Normalize
        
        # Get the next eigenvector with periodic boundary conditions
        ip1 = (i + 1) % n_theta
        psi_np1 = eigvecs[ip1, state_idx, :]
        psi_np1 = psi_np1 / np.linalg.norm(psi_np1)  # Normalize
        
        # Compute the Berry connection
        A_theta[i] = 1j * np.vdot(psi_n, (psi_np1 - psi_n) / (theta_vals[ip1] - theta_vals[i]))
    
    # Compute Berry curvature as the derivative of A_theta
    Omega = np.gradient(A_theta, theta_vals, edge_order=2)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Plot A_theta
    plt.subplot(1, 2, 1)
    plt.plot(theta_vals, A_theta.real, label='Re(A_θ)')
    plt.plot(theta_vals, A_theta.imag, '--', label='Im(A_θ)')
    plt.xlabel('θ')
    plt.ylabel('Berry connection A_θ')
    plt.title('Berry connection in θ direction')
    plt.legend()
    plt.grid(True)
    
    # Plot Berry curvature
    plt.subplot(1, 2, 2)
    plt.plot(theta_vals, Omega, 'r-', label='Ω')
    plt.xlabel('θ')
    plt.ylabel('Berry curvature Ω')
    plt.title('Berry curvature')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'berry_curvature_d_{target_d}.png')
    plt.close()
    
    print(f"Plots saved as berry_curvature_d_{target_d}.png")
    
    return theta_vals, A_theta, Omega


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(base_dir, "lefutasok")
    
    # Define d values to match your directory names
    d_vals = np.linspace(0.001, 0.01, 20)
    
    print("Loading data...")
    all_data = load_data(base_dir, d_vals)
    
    if not all_data:
        print("Error: No valid data found.")
        return
    
    # Compute Berry curvature using the Hamiltonian approach
    target_d = 0.01
    state_idx = 0  # Compute for the first band
    
    try:
        theta_vals, Omega = compute_berry_curvature_hamiltonian(
            all_data, target_d=target_d, state_idx=state_idx
        )
        
        # Compute the integrated Berry phase
        from scipy.integrate import trapezoid
        berry_phase = trapezoid(Omega, theta_vals)
        print(f"\nIntegrated Berry phase: {berry_phase:.6f}")
        print(f"Berry phase / π: {berry_phase/np.pi:.6f}π")
        
    except Exception as e:
        print(f"Error computing Berry curvature: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()