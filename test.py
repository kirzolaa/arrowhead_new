import numpy as np
from scipy.linalg import eigh

def test_berry_hamiltonian(theta_vals):
    H_thetas = [np.array([
        [np.cos(theta), np.sin(theta)],
        [np.sin(theta), -np.cos(theta)]
    ]) for theta in theta_vals]
    return H_thetas

def calculate_berry_phase(Hamiltonians, theta_vals, eigenstate_index):
    """Calculates the Berry phase with phase fixing."""
    
    berry_phase = 0.0
    
    # Calculate eigenvectors for the first Hamiltonian
    eigenvalues_i, eigenvectors_i = eigh(Hamiltonians[0])
    ground_state_i = eigenvectors_i[:, eigenstate_index]
    
    for i in range(len(theta_vals) - 1):
        H_ip1 = Hamiltonians[i + 1]
        
        # Calculate eigenvectors
        eigenvalues_ip1, eigenvectors_ip1 = eigh(H_ip1)
        ground_state_ip1 = eigenvectors_ip1[:, eigenstate_index]
        
        # Phase fixing: Ensure continuity
        overlap = np.dot(ground_state_i.conj(), ground_state_ip1)
        ground_state_ip1 = ground_state_ip1 * np.exp(-1j * np.angle(overlap))
        
        # Calculate the Berry connection
        berry_connection = np.angle(np.dot(ground_state_i.conj(), ground_state_ip1))
        
        # Accumulate the Berry phase
        berry_phase += berry_connection
        
        # Update the current eigenvector
        ground_state_i = ground_state_ip1
        
    return berry_phase


def calculate_berry_phase_test(Hamiltonians, theta_vals, eigenstate_index):
    """Calculates the Berry phase for a given set of Hamiltonians and eigenstate."""
    
    berry_phase = 0.0
    
    # Calculate eigenvectors for the first Hamiltonian
    eigenvalues_i, eigenvectors_i = eigh(Hamiltonians[0])
    ground_state_i = eigenvectors_i[:, eigenstate_index]
    
    for i in range(len(theta_vals) - 1):
        H_ip1 = Hamiltonians[i + 1]
        
        # Calculate eigenvectors
        eigenvalues_ip1, eigenvectors_ip1 = eigh(H_ip1)
        ground_state_ip1 = eigenvectors_ip1[:, eigenstate_index]
        
        # Calculate the overlap
        overlap = np.dot(ground_state_i.conj(), ground_state_ip1)
        
        # Calculate the Berry connection
        berry_connection = np.angle(overlap)
        
        # Accumulate the Berry phase
        berry_phase += berry_connection
        
        # Update the current eigenvector
        ground_state_i = ground_state_ip1
        
    return berry_phase

if __name__ == '__main__':
    # Set up the theta values
    theta_vals = np.linspace(0, 2 * np.pi, 1000)
    
    # Calculate the Hamiltonians
    Hamiltonians = test_berry_hamiltonian(theta_vals)

    # Calculate the Berry phase for the first eigenstate (index 0)
    berry_phase_ground = calculate_berry_phase_test(Hamiltonians, theta_vals, 0)
    print(f"Berry phase (ground state): {berry_phase_ground}")

    # Calculate the Berry phase for the second eigenstate (index 1)
    berry_phase_excited = calculate_berry_phase_test(Hamiltonians, theta_vals, 1)
    print(f"Berry phase (excited state): {berry_phase_excited}")