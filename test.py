import numpy as np
from berry_phase_analysis import calculate_berry_phase_with_berry_curvature_simplified_multiprocessing

def generate_berry_phase_hamiltonian(theta):
    """
    Generates a 2x2 Hamiltonian with a known Berry phase.
    This Hamiltonian describes a simple Dirac cone with a Berry phase of pi.

    Args:
        theta (float): The parameter theta.

    Returns:
        numpy.ndarray: The 2x2 Hamiltonian matrix.
    """
    kx = np.cos(theta)
    ky = np.sin(theta)

    hamiltonian = np.array([
        [0, kx - 1j * ky],
        [kx + 1j * ky, 0]
    ])
    return hamiltonian

def calculate_eigenvectors(theta_vals):
    """
    Calculates eigenvectors for the generated Hamiltonian at given theta values.

    Args:
        theta_vals (numpy.ndarray): Array of theta values.

    Returns:
        numpy.ndarray: Array of eigenvectors at each theta value.
    """
    eigenvectors = np.zeros((len(theta_vals), 2, 2), dtype=complex)
    for i, theta in enumerate(theta_vals):
        hamiltonian = generate_berry_phase_hamiltonian(theta)
        eigenvalues, vecs = np.linalg.eigh(hamiltonian)
        eigenvectors[i] = vecs
    return eigenvectors

# Example usage for testing:
import os

def test_berry_phase_calculation():
    """
    Tests the Berry phase calculation using the generated Hamiltonian.
    """
    theta_vals = np.linspace(0, 2 * np.pi, 101)  # Increased points for better accuracy

    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    eigenvectors = calculate_eigenvectors(theta_vals)
    berry_phases, accumulated_phases = calculate_berry_phase_with_berry_curvature_simplified_multiprocessing(theta_vals, eigenvectors, output_dir)

    print("Calculated Berry Phases:", berry_phases)
    # The berry phase of the lower band should be close to pi.
    print("Expected Berry Phase (lower band): pi")

    # Check for expected Berry phase.
    expected_phase = np.pi
    tolerance = 0.1  # Adjust tolerance as needed

    if abs(berry_phases[0] - expected_phase) < tolerance:
        print("Test Passed: Berry phase calculation is within tolerance.")
    else:
        print("Test Failed: Berry phase calculation is outside tolerance.")

    # You can also add more checks or analysis of the results here

if __name__ == "__main__":
    test_berry_phase_calculation()