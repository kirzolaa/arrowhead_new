import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'generalized'))
from vector_utils import create_perfect_orthogonal_vectors, multiprocessing_create_perfect_orthogonal_circle, create_perfect_orthogonal_circle
from main import *
print("Successfully imported create_perfect_orthogonal_vectors from arrowhead/generalized package.")
import datetime
from scipy.constants import hbar


class Hamiltonian:
    """
    Hamiltonian class for a quantum system with a potentials:

    V(x) = aVx * x^2, and
    Va(x) = aVa * (x - x_shift)^2 + c

    Creates a 4x4 matrix with an arrowhead structure, where
    
    |‾                                                  ‾|
    |    hbar*omega + Σ Vx(i)  tdm01   tdm02    tdm03    |
    |    tdm01                 V_e(i)  0        0        |
    |    tdm02                 0       V_e(i+1) 0        |
    |    tdm03                 0       0        V_e(i+2) |
    |_                                                  _|
    
    where Σ Vx(i) represents the sum of Vx values from i=0 to N, and
    V_e(i) represents the potential Σ Vx(i) + Va(i) - Vx(i) at angle i.
    
    """
    def __init__(self, omega, aVx, aVa, x_shift, c_const, R_0, d, theta_range):
        """
        Initialize the Hamiltonian with parameters.
        
        Parameters:
        omega (float): Angular frequency of the oscillator
        aVx (float): Parameter of the potential Vx
        aVa (float): Parameter of the potential Va
        x_shift (float): Shift in the potential Va
        c_const (float): Constant in the potential
        R_0 (float): Radius of the circle
        d (float): Radius of the circle
        theta_range (list): List of angles for which the Hamiltonian is calculated
        """
        self.omega = omega
        self.aVx = aVx
        self.aVa = aVa
        self.x_shift = x_shift
        self.c = c_const #constant in the potential
        self.R_0 = R_0
        self.d = d
        self.theta_range = theta_range

    def R_theta(self, theta):
        """
        Create a vector that traces a perfect circle orthogonal to the x=y=z line using the
        create_perfect_orthogonal_vectors function from the Arrowhead/generalized package.
        
        Parameters:
        d (float): The radius of the circle
        theta (float): The angle parameter
        
        Returns:
        numpy.ndarray: A 3D vector orthogonal to the x=y=z line
        """
        # Generate the perfect orthogonal vector
        return create_perfect_orthogonal_vectors(self.R_0, self.d, theta)

    def V_x(self, R_theta_val):
        """
        Returns the potential Vx for a given R_theta_val
        
        Parameters:
        R_theta_val (numpy.ndarray): A 3D vector orthogonal to the x=y=z line
        
        Returns:
        float: The potential Vx
        """
        return self.aVx * (R_theta_val**2)

    def V_a(self, R_theta_val):
        """
        Returns the potential Va for a given R_theta_val
        
        Parameters:
        R_theta_val (numpy.ndarray): A 3D vector orthogonal to the x=y=z line
        
        Returns:
        float: The potential Va
        """
        return self.aVa * ((R_theta_val - self.x_shift)**2 + self.c)

    #define Vx and Va functions, but input _arrays
    def Vx(self, R_theta_1d_array):
        """
        Returns the potential array Vx for an array of R_theta_val
        
        Parameters:
        R_theta_1d_array (numpy.ndarray): An array of 3D vectors orthogonal to the x=y=z line
        
        Returns:
        numpy.ndarray: An array of the components of the potential Vx
        """
        return [self.aVx * (R_theta_1d_array[i]**2) for i in range(len(R_theta_1d_array))]
    
    def Va(self, R_theta_1d_array):
        """
        Returns the potential array Va for an array of R_theta_val
        
        Parameters:
        R_theta_1d_array (numpy.ndarray): An array of 3D vectors orthogonal to the x=y=z line
        
        Returns:
        numpy.ndarray: An array of the components of the potential Va
        """
        return [self.aVa * ((R_theta_1d_array[i] - self.x_shift)**2 + self.c) for i in range(len(R_theta_1d_array))]
        
    def tdm(self, R, R0):
        """
        Returns the approximated transitonal dipole moment for a given R and R0
        
        Parameters:
        R (float): The radius of the circle
        R0 (float): The radius of the circle
        
        Returns:
        float: The approximated transitonal dipole moment
        """
        return 0.2 + (R - R0) / 100

    def construct_matrix(self, theta):
        """
        Constructs the Hamiltonian matrix for a given angle theta
        
        Parameters:
        theta (float): The angle parameter
        
        Returns:
        numpy.ndarray: The Hamiltonian matrix
        """
        R_theta_val = self.R_theta(theta)
        Vx = [self.aVx * (R_theta_val[i] ** 2) for i in range(3)]
        Va = [self.aVa * (R_theta_val[i] ** 2) for i in range(3)]
        
        H = np.zeros((4, 4), dtype=complex)
        sumVx = sum(Vx)
        H[0, 0] = hbar * self.omega + sumVx
        for i in range(1, len(H)):
            H[i, i] = H[0, 0] + Va[i-1] - Vx[i-1] #H11 = Vx1 + Vx2 + Vx3 + Va1 - Vx1
            
        for j in range(len(R_theta_val)):
            H[0, j+1] = H[j+1, 0] = self.tdm(R_theta_val[j], self.R_0[j])

        return H

    def H_thetas(self):
        """
        Returns the Hamiltonian matrix for all angles in a given theta_range
        
        Returns:
        list: A list of Hamiltonian matrices for each angle in the theta_range
        """
        return [self.construct_matrix(theta) for theta in self.theta_range]

    def R_thetas(self):
        """
        Returns the radius of the circle for all angles in a given theta_range
        
        Returns:
        list: A list of radius values for each angle in the theta_range
        """
        return [self.R_theta(theta) for theta in self.theta_range]
    
    def Vx_theta_vals(self, R_thetas):
        #return the potentials in theta_vals
        #get the R_thetas from the function above
        """
        Returns the potential Vx for all angles in a given theta_range
        
        Parameters:
        R_thetas (list or np.ndarray): A list or array of 3D parameter vectors
        
        Returns:
        list: A list of potential values for each angle in the theta_range
        """
        if R_thetas is None:
            R_thetas = np.array(self.R_thetas())
        return [self.Vx(R_theta) for R_theta in R_thetas]
    
    def Va_theta_vals(self, R_thetas):
        #return the potentials in theta_vals
        #get the R_thetas from the function above
        """
        Returns the potential Va for all angles in a given theta_range
        
        Parameters:
        R_thetas (list or np.ndarray): A list or array of 3D parameter vectors
        
        Returns:
        list: A list of potential values for each angle in the theta_range
        """
        if R_thetas is None: #can be called independently, too
            R_thetas = np.array(self.R_thetas())
        return [self.Va(R_theta) for R_theta in R_thetas]
        

class NewBerryPhaseCalculator:
    def __init__(self, hamiltonian, R_thetas, eigenstates, theta_range):
        """
        Initializes the new BerryPhaseCalculator with the Hamiltonian,
        parameter vectors R_thetas, corresponding eigenstates, and the
        range of the parameter theta.

        Args:
            hamiltonian: An instance of the Hamiltonian class.
            R_thetas (list or np.ndarray): A list or array of 3D parameter vectors.
            eigenstates (np.ndarray): An array of eigenstates corresponding to R_thetas
                                       (shape: (num_points, num_dimensions, num_states)).
            theta_range (np.ndarray): An array of theta values used to generate R_thetas.
        """
        self.hamiltonian = hamiltonian
        self.R_thetas = R_thetas
        self.eigenstates = eigenstates
        self.theta_range = theta_range

    def smooth_gauge(self, previous_eigenstates, current_eigenstates): #aka self, current, next eigenstates
        """
        Smooth the eigenstates at the next point.
        In a good implementation, the overlap should be close to 1 or minus 1.
        Also, the real implementation we don't need gauge fixing,
        since the calculations are gauge-invariant.
        """
        num_states = current_eigenstates.shape[1]
        smoothed_eigenstates = np.copy(current_eigenstates)
        for n in range(num_states):
            overlap = np.vdot(previous_eigenstates[:, n], current_eigenstates[:, n])
            phase_factor = np.exp(-1j * np.angle(overlap))
            smoothed_eigenstates[:, n] = current_eigenstates[:, n] * phase_factor
        return smoothed_eigenstates

    def calculate_berry_connection(self):
        """
        Calculate Berry connection A_n(R) ≈ <n(R_i)| i (|n(R_{i+1})> - |n(R_i)>) / (R_{i+1} - R_i)
        Here, the division by a vector should be interpreted element-wise for each component of dR.
        The result A_R will be a matrix where A_R[i, n, j] is the j-th component of the Berry connection
        for the n-th state at the i-th point.
        """
        num_points = len(self.R_thetas)
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        A_R = np.zeros((num_points - 1, num_states, 3), dtype=complex)  # Berry connection is a 3D vector

        for i in range(num_points - 1):
            # Smooth the eigenstates at the next point
            smoothed_eigenstates = self.smooth_gauge(self.eigenstates[i], self.eigenstates[(i + 1) % num_points])
            dR = self.R_thetas[(i + 1) % num_points] - self.R_thetas[i] #wrapping around the circle
            for n in range(num_states):
                v = self.eigenstates[i][:, n]
                next_v = smoothed_eigenstates[:, n]
                dv = next_v - v

                # Element-wise division by the components of dR
                with np.errstate(divide='ignore', invalid='ignore'):
                    dvdR = np.where(dR != 0, dv[:, np.newaxis] / dR, 0) # Shape (4, 3)

                # Project onto the current state
                A_R[i, n] = np.dot(np.conj(v).T, 1j * dvdR)

        return A_R

    def calculate_berry_phase(self):
        """
        Calculate Berry phase γ_n = ∫ A_n(R_theta) dR_theta
        """
        A_R = self.calculate_berry_connection()
        # Calculate berry phase for each state by integrating the Berry connection
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        berry_phases = np.zeros(num_states, dtype=float) #berry phase is real

        for n in range(num_states):
            phase = 0.0
            for i in range(len(A_R)):
                dR = self.R_thetas[i + 1] - self.R_thetas[i]
                phase += np.dot(A_R[i, n], dR) # Take full complex Berry connection for dot product
            berry_phases[n] = phase

        return berry_phases

    def calculate_berry_connection_theta_derivative(self):
        """
        Calculates the Berry connection in the basis of the parameter theta
        by approximating the derivative of the eigenstates with respect to theta
        using a central difference scheme. It also calculates the derivative
        of the parameter vector R with respect to theta.

        Returns:
            tuple: A tuple containing:
                - A_theta (np.ndarray): A complex array of shape (num_points, num_states)
                  where A_theta[i, n] is the Berry connection <n(theta_i)|i d/dtheta |n(theta_i)>.
                - dR_dtheta (np.ndarray): A float array of shape (num_points, 3)
                  where dR_dtheta[i] is the approximate derivative of the
                  parameter vector R with respect to theta at theta_i.
        """
        num_points = len(self.R_thetas)
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        A_theta = np.zeros((num_points, num_states), dtype=complex)
        dR_dtheta = np.zeros((num_points, 3), dtype=float)

        for i in range(num_points):
            # Use central difference for theta, handling periodic boundary conditions
            theta_plus = self.theta_range[(i + 1) % num_points]
            theta_minus = self.theta_range[(i - 1 + num_points) % num_points]
            d_theta = theta_plus - theta_minus

            # Calculate the approximate derivative of the parameter vector R with respect to theta
            dR_dtheta[i] = (self.R_thetas[(i + 1) % num_points] - self.R_thetas[(i - 1 + num_points) % num_points]) / d_theta

            for n in range(num_states):
                # Get the eigenstates at theta_{i+1} and theta_{i-1}
                v_plus = self.eigenstates[(i + 1) % num_points][:, n]
                v_minus = self.eigenstates[(i - 1 + num_points) % num_points][:, n]

                # Approximate the derivative of the n-th eigenstate with respect to theta using central difference
                dv_dtheta = (v_plus - v_minus) / d_theta

                # Calculate the Berry connection A_theta = <n(theta_i)|i d/dtheta |n(theta_i)>
                A_theta[i, n] = np.vdot((self.eigenstates[i][:, n]).T, 1j * dv_dtheta)

        return A_theta, dR_dtheta

    def calculate_berry_curvature_from_connection(self, A_theta, dR_dtheta):
        """
        Calculates the Berry curvature from the Berry connection.

        This is a helper function called by `calculate_berry_curvature`.
        """

        num_points = len(self.R_thetas)
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        berry_curvature = np.zeros((num_points, num_states))
        for i in range(num_points):
            for n in range(num_states):
                # Approximate dA/dR using dA/dtheta and dtheta/dR
                # We have dR/dtheta, so we approximate dtheta/dR as the inverse.
                # Be careful about dividing by zero!  If |dR/dtheta| is small,
                # the curvature may be ill-defined.  A more robust approach
                # might involve a pseudo-inverse or regularization.
                norm_dR_dtheta = np.linalg.norm(dR_dtheta[i])
                if norm_dR_dtheta > 1e-12:  # Avoid division by very small number
                    dA_dR = np.gradient(A_theta[:, n], self.theta_range)[i] / norm_dR_dtheta
                    berry_curvature[i, n] = dA_dR
                else:
                    berry_curvature[i, n] = 0.0  # Or a more sophisticated handling

        return berry_curvature

    def calculate_berry_curvature(self):
        """
        Calculates the Berry curvature.

        The Berry curvature is defined as the curl of the Berry connection.  Since
        we are working with a 1D parameter space (theta), the "curl" simplifies.
        In this case, we approximate the derivative of the Berry connection
        with respect to the parameter R.  Since we have A_theta (derivative with
        respect to theta), we need to relate d/dR to d/dtheta.

        Returns:
            np.ndarray: The calculated Berry curvature.
        """

        A_theta, dR_dtheta = self.calculate_berry_connection_theta_derivative()
        berry_curvature = self.calculate_berry_curvature_from_connection(A_theta, dR_dtheta)
        return berry_curvature

    def calculate_berry_phase_theta_derivative(self):
        """
        Calculates the Berry phase by integrating the Berry connection A_theta
        over the parameter theta. It attempts to relate A_theta to the Berry
        connection in the R space by projecting the derivative of R with
        respect to theta onto the approximate tangent of the path in R space.

        Returns:
            np.ndarray: A float array of shape (num_states) where each element
                         is the Berry phase for the corresponding eigenstate.
        """
        # Calculate the Berry connection in theta basis and dR/dtheta
        A_theta, dR_dtheta = self.calculate_berry_connection_theta_derivative()
        num_points = len(self.R_thetas)
        num_states = self.eigenstates.shape[2] if len(self.eigenstates.shape) > 2 else self.eigenstates.shape[1]
        berry_phases = np.zeros(num_states, dtype=float) #berry phase is real

        for n in range(num_states):
            phase = 0.0
            # Integrate over the discretized theta range
            for i in range(num_points - 1):
                # The change in the parameter vector R between consecutive points
                dR = self.R_thetas[(i + 1) % num_points] - self.R_thetas[i]
                # The Berry connection component along d/dtheta
                tangent_component = A_theta[i, n]
                # Approximate the tangent direction of the path in R space
                norm_dR = np.linalg.norm(dR)
                tangent_direction = dR / norm_dR if norm_dR > 1e-12 else np.zeros(3)
                # Project dR_dtheta onto the approximate tangent direction
                tangent_dR_dtheta = np.dot(dR_dtheta[i], tangent_direction)
                # Approximate the integral of A . dR ≈ A_theta * (dtheta/dR) * dR ≈ A_theta * dtheta
                phase += tangent_component * (self.theta_range[i+1] - self.theta_range[i]) # Integral over dtheta
            # The Berry phase should be real for a closed path
            berry_phases[n] = np.real(phase)

        return berry_phases

if __name__ == "__main__":
    #let a be an aVx and an aVa parameter
    aVx = 1.0
    aVa = 5.0
    c_const = 1.0  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 1.0  # Shift in x direction
    d = 1.0  # Radius of the circle, use unit circle for bigger radius
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 5000
    R_0 = (0, 0, 0)
    # Generate the arrowhead matrix and Va, Vx
    theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

    # Save results
    output_dir = f'results_thetamin_{theta_min:.2f}_thetamax_{theta_max:.2f}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    plot_dir = f'{output_dir}/plots'
    abs_dir = f'{plot_dir}/abs'
    real_dir = f'{plot_dir}/real'
    imag_dir = f'{plot_dir}/imag'
    total_sum_dir = f'{plot_dir}/total_sum'
    npy_dir = f'{output_dir}/npy'
    save_dir = f'{output_dir}/vectors'
    combined_dir = f'{plot_dir}/combined'
    combined_for_states_dir = f'{plot_dir}/combined_for_states'
    all_types_for_combined_dir = f'{plot_dir}/all_types_for_combined'
    os.makedirs(all_types_for_combined_dir, exist_ok=True)
    # os.makedirs(combined_for_states_dir, exist_ok=True)
    # os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(abs_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(imag_dir, exist_ok=True)
    # os.makedirs(total_sum_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # Calculate Hamiltonians and eigenvectors at each theta value, explicitly including endpoint
    hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals)
    H_thetas = H_theta = hamiltonian.H_thetas()
    r_theta = R_thetas = hamiltonian.R_thetas()
    
    # Calculate eigenvectors
    eigenvectors = eigvecs_all = np.array([np.linalg.eigh(H)[1] for H in H_theta])
    eigvals_all = np.array([np.linalg.eigh(H)[0] for H in H_theta])
    
    # Calculate overlaps between eigenstates at different theta values
    overlaps = np.zeros((eigenvectors.shape[2], len(theta_vals)), dtype=complex)
    for state in range(eigenvectors.shape[2]):
        for i in range(len(theta_vals)):
            current_eigenvector = eigenvectors[i, :, state]
            next_eigenvector = eigenvectors[(i + 1) % len(theta_vals), :, state]
            # Include endpoint by using the first eigenvector for the last point
            if i == len(theta_vals) - 1:
                next_eigenvector = eigenvectors[0, :, state]
            overlaps[state, i] = np.conj(current_eigenvector).T @ next_eigenvector
            if i == len(theta_vals): #if we are at the endpoint-1 point, the current eigvec is end-1th and the startpoint will bee the next eigvec
                next_eigenvector = eigenvectors[0, :, state] # endpoint
                overlaps[state, i] = np.conj(current_eigenvector).T @ next_eigenvector

    #create a report of the imag(eigenvectors) into the output_dir
    with open(f'{output_dir}/imag_eigenvectors.out', 'w') as file:
        file.write("#Imaginary parts of eigenvectors\n===============================================================\n")
        file.write("#Theta(degrees)\tState 0\tState 1\tState 2\tState 3\n")
        for i in range(len(theta_vals)):
            theta_deg = np.degrees(theta_vals[i])
            file.write(f'{theta_deg:.2f}\t')
            for state in range(eigenvectors.shape[2]):
                file.write('\t'.join(f'{val:.14f}' for val in np.imag(eigenvectors[i, :, state])) + '\t')
            file.write('\n')
    # create a raiser for the eigenvec_imag parts if the imag(eigenvector) is above 1e-14 treshold
    #save the report to a imag_eigenvecs.report file
    with open(f'{output_dir}/imag_eigenvecs.report', 'w') as file:
        file.write("#Imaginary parts of eigenvectors\n===============================================================\n")
        file.write("#Theta(degrees)\tState 0\tState 1\tState 2\tState 3\n")
        for i in range(len(theta_vals)):
            theta_deg = np.degrees(theta_vals[i])
            #create a raiser for the eigenvec_imag parts if the imag(eigenvector) is above 1e-14 treshold
            for state in range(eigenvectors.shape[2]):
                # Use np.any() to check if any element exceeds the threshold
                if np.any(np.abs(np.imag(eigenvectors[i, :, state])) > 1e-14):  
                    file.write(f'Imaginary part of eigenvector {state} at theta {theta_vals[i]} is above 1e-14 treshold\n')
                else:
                    file.write(f"All of the imaginary parts of the eigenvector {state} components at theta {theta_vals[i]} are zero.\n")

    # Save overlaps
    np.save(f'{npy_dir}/overlaps_{state}.npy', overlaps)
    #save the overlaps into an overlaps.out file too
    # Write eigenstate overlaps to file
    with open(f'{output_dir}/eigenstate_overlaps.out', 'w') as f:
        f.write('# Eigenstate Overlaps vs Theta\n')
        f.write('# Theta (degrees)\tState 0\tState 1\tState 2\tState 3\n')
        for i, theta in enumerate(theta_vals):
            theta_deg = np.degrees(theta)
            f.write(f'{theta_deg:.2f}\t')
            for state in range(eigenvectors.shape[2]):
                f.write(f'{overlaps[state, i]:.8f}\t')
            f.write('\n')
    
    #plot overlaps in 2x2 grid
    plt.figure()
    for state in range(eigenvectors.shape[2]):
        plt.subplot(2, 2, state + 1) #add1 to the titles since 1,2,3,4 states
        plt.plot(theta_vals, overlaps[state])
        plt.xlabel('Theta')
        plt.ylabel('Overlap')
        plt.title(f'Overlap between eigenstates {state}')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/overlap.png')
    plt.close()
    
    # Plot eigenvector components (4 subplots in a 2x2 grid for each eigenstate)
    for state in range(eigenvectors.shape[2]):
        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvector Components - State {state}', fontsize=16)  # Overall title
        
        plt.subplot(2, 2, 1)  # Top left subplot
        plt.plot(theta_vals, np.real(eigenvectors[:, 0, state]), label='Re(Comp 0)')
        plt.plot(theta_vals, np.imag(eigenvectors[:, 0, state]), label='Im(Comp 0)')
        plt.plot(theta_vals, np.abs(eigenvectors[:, 0, state]), label='Abs(Comp 0)')
        plt.xlabel('Theta')
        plt.ylabel('Component 0')
        plt.legend()
        
        plt.subplot(2, 2, 2)  # Top right subplot
        plt.plot(theta_vals, np.real(eigenvectors[:, 1, state]), label='Re(Comp 1)')
        plt.plot(theta_vals, np.imag(eigenvectors[:, 1, state]), label='Im(Comp 1)')
        plt.plot(theta_vals, np.abs(eigenvectors[:, 1, state]), label='Abs(Comp 1)')
        plt.xlabel('Theta')
        plt.ylabel('Component 1')
        plt.legend()
        
        plt.subplot(2, 2, 3)  # Bottom left subplot
        plt.plot(theta_vals, np.real(eigenvectors[:, 2, state]), label='Re(Comp 2)')
        plt.plot(theta_vals, np.imag(eigenvectors[:, 2, state]), label='Im(Comp 2)')
        plt.plot(theta_vals, np.abs(eigenvectors[:, 2, state]), label='Abs(Comp 2)')
        plt.xlabel('Theta')
        plt.ylabel('Component 2')
        plt.legend()
        
        plt.subplot(2, 2, 4)  # Bottom right subplot
        plt.plot(theta_vals, np.real(eigenvectors[:, 3, state]), label='Re(Comp 3)')
        plt.plot(theta_vals, np.imag(eigenvectors[:, 3, state]), label='Im(Comp 3)')
        plt.plot(theta_vals, np.abs(eigenvectors[:, 3, state]), label='Abs(Comp 3)')
        plt.xlabel('Theta')
        plt.ylabel('Component 3')
        plt.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for overall title
        plt.savefig(f'{plot_dir}/eigenvector_components_state_{state}_2x2.png')
        plt.close()

    # Calculate H*v for each theta value
    num_states = eigvecs_all.shape[2]

    Hv_results = np.zeros((len(theta_vals), num_states), dtype=complex)
    #get eigenvaluesof each H_theta, it is not theta vals
    #calculate H_thetas array by calculating H_theta, it should be a (num_points, 4, 4) array, like (theta_value, 4, 4)
    #H_thetas = np.array([hamiltonian(theta, omega, aVx, aVa, c_const, x_shift, d)[0] for theta in theta_vals])
    #print(H_thetas.shape)
    # Get all the eigenvalues
    eigenvalues = np.array([np.linalg.eigvalsh(H) for H in H_thetas])
    
    # Get all eigenvalues and eigenvectors separately
    eigenvals_eigvecs = [np.linalg.eigh(H) for H in H_thetas]
    eigenvalues_full = np.array([ev[0] for ev in eigenvals_eigvecs])
    
    # Extract the eigenvalues and eigenvectors
    eigenvalues = np.array([ev[0] for ev in eigenvals_eigvecs])
    eigenstates = np.array([ev[1] for ev in eigenvals_eigvecs])
    #print(eigenstates)
    plt.plot(theta_vals, eigenvalues[:,0], 'r-')
    plt.plot(theta_vals, eigenvalues[:,1], 'b-')
    plt.plot(theta_vals, eigenvalues[:,2], 'g-')
    plt.plot(theta_vals, eigenvalues[:,3], 'c-')
    plt.xlabel('Theta')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues vs Theta')
    plt.savefig(f'{plot_dir}/eigenvalues.png')
    plt.close()

    #plot R_thetas 3 components for each R_theta in R_thetas vs theta in theta_vals
    plt.figure()
    for i in range(3):
        plt.plot(theta_vals, np.array(R_thetas)[:, i], label=f'R_theta {i}')
    plt.xlabel('Theta')
    plt.ylabel('R_theta')
    plt.title(f'R_thetas vs Theta')
    plt.legend()
    plt.savefig(f'{save_dir}/R_thetas.png')
    plt.close()


    for state in range(num_states):
        for j, theta in enumerate(theta_vals):
            Hv_results[j] = H_thetas[j] @ eigenstates[j, :, state]

    # MAYBE THIS COULD BE A PROBLEM
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for states in range(4):
        plot_type = "Magnitude"
        axs[states].plot(theta_vals, np.abs(Hv_results[:, states]), 'r-', label='|H*v|')
        axs[states].set_title(f'State {states} - {plot_type}')
        axs[states].grid(True)
        if states == 0:
            axs[states].legend()
        if states >= 2:  # Only set x-label on bottom rows
            axs[states].set_xlabel('Theta')
        if states == 0 or states == 2:  # Only set y-label on left columns
            axs[states].set_ylabel('Value')
    plt.tight_layout()
    plt.suptitle('H*v Magnitudes for All States')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/abs/all_states_magnitudes.png')
    plt.close()

    # Real part plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for state in range(4):
        plot_type = "Real"
        axs[state].plot(theta_vals, np.real(Hv_results[:, state]), 'b-', label='Re(H*v)')
        axs[state].set_title(f'State {state} - {plot_type}')
        axs[state].grid(True)
        if state >= 2:  # Only set x-label on bottom rows
            axs[state].set_xlabel('Theta')
        if state == 0 or state == 2:  # Only set y-label on left columns
            axs[state].set_ylabel('Value')
    plt.tight_layout()
    plt.suptitle('H*v Real Parts for All States')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/real/all_states_real.png')
    plt.close()

    # Imaginary part plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for state in range(4):
        plot_type = "Imaginary"
        axs[state].plot(theta_vals, np.imag(Hv_results[:, state]), 'g-', label='Im(H*v)')
        axs[state].set_title(f'State {state} - {plot_type}')
        axs[state].grid(True)
        if state >= 2:  # Only set x-label on bottom rows
            axs[state].set_xlabel('Theta')
        if state == 0 or state == 2:  # Only set y-label on left columns
            axs[state].set_ylabel('Value')
    plt.tight_layout()
    plt.suptitle('H*v Imaginary Parts for All States')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/imag/all_states_imaginary.png')
    plt.close()

    # All types plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for state in range(4):
        plot_type = "All Types"
        axs[state].plot(theta_vals, np.abs(Hv_results[:, state]), 'r-', label='|H*v|')
        axs[state].plot(theta_vals, np.real(Hv_results[:, state]), 'b-', label='Re(H*v)')
        axs[state].plot(theta_vals, np.imag(Hv_results[:, state]), 'g-', label='Im(H*v)')
        axs[state].set_title(f'State {state} - {plot_type}')
        axs[state].grid(True)
        axs[state].legend()
        if state >= 2:  # Only set x-label on bottom rows
            axs[state].set_xlabel('Theta')
        if state == 0 or state == 2:  # Only set y-label on left columns
            axs[state].set_ylabel('Value')
    plt.tight_layout()
    plt.suptitle('H*v All Types for All States')
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{plot_dir}/all_types_for_combined/all_states_all_types.png')
    plt.close()


    # Calculate Berry phase using the original method
    berry_phase_calculator_original = NewBerryPhaseCalculator(hamiltonian, r_theta, eigenvectors, theta_vals)
    berry_phase_original = berry_phase_calculator_original.calculate_berry_phase()
    print("Berry phases (original method):")
    for i, phase in enumerate(berry_phase_original):
        print(f"  Eigenstate {i}: {phase}")

    # Calculate Berry phase using the finite difference with respect to theta
    berry_phase_calculator_theta = NewBerryPhaseCalculator(hamiltonian, r_theta, eigenvectors, theta_vals)
    berry_phase_theta = berry_phase_calculator_theta.calculate_berry_phase_theta_derivative()
    print("\nBerry phases (finite difference w.r.t. theta):")
    for i, phase in enumerate(berry_phase_theta):
        print(f"  Eigenstate {i}: {phase}")
    
    # Calculate the Berry curvature
    berry_curvature = berry_phase_calculator_theta.calculate_berry_curvature()

    # Use berry_curvature (it's a numpy array)
    print("Berry Curvature:", berry_curvature)

    # Save the Berry curvature to a file
    np.savetxt(f'{output_dir}/berry_curvature.txt', berry_curvature)
    
    #use the perfect_orthogonal_circle.py script to visualize the R_theta vectors
    from perfect_orthogonal_circle import verify_circle_properties, visualize_perfect_orthogonal_circle, generate_perfect_orthogonal_circle
    
    #visualize the R_theta vectors
    points = multiprocessing_create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max) #we already have a method for this
    #points = create_perfect_orthogonal_circle(R_0, d, num_points, theta_min, theta_max)
    print(points.shape)
    visualize_perfect_orthogonal_circle(points, save_dir)
    verify_circle_properties(d, num_points, points, save_dir)

    with open(f'{output_dir}/eigenvector_diff.out', "a") as log_file:
        log_file.write('#State Theta Norm_Diff\n')
        for i in range(1, len(theta_vals)):
            for j in range(eigenvectors.shape[2]):
                log_file.write(f"State {j}, Theta {theta_vals[i]:.2f}: {np.linalg.norm(eigenvectors[i, j] - eigenvectors[i-1, j]):.6f}\n")
        log_file.close()

    Va_values = []
    Vx_values = []
    Hamiltonians = []

    # Convert lists to numpy arrays
    Hamiltonians = np.array(H_theta)
    Va_values = np.array(hamiltonian.Va_theta_vals(R_thetas))
    Vx_values = np.array(hamiltonian.Vx_theta_vals(R_thetas))

    #create a directory in the output directory for npy files
    #output_dir = os.path.join(output_dir, 'output_berry_phase_results_thetamin_0.00_thetamax_6.28_20250324150750')
    npy_dir = os.path.join(output_dir, 'npy')

    # Create the directory if it doesn't exist
    os.makedirs(npy_dir, exist_ok=True)

    # Save the Hamiltonians, Va and Vx into .npy files
    np.save(f'{npy_dir}/Hamiltonians.npy', Hamiltonians)
    np.save(f'{npy_dir}/Va_values.npy', Va_values)
    np.save(f'{npy_dir}/Vx_values.npy', Vx_values)

    # Load potential values
    Va_values = np.load(f'{npy_dir}/Va_values.npy')
    Vx_values = np.load(f'{npy_dir}/Vx_values.npy')
    theta_values = np.linspace(0, 2*np.pi, len(Va_values))
    
    #plot Va potential components
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals, Va_values[:, i], label=f'Va[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Va Components')
    plt.title('Va Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Va_components.png')
    print("Va plots saved to figures directory.")

    #plot Vx potential components
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(theta_vals, Vx_values[:, i], label=f'Vx[{i}]')
    plt.xlabel('Theta (θ)')
    plt.ylabel('Vx Components')
    plt.title('Vx Components vs Theta')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Vx_components.png')
    print("Vx plots saved to figures directory.")

    #plot Va and Vx potentials in 3D as two subplots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create 3D parametric plot for Va potential
    ax2.plot(Va_values[:, 0], Va_values[:, 1], Va_values[:, 2], label='Va potential')
    ax2.set_xlabel('Va_x')
    ax2.set_ylabel('Va_y')
    ax2.set_zlabel('Va_z')
    ax2.set_title('3D Va Potential')
    ax2.legend()
    
    # Create 3D parametric plot for Vx potential
    ax1.plot(Vx_values[:, 0], Vx_values[:, 1], Vx_values[:, 2], label='Vx potential')
    ax1.set_xlabel('Vx_x')
    ax1.set_ylabel('Vx_y')
    ax1.set_zlabel('Vx_z')
    ax1.set_title('3D Vx Potential')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Va_Vx_3D.png')
    print("3D Va and Vx plots saved to figures directory.")