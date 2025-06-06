import numpy as np
import os
from new_bph import Hamiltonian
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import animation

def fix_sign(eigvecs, printout, output_dir='eigenvals_vs_basis1_basis2_for_d'):
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

def main_plotting(d):
        #clear the console   
        os.system('clear')
        print("Processing plotting for d =", d)
        #let a be an aVx and an aVa parameter
        
        # Generate the arrowhead matrix and Va, Vx
        theta_vals = np.linspace(theta_min, theta_max, num_points, endpoint=True)

        #create a directory for the output
        output_dir = 'eigenvals_vs_basis1_basis2_for_d'
        os.makedirs(output_dir, exist_ok=True)
        
        #create a directory for the npy files
        npy_dir = os.path.join(output_dir, 'npy')
        os.makedirs(npy_dir, exist_ok=True)
        
        #create a directory for the plots
        plot_dir = os.path.join(output_dir, 'plots', f'd_{d}')
        os.makedirs(plot_dir, exist_ok=True)

        #get the eigenvalues
        hamiltonian = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d, theta_vals) #only for one single d
        H_thetas = hamiltonian.H_thetas() #only for one single d
        R_thetas = hamiltonian.R_thetas() #only for one single d
        
        # Calculate eigenvectors
        eigenvectors = fix_sign(np.array([np.linalg.eigh(H)[1] for H in H_thetas]), printout=0)
        eigenvectors = fix_sign(eigenvectors, printout=0)

        # Check if eigenvalues are properly sorted for each H in H_thetas
        print("\nChecking eigenvalue sorting for first 5 theta values:")
        for i in range(min(5, len(H_thetas))):
            H = H_thetas[i]
            eigvals = np.linalg.eigh(H)[0]
            print(f"\nTheta index {i}:")
            print(f"Eigenvalues: {eigvals}")
            print(f"Sorted? {np.all(np.diff(eigvals) >= 0)}")
        
        # Get all eigenvalues (already sorted by eigh)
        eigvals_all = np.array([np.linalg.eigh(H)[0] for H in H_thetas])
        
        #plot the eigvals vs theta into the plot dir
        plt.plot(theta_vals, eigvals_all[:,0], 'r-')
        plt.plot(theta_vals, eigvals_all[:,1], 'b-')
        plt.plot(theta_vals, eigvals_all[:,2], 'g-')
        plt.plot(theta_vals, eigvals_all[:,3], 'c-')
        plt.xlabel('Theta')
        plt.ylabel('Eigenvalue')
        plt.title(f'Eigenvalues vs Theta')
        plt.legend(['State 1', 'State 2', 'State 3', 'State 4'])
        plt.savefig(f'{output_dir}/eigenvalues_d_{d}.png')
        plt.close()

        # Verify all eigenvalues are in ascending order for each theta
        for i in range(len(eigvals_all)):
            if not np.all(np.diff(eigvals_all[i]) >= 0):
                print(f"WARNING: Eigenvalues not in ascending order for theta index {i}")
        
        print("\nEigenvalue statistics across all thetas:")
        for i in range(4):  # For each eigenvalue index (assuming 4x4 matrices)
            print(f"Eigenvalue {i}: min={np.min(eigvals_all[:, i]):.4f}, max={np.max(eigvals_all[:, i]):.4f}")
        
        # Plot eigenvector components
        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Eigenvectors, weighted by eigenvalues vs basis1 and basis2\n(c={c_const}, x_shift={x_shift}, d={d})', fontsize=16)
        
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([+2, -1, -1, 0])  # First basis vector
        basis2 = np.array([ 0, -1, +1, 0])   # Second basis vector
        
        # Normalize basis vectors
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Project and weight eigenvectors using eigenvalues
        proj_basis1 = np.zeros((len(theta_vals), 4))  # 4 states
        proj_basis2 = np.zeros((len(theta_vals), 4))  # 4 states
        
        for i in range(4):  # For each state
            for j in range(len(theta_vals)):  # For each theta
                # Get the eigenvector and its eigenvalue
                vec = eigenvectors[j, :, i]
                evalue = eigvals_all[j, i]
                
                # Weight the eigenvector by its eigenvalue
                weighted_vec = vec * np.abs(evalue)
                
                # Project the weighted vector onto basis vectors
                proj_basis1[j, i] = np.dot(weighted_vec, basis1)
                proj_basis2[j, i] = np.dot(weighted_vec, basis2)
        

        
        # Set up the figure and 3D axis
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each state with a different color and store the scatter plots
        colors = ['r', 'g', 'b', 'purple']
        scatters = []
        for i in range(4):
            scatter = ax.scatter(proj_basis1[:, i], proj_basis2[:, i], eigvals_all[:, i], 
                            c=colors[i], label=f'State {i}', s=10, alpha=0.7)
            scatters.append(scatter)
        
        # Set labels and title
        ax.set_xlabel('Projection on basis1')
        ax.set_ylabel('Projection on basis2')
        ax.set_zlabel('Eigenfunc.')
        ax.set_title(f'3D Plot of Eigenvectors, weighted by eigenvalues vs Basis Projections\n(c={c_const}, x_shift={x_shift}, d={d})')
        ax.legend()
        
        # Adjust the viewing angle and distance
        ax.view_init(elev=30, azim=45)
        
        # Function to rotate the view
        def rotate(angle):
            ax.view_init(elev=30, azim=angle)
        
        # Save multiple views of the 3D plot
        for angle in range(0, 360, 30):
            rotate(angle)
            plt.draw()
            plt.savefig(f'{plot_dir}/3D_plot_angle_{angle:03d}_c_{c_const}_x_shift_{x_shift}_d_{d}.png', 
                    dpi=100, bbox_inches='tight')
        
        # Save the final view
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/3D_plot_c_{c_const}_x_shift_{x_shift}_d_{d}.png', 
                dpi=100, bbox_inches='tight')
        plt.close()



        # Create surface plot directory
        surface_plot_dir = os.path.join(plot_dir, 'surfaces')
        os.makedirs(surface_plot_dir, exist_ok=True)

        # Optional: different colormaps or keep them all 'viridis' with varying shades
        colormaps = ['Reds', 'Greens', 'Blues', 'Purples']

        # Create a single figure for all states
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Loop over each state and plot as a transparent surface
        for i in range(4):
            x = proj_basis1[:, i]
            y = proj_basis2[:, i]
            z = eigvals_all[:, i]

            # Create regular grid
            xi = np.linspace(np.min(x), np.max(x), 200)
            yi = np.linspace(np.min(y), np.max(y), 200)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate z values onto the grid
            zi = griddata((x, y), z, (xi, yi), method='cubic')

            # Plot surface with transparency
            surf = ax.plot_surface(
                xi, yi, zi,
                cmap=colormaps[i],
                edgecolor='k',
                linewidth=0.2,
                antialiased=True,
                alpha=0.65,
                label=f'State {i}'  # Note: not shown in legend directly
            )

        # Annotate the axes
        ax.set_xlabel('Projection on basis1', fontsize=12)
        ax.set_ylabel('Projection on basis2', fontsize=12)
        ax.set_zlabel('Eigenfunc.', fontsize=12)
        ax.set_title(f'Combined Surface Plot of Eigenstates\n(c={c_const}, x_shift={x_shift}, d={d})', fontsize=14)

        # Optional: customize view angle, use front view for default
        ax.view_init(elev=15, azim=30)

        # Save the combined plot into a combined folder
        # create the combined folder if it doesn't exist
        combined_dir = os.path.join(surface_plot_dir, 'combined')
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(combined_dir, f'combined_surface_plot_c_{c_const}_x_{x_shift}_d_{d}.png')
        plt.tight_layout()
        plt.savefig(combined_path, dpi=150)
        plt.close()

        #PLOT FOR EACH INDIVIDUAL STATE INTO THE COMBINED DIR
        for i in range(4):
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(proj_basis1[:, i], proj_basis2[:, i], eigvals_all[:, i], c=colors[i], label=f'State {i}', s=10, alpha=0.7)
            ax.set_xlabel('Projection on basis1')
            ax.set_ylabel('Projection on basis2')
            ax.set_zlabel('Eigenfunc.')
            ax.set_title(f'3D Plot of Eigenvector, weighted by eigenvalue vs Basis Projections\n(c={c_const}, x_shift={x_shift}, d={d})')
            ax.legend()
            plt.tight_layout()
            #save these plots into the surface plot dir
            plt.savefig(f'{surface_plot_dir}/state_{i}_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
            plt.close()

        #plot only the second and the third state into the combined dir
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(proj_basis1[:, 1], proj_basis2[:, 1], eigvals_all[:, 1], c=colors[1], label=f'State 1', s=10, alpha=0.7)
        ax.scatter(proj_basis1[:, 2], proj_basis2[:, 2], eigvals_all[:, 2], c=colors[2], label=f'State 2', s=10, alpha=0.7)
        ax.set_xlabel('Projection on basis1')
        ax.set_ylabel('Projection on basis2')
        ax.set_zlabel('Eigenfunc.')
        ax.set_title(f'3D Plot of Eigenvector, weighted by eigenvalue vs Basis Projections\n(c={c_const}, x_shift={x_shift}, d={d})')
        ax.legend()
        plt.tight_layout()
        #save these plots into the surface plot dir
        plt.savefig(f'{surface_plot_dir}/state_1_2_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
        plt.close()

        # Create surface plot directory
        surface_plot_dir = os.path.join(plot_dir, 'surfaces')
        os.makedirs(surface_plot_dir, exist_ok=True)

        # Optional: different colormaps or keep them all 'viridis' with varying shades
        colormaps = ['Reds', 'Greens', 'Blues', 'Purples']

        # === Individual state surface plots ===
        for i in range(4):
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            x = proj_basis1[:, i]
            y = proj_basis2[:, i]
            z = eigvals_all[:, i]

            # Create regular grid
            xi = np.linspace(np.min(x), np.max(x), 200)
            yi = np.linspace(np.min(y), np.max(y), 200)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate z values
            zi = griddata((x, y), z, (xi, yi), method='cubic')

            # Plot surface
            surf = ax.plot_surface(
                xi, yi, zi,
                cmap=colormaps[i],
                edgecolor='k',
                linewidth=0.2,
                antialiased=True,
                alpha=0.7
            )

            ax.set_xlabel('Projection on basis1')
            ax.set_ylabel('Projection on basis2')
            ax.set_zlabel('Eigenfunc.')
            ax.set_title(f'Surface Plot of State {i}\n(c={c_const}, x_shift={x_shift}, d={d})')
            plt.tight_layout()
            plt.savefig(f'{surface_plot_dir}/corr_state_{i}_surface_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
            plt.close()

            # === Combined surface plot for state 1 and 2 ===
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            for i in [1, 2]:
                x = proj_basis1[:, i]
                y = proj_basis2[:, i]
                z = eigvals_all[:, i]

                xi = np.linspace(np.min(x), np.max(x), 200)
                yi = np.linspace(np.min(y), np.max(y), 200)
                xi, yi = np.meshgrid(xi, yi)

                zi = griddata((x, y), z, (xi, yi), method='cubic')

                ax.plot_surface(
                    xi, yi, zi,
                    cmap=colormaps[i],
                    edgecolor='k',
                    linewidth=0.2,
                    antialiased=True,
                    alpha=0.7,
                    label=f'State {i}'  # won't show in legend but keeps label structure
                )

            ax.set_xlabel('Projection on basis1')
            ax.set_ylabel('Projection on basis2')
            ax.set_zlabel('Eigenfunc.')
            ax.set_title(f'Surface Plot of State 1 and 2\n(c={c_const}, x_shift={x_shift}, d={d})')
            plt.tight_layout()
            plt.savefig(f'{surface_plot_dir}/corr_state_1_2_surface_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
            plt.close()


        # Ensure surface plot directory exists
        surface_plot_dir2 = os.path.join(plot_dir, 'surfaces', 'surfaces_fv') # for saving front view plots
        os.makedirs(surface_plot_dir2, exist_ok=True)

        # Use custom colormaps for clarity
        colormaps = ['Reds', 'Greens', 'Blues', 'Purples']

        # Front-view settings: looking along basis2 (Y) and Z axes
        elev_angle = 20   # slight tilt down
        azim_angle = 0    # 0° azimuth = looking along Y axis (basis2)

        # === Individual surface plots ===
        for i in range(4):
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            x = proj_basis1[:, i]
            y = proj_basis2[:, i]
            z = eigvals_all[:, i]

            # Grid for interpolation
            xi = np.linspace(np.min(x), np.max(x), 200)
            yi = np.linspace(np.min(y), np.max(y), 200)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            zi = griddata((x, y), z, (xi, yi), method='cubic')

            # Plot surface
            ax.plot_surface(
                xi, yi, zi,
                cmap=colormaps[i],
                edgecolor='k',
                linewidth=0.2,
                antialiased=True,
                alpha=0.8
            )

            ax.set_xlabel('Projection on basis1')
            ax.set_ylabel('Projection on basis2')
            ax.set_zlabel('Eigenfunc.')
            ax.set_title(f'Surface Plot of State {i}\n(c={c_const}, x_shift={x_shift}, d={d})')

            ax.view_init(elev=elev_angle, azim=azim_angle)  # front view

            plt.tight_layout()
            plt.savefig(f'{surface_plot_dir2}/corr_state_{i}_surface_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
            plt.close()

        # === Combined surface plot for state 1 and 2 ===
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in [1, 2]:
            x = proj_basis1[:, i]
            y = proj_basis2[:, i]
            z = eigvals_all[:, i]

            xi = np.linspace(np.min(x), np.max(x), 200)
            yi = np.linspace(np.min(y), np.max(y), 200)
            xi, yi = np.meshgrid(xi, yi)

            zi = griddata((x, y), z, (xi, yi), method='cubic')

            ax.plot_surface(
                xi, yi, zi,
                cmap=colormaps[i],
                edgecolor='k',
                linewidth=0.2,
                antialiased=True,
                alpha=0.8
            )

        ax.set_xlabel('Projection on basis1')
        ax.set_ylabel('Projection on basis2')
        ax.set_zlabel('Eigenfunc.')
        ax.set_title(f'Surface Plot of State 1 and 2\n(c={c_const}, x_shift={x_shift}, d={d})')

        ax.view_init(elev=elev_angle, azim=azim_angle)  # front view

        plt.tight_layout()
        plt.savefig(f'{surface_plot_dir2}/corr_state_1_2_surface_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
        plt.close()

        elev_angle = 20    # slight downward tilt
        azim_angle = 90    # rotate to look along basis1

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in [1, 2]:
            x = proj_basis1[:, i]
            y = proj_basis2[:, i]
            z = eigvals_all[:, i]
            ax.view_init(elev=elev_angle, azim=azim_angle)  # front view
            xi = np.linspace(np.min(x), np.max(x), 200)
            yi = np.linspace(np.min(y), np.max(y), 200)
            xi, yi = np.meshgrid(xi, yi)

            zi = griddata((x, y), z, (xi, yi), method='cubic')

            ax.plot_surface(
                xi, yi, zi,
                cmap=colormaps[i],
                edgecolor='k',
                linewidth=0.2,
                antialiased=True,
                alpha=0.7,
                label=f'State {i}'  # won't show in legend but keeps label structure
            )

        ax.set_xlabel('Projection on basis1')
        ax.set_ylabel('Projection on basis2')
        ax.set_zlabel('Eigenfunc.')
        ax.set_title(f'Surface Plot of State 1 and 2\n(c={c_const}, x_shift={x_shift}, d={d})')
        plt.tight_layout()
        plt.savefig(f'{surface_plot_dir2}/corr_b1vsz_state_1_2_surface_c_{c_const}_x_{x_shift}_d_{d}.png', dpi=150)
        plt.close()

            
    
if __name__ == "__main__":
    aVx = 1.0
    aVa = 1.3
    c_const = 0.01  # Potential constant, shifts the 2d parabola on the y axis
    x_shift = 0.1  # Shift in x direction
    theta_min = 0
    theta_max = 2 * np.pi
    omega = 0.1
    num_points = 500
    R_0 = (0.4333, 0.4333, 0.4333)
    d=0.001
    main_plotting(d)
