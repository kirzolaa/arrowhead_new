import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines # For custom legend
from mpl_toolkits.mplot3d import Axes3D # Corrected import
import sys
import os

# Add the directory containing new_bph to sys.path
# Assumes new_bph.py is in the same directory as va_vx_plotting.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_bph import Hamiltonian 

def main():
    # Parameters from user and Hamiltonian class
    # Good parameter set from gabor_bph.py
    aVx = 1.0
    aVa = 1.3
    c_const = 0.01  # Potential constant for Va
    x_shift = 0.1   # Shift for Va
    omega = 0.1     # Angular frequency
    R_0 = np.array([0.0, 0.0, 0.0]) # Origin

    # Ranges for d and theta
    d_vals = np.linspace(0.002, 1.5, 30)  # 30 steps for d
    theta_vals = np.linspace(0, 2 * np.pi, 60) # 60 steps for theta

    # Basis vectors for projection (plane orthogonal to (1,1,1))
    b1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    b2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)

    num_d_steps = len(d_vals)
    num_theta_steps = len(theta_vals)

    Plot_X_Va = np.zeros((num_d_steps, num_theta_steps))
    Plot_Y_Va = np.zeros((num_d_steps, num_theta_steps))
    Va_Z = np.zeros((num_d_steps, num_theta_steps))

    # Vx will share the same X, Y projection coordinates
    Vx_Z = np.zeros((num_d_steps, num_theta_steps))

    intersection_points = []
    intersection_tolerance = 0.05 # Tolerance for Z values to be considered an intersection

    for i, d_val in enumerate(d_vals):
        hamiltonian_instance = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, d_val, theta_range=[0.0]) # dummy theta_range

        for j, theta_val in enumerate(theta_vals):
            R_vec = hamiltonian_instance.R_theta(theta_val)

            plot_x_coord = np.dot(R_vec, b1)
            plot_y_coord = np.dot(R_vec, b2)

            Plot_X_Va[i, j] = plot_x_coord
            Plot_Y_Va[i, j] = plot_y_coord
            
            Va_components = hamiltonian_instance.V_a(R_vec)
            Va_Z[i, j] = np.sum(Va_components)

            Vx_components = hamiltonian_instance.V_x(R_vec)
            Vx_Z[i, j] = np.sum(Vx_components)

            # Check for intersection
            if np.abs(Va_Z[i, j] - Vx_Z[i, j]) < intersection_tolerance:
                intersection_points.append((Plot_X_Va[i, j], Plot_Y_Va[i, j], (Va_Z[i, j] + Vx_Z[i, j]) / 2.0))

    # Plot the surfaces
    fig = plt.figure(figsize=(24, 8)) # Adjusted for 1x3 layout

    fig.suptitle(f"Va and Vx Potential Surfaces over d=[{d_vals[0]:.1f}-{d_vals[-1]:.1f}] and $\theta$=[0-2$\\pi$] Parameter Space", fontsize=16)

    intersection_points_np = np.array(intersection_points)

    # Va surface (subplot 1)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(Plot_X_Va, Plot_Y_Va, Va_Z, color='red', alpha=0.8, rstride=1, cstride=1)
    ax1.set_title("Va Potential Surface")
    ax1.set_xlabel("Projection on b1")
    ax1.set_ylabel("Projection on b2")
    ax1.set_zlabel("Va (sum of components)")
    ax1.view_init(elev=30, azim=165)

    # Vx surface (subplot 2)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(Plot_X_Va, Plot_Y_Va, Vx_Z, color='blue', alpha=0.8, rstride=1, cstride=1)
    ax2.set_title("Vx Potential Surface")
    ax2.set_xlabel("Projection on b1")
    ax2.set_ylabel("Projection on b2")
    ax2.set_zlabel("Vx (sum of components)")
    ax2.view_init(elev=30, azim=165)

    # Combined Va and Vx surface (subplot 3)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(Plot_X_Va, Plot_Y_Va, Va_Z, color='red', alpha=0.6, rstride=1, cstride=1, label='Va Potential')
    ax3.plot_surface(Plot_X_Va, Plot_Y_Va, Vx_Z, color='blue', alpha=0.6, rstride=1, cstride=1, label='Vx Potential')
    if intersection_points_np.shape[0] > 0:
        ax3.scatter(intersection_points_np[:, 0], intersection_points_np[:, 1], intersection_points_np[:, 2], color='green', s=15, label='Intersection', depthshade=False, edgecolor='black', linewidth=0.5)
    ax3.set_title("Combined Va and Vx Surfaces")
    ax3.set_xlabel("Projection on b1")
    ax3.set_ylabel("Projection on b2")
    ax3.set_zlabel("Potential Value")
    ax3.view_init(elev=30, azim=165)

    # Common Legend
    red_proxy = mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Va Potential')
    blue_proxy = mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=10, label='Vx Potential')
    handles_for_legend = [red_proxy, blue_proxy]
    if intersection_points_np.shape[0] > 0:
        green_proxy_intersect = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='Intersection', markeredgecolor='black')
        handles_for_legend.append(green_proxy_intersect)
    fig.legend(handles=handles_for_legend, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=len(handles_for_legend), fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.88]) # Adjust rect to make space for suptitle and legend
    
    # Save the 1x3 enhanced figure
    plot_filename_enhanced = "va_vx_surface_plot_enhanced.png"
    save_path_enhanced = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename_enhanced)
    plt.savefig(save_path_enhanced)
    print(f"Enhanced plot saved to {save_path_enhanced}")

    # Create a separate figure for the combined plot with multiple viewpoints
    fig_combined_multi_view = plt.figure(figsize=(16, 14)) # Adjusted size for 2x2 grid
    fig_combined_multi_view.suptitle(f"Combined Va & Vx Surfaces - Multiple Viewpoints\n(d=[{d_vals[0]:.1f}-{d_vals[-1]:.1f}], $\theta$=[0-2$\\pi$])", fontsize=16)

    viewpoints = [
        {'elev': 90, 'azim': -90, 'title': 'XY Plane View (Top-Down)'}, # Top-Left
        {'elev': 0,  'azim': -90, 'title': 'XZ Plane View (Front)'},    # Top-Right
        {'elev': 0,  'azim': 0,   'title': 'YZ Plane View (Side)'},     # Bottom-Left
        {'elev': 30, 'azim': 165, 'title': 'Perspective View'}          # Bottom-Right
    ]

    for i, vp in enumerate(viewpoints):
        ax = fig_combined_multi_view.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(Plot_X_Va, Plot_Y_Va, Va_Z, color='red', alpha=0.6, rstride=1, cstride=1, label='Va Potential')
        ax.plot_surface(Plot_X_Va, Plot_Y_Va, Vx_Z, color='blue', alpha=0.6, rstride=1, cstride=1, label='Vx Potential')
        if intersection_points_np.shape[0] > 0:
            ax.scatter(intersection_points_np[:, 0], intersection_points_np[:, 1], intersection_points_np[:, 2], color='green', s=15, label='Intersection', depthshade=False, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Proj. on b1 (X')")
        ax.set_ylabel("Proj. on b2 (Y')")
        ax.set_zlabel("Potential (Z')")
        ax.set_title(vp['title'])
        ax.view_init(elev=vp['elev'], azim=vp['azim'])

    # Common Legend for the multi-view combined plot
    red_proxy_multi = mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Va Potential')
    blue_proxy_multi = mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=10, label='Vx Potential')
    handles_for_multi_legend = [red_proxy_multi, blue_proxy_multi]
    if intersection_points_np.shape[0] > 0:
        green_proxy_multi_intersect = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='Intersection', markeredgecolor='black')
        handles_for_multi_legend.append(green_proxy_multi_intersect)
    fig_combined_multi_view.legend(handles=handles_for_multi_legend, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles_for_multi_legend), fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust for suptitle and legend

    # Save the multi-view combined figure
    plot_filename_combined_multi = "va_vx_combined_multi_view_plot.png" # New name for this plot
    save_path_combined_multi = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename_combined_multi)
    plt.savefig(save_path_combined_multi)
    print(f"Multi-view combined plot saved to {save_path_combined_multi}")

    # --- New Plot: b1 vs b2 vs Theta, color by Potential ---
    print("\nGenerating b1 vs b2 vs Theta plot (potential as color, theta as Z-axis)...")

    Theta_Z_grid = np.tile(theta_vals, (num_d_steps, 1))

    # Normalize potential values for coloring
    # Handle cases where all values in Va_Z or Vx_Z might be the same, causing norm issues
    va_min, va_max = Va_Z.min(), Va_Z.max()
    vx_min, vx_max = Vx_Z.min(), Vx_Z.max()

    norm_Va = plt.Normalize(va_min, va_max if va_max > va_min else va_max + 1e-6)
    facecolors_Va = plt.cm.Reds(norm_Va(Va_Z))

    norm_Vx = plt.Normalize(vx_min, vx_max if vx_max > vx_min else vx_max + 1e-6)
    facecolors_Vx = plt.cm.Blues(norm_Vx(Vx_Z))

    fig_theta_plot = plt.figure(figsize=(14, 10))
    ax_theta = fig_theta_plot.add_subplot(111, projection='3d')

    # Plot Va surface (color by Va potential, Z by Theta)
    ax_theta.plot_surface(Plot_X_Va, Plot_Y_Va, Theta_Z_grid, 
                          facecolors=facecolors_Va, alpha=0.6, 
                          rstride=1, cstride=1, shade=False, antialiased=True)

    # Plot Vx surface (color by Vx potential, Z by Theta)
    ax_theta.plot_surface(Plot_X_Va, Plot_Y_Va, Theta_Z_grid, 
                          facecolors=facecolors_Vx, alpha=0.6, 
                          rstride=1, cstride=1, shade=False, antialiased=True)

    ax_theta.set_xlabel("Projection on b1")
    ax_theta.set_ylabel("Projection on b2")
    ax_theta.set_zlabel("Theta (radians)")
    ax_theta.set_title("Va/Vx Surfaces: (Proj. b1, Proj. b2) vs. Theta, Colored by Potential", fontsize=14)
    ax_theta.view_init(elev=25, azim=135) # Adjusted view for better Z-axis (theta) visibility

    # Colorbar for Va Potential
    mappable_Va = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm_Va)
    mappable_Va.set_array([]) # Necessary for ScalarMappable not tied to an artist
    fig_theta_plot.colorbar(mappable_Va, ax=ax_theta, shrink=0.6, aspect=20, pad=0.05, label='Va Potential Value')

    # Colorbar for Vx Potential
    mappable_Vx = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm_Vx)
    mappable_Vx.set_array([]) # Necessary for ScalarMappable
    fig_theta_plot.colorbar(mappable_Vx, ax=ax_theta, shrink=0.6, aspect=20, pad=0.15, label='Vx Potential Value')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for title and colorbars

    # Save the new plot
    plot_filename_theta = "va_vx_theta_Z_plot.png"
    save_path_theta = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename_theta)
    plt.savefig(save_path_theta)
    print(f"Theta as Z-axis plot saved to {save_path_theta}")

    # --- New Plot: 2D Slices of Potentials ---
    print("\nGenerating 2D slice plots...")
    fig_slices, axs_slices = plt.subplots(2, 2, figsize=(15, 12))
    fig_slices.suptitle("2D Slices of Va and Vx Potentials", fontsize=16)

    central_d_idx = num_d_steps // 2
    central_theta_idx = num_theta_steps // 2

    # Slice 1: Potential vs. theta (at central d)
    ax = axs_slices[0, 0]
    ax.plot(theta_vals, Va_Z[central_d_idx, :], label='Va Potential', color='red')
    ax.plot(theta_vals, Vx_Z[central_d_idx, :], label='Vx Potential', color='blue')
    ax.set_xlabel(f"Theta (radians) at d = {d_vals[central_d_idx]:.2f}")
    ax.set_ylabel("Potential Value")
    ax.set_title("Potential vs. Theta (Central d)")
    ax.legend()
    ax.grid(True)

    # Slice 2: Potential vs. d (at central theta)
    ax = axs_slices[0, 1]
    ax.plot(d_vals, Va_Z[:, central_theta_idx], label='Va Potential', color='red')
    ax.plot(d_vals, Vx_Z[:, central_theta_idx], label='Vx Potential', color='blue')
    ax.set_xlabel(f"d (radius) at theta = {theta_vals[central_theta_idx]:.2f} rad")
    ax.set_ylabel("Potential Value")
    ax.set_title("Potential vs. d (Central theta)")
    ax.legend()
    ax.grid(True)

    # Slice 3: Potential vs. Projection on b1 (at central theta)
    ax = axs_slices[1, 0]
    proj_b1_slice = Plot_X_Va[:, central_theta_idx]
    ax.plot(proj_b1_slice, Va_Z[:, central_theta_idx], label='Va Potential', color='red')
    ax.plot(proj_b1_slice, Vx_Z[:, central_theta_idx], label='Vx Potential', color='blue')
    ax.set_xlabel(f"Projection on b1 (at theta = {theta_vals[central_theta_idx]:.2f} rad)")
    ax.set_ylabel("Potential Value")
    ax.set_title("Potential vs. Proj. on b1 (Central theta)")
    ax.legend()
    ax.grid(True)

    # Slice 4: Potential vs. Projection on b2 (at central d)
    ax = axs_slices[1, 1]
    proj_b2_slice = Plot_Y_Va[central_d_idx, :]
    ax.plot(proj_b2_slice, Va_Z[central_d_idx, :], label='Va Potential', color='red')
    ax.plot(proj_b2_slice, Vx_Z[central_d_idx, :], label='Vx Potential', color='blue')
    ax.set_xlabel(f"Projection on b2 (at d = {d_vals[central_d_idx]:.2f})")
    ax.set_ylabel("Potential Value")
    ax.set_title("Potential vs. Proj. on b2 (Central d)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_slices = "va_vx_slice_plots.png"
    save_path_slices = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename_slices)
    plt.savefig(save_path_slices)
    print(f"2D slice plots saved to {save_path_slices}")

    # --- New Plot: 1x2 Grid - Potential vs. Proj. b1 and Proj. b2 Slices ---
    print("\nGenerating 1x2 (Proj. b1, Proj. b2) slice plots...")
    fig_b_slices, axs_b_slices = plt.subplots(2, 1, figsize=(18, 16))
    fig_b_slices.suptitle("Potential Slices along Projected Axes", fontsize=16)

    # Re-use central_d_idx and central_theta_idx from previous section
    # central_d_idx = num_d_steps // 2
    # central_theta_idx = num_theta_steps // 2

    # Slice 1 (Left): Potential vs. Projection on b1 (at central theta, including negative d)
    ax_b1 = axs_b_slices[0]
    
    # Include negative d values to see the full parabola
    d_vals_full = np.concatenate([-d_vals[::-1], d_vals])
    
    # Calculate projections for negative d values
    proj_b1_neg_d = []
    va_z_neg_d = []
    vx_z_neg_d = []
    
    # Recalculate for negative d values
    proj_b1_neg_d = []
    va_z_neg_d = []
    vx_z_neg_d = []
    
    # For negative d values, we need to flip the sign of the projection
    for d_val in d_vals[::-1]:  # Go backwards to maintain order
        hamiltonian_instance = Hamiltonian(omega, aVx, aVa, x_shift, c_const, R_0, abs(d_val), theta_range=[theta_vals[central_theta_idx]])
        R_vec = hamiltonian_instance.R_theta(theta_vals[central_theta_idx])
        # Flip the sign of the projection for negative d values
        proj_x = -np.dot(R_vec, b1)
        proj_b1_neg_d.append(proj_x)
        
        va_components = hamiltonian_instance.V_a(R_vec)
        va_z_neg_d.append(np.sum(va_components))
        
        vx_components = hamiltonian_instance.V_x(R_vec)
        vx_z_neg_d.append(np.sum(vx_components))
    
    # For positive d values, use the original projections
    proj_b1_pos_d = Plot_X_Va[:, central_theta_idx]
    va_z_pos_d = Va_Z[:, central_theta_idx]
    vx_z_pos_d = Vx_Z[:, central_theta_idx]
    
    # Combine negative and positive d values
    proj_b1_full = np.concatenate([proj_b1_neg_d, proj_b1_pos_d])
    va_z_full = np.concatenate([va_z_neg_d, va_z_pos_d])
    vx_z_full = np.concatenate([vx_z_neg_d, vx_z_pos_d])
    
    # Sort the combined arrays by projection value for clean plotting
    sort_idx = np.argsort(proj_b1_full)
    proj_b1_sorted = proj_b1_full[sort_idx]
    va_z_sorted = va_z_full[sort_idx]
    vx_z_sorted = vx_z_full[sort_idx]
    
    # The arrays are already combined and sorted above
    
    # Plot the full parabola
    ax_b1.plot(proj_b1_sorted, va_z_sorted, label='Va Potential', color='red')
    ax_b1.plot(proj_b1_sorted, vx_z_sorted, label='Vx Potential', color='blue')
    ax_b1.set_xlabel(f"Projection on b1 (at theta = {theta_vals[central_theta_idx]:.2f} rad, full d range)")
    ax_b1.set_ylabel("Potential Value")
    ax_b1.set_title("Potential vs. Proj. on b1 (Central Theta, Full Range)")
    ax_b1.legend()
    ax_b1.grid(True)

    # Slice 2 (Right): Potential vs. Projection on b2 (at central d)
    ax_b2 = axs_b_slices[1]
    proj_b2_full_theta_slice = Plot_Y_Va[central_d_idx, :]
    ax_b2.plot(proj_b2_full_theta_slice, Va_Z[central_d_idx, :], label='Va Potential', color='red')
    ax_b2.plot(proj_b2_full_theta_slice, Vx_Z[central_d_idx, :], label='Vx Potential', color='blue')
    ax_b2.set_xlabel(f"Projection on b2 (at d = {d_vals[central_d_idx]:.2f}, full theta range)")
    ax_b2.set_ylabel("Potential Value")
    ax_b2.set_title("Potential vs. Proj. on b2 (Central d)")
    ax_b2.legend()
    ax_b2.grid(True)
    #ax_b2.set_xlim(-1.5, 1.5)  # Explicitly set X-axis limits

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust for suptitle
    plot_filename_b_slices = "va_vx_b1_b2_slice_plots.png"
    save_path_b_slices = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename_b_slices)
    plt.savefig(save_path_b_slices)
    print(f"1x2 (Proj. b1, Proj. b2) slice plots saved to {save_path_b_slices}")

if __name__ == "__main__":
    main()
