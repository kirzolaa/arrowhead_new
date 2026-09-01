"""
CI Analysis Plot Generator
==========================
Uses the new scientific_plotter.py to generate improved diagnostic plots
for CI analysis data stored in .npy files.

Processes the four CI folders and generates enhanced visualizations using
the Plot2D class from scientific_plotter.py.
"""

import numpy as np
import os
import glob
from scientific_plotter import Plot2D, PlotStyle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.constants import hbar
matplotlib.use('Agg')

def load_ci_data(npy_dir):
    """
    Load CI analysis data from .npy files.
    
    Parameters:
    - npy_dir: Directory containing .npy files
    
    Returns:
    - Dictionary containing loaded data
    """
    data = {}
    
    # Define expected files
    expected_files = {
        'gamma': 'gamma.npy',
        'tau': 'tau.npy', 
        'eigvecs': 'eigvecs.npy',
        'theta_vals': 'theta_vals.npy',
        'eigvals': 'eigvals.npy',
        'Va_values': 'Va_values.npy',
        'Vx_values': 'Vx_values.npy',
        'Hamiltonians': 'Hamiltonians.npy',
        'R_thetas' : "R_thetas.npy"
    }
    
    for key, filename in expected_files.items():
        filepath = os.path.join(npy_dir, filename)
        if os.path.exists(filepath):
            try:
                # First try without pickle (safer)
                loaded_data = np.load(filepath, allow_pickle=False)
                if isinstance(loaded_data, np.ndarray):
                    data[key] = loaded_data
                    print(f"Loaded {key} from {filename} (shape: {loaded_data.shape})")
                else:
                    print(f"Skipping {filename} - not a numpy array (type: {type(loaded_data)})")
            except (ValueError, AttributeError):
                # If that fails, try with pickle
                try:
                    loaded_data = np.load(filepath, allow_pickle=True)
                    if isinstance(loaded_data, np.ndarray):
                        data[key] = loaded_data
                        print(f"Loaded {key} from {filename} with pickle (shape: {loaded_data.shape})")
                    else:
                        print(f"Skipping {filename} - pickled object not a numpy array")
                except Exception as e:
                    print(f"Skipping {filename} - cannot load even with pickle (error: {type(e).__name__})")
                    continue
        else:
            print(f"Warning: {filename} not found")
    
    return data

def extract_ci_coords(folder_name):
    """
    Extract CI coordinates from folder name.
    
    Parameters:
    - folder_name: Folder name like 'd_0.001_-0.867_-0.867_1.733'
    
    Returns:
    - Tuple of (d, coord1, coord2, coord3)
    """
    # Remove 'd_' prefix and split by '_'
    parts = folder_name.replace('d_', '').split('_')
    return tuple(float(p) for p in parts)

def generate_gamma_evolution_plot(data, output_dir):
    """
    Generate improved gamma evolution plot using Plot2D.
    Only plots gamma[2,3] and gamma[3,2] as requested.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'gamma' not in data or 'theta_vals' not in data:
        print("Skipping gamma evolution plot - missing data")
        return
    
    gamma = data['gamma']
    theta_vals = data['theta_vals']
    print(f"theta_vals range: {theta_vals.min()/np.pi:.4f} to {theta_vals.max()/np.pi:.4f} (×π)")
    
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    # Only plot gamma[2,3] and gamma[3,2] as requested
    plotter.plot(theta_vals[1:]/np.pi, gamma[1, 2, 1:]/np.pi,
                color=PlotStyle.COLORS['primary'],
                linewidth=4, label=rf'$\gamma_{{2,3}}/\pi$')
    plotter.plot(theta_vals[1:]/np.pi, gamma[2, 1, 1:]/np.pi,
                color=PlotStyle.COLORS['secondary'],
                linewidth=4, label=rf'$\gamma_{{3,2}}/\pi$')
    
    plotter.set_labels(r'$\theta/\pi$', r'$\gamma/\pi$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='both', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('gamma_evolution_improved.png', save_dir=output_dir)
    print(f"Saved gamma evolution plot to {output_dir}")

def generate_tau_evolution_plot(data, output_dir):
    """
    Generate improved tau evolution plot using Plot2D.
    Only plots tau[2,3] and tau[3,2] as requested.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'tau' not in data or 'theta_vals' not in data:
        print("Skipping tau evolution plot - missing data")
        return
    
    tau = data['tau']
    theta_vals = data['theta_vals']
    
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    # Only plot tau[2,3] and tau[3,2] as requested
    plotter.plot(theta_vals[1:]/np.pi, np.imag(tau[1, 2, 1:]),
                color=PlotStyle.COLORS['primary'],
                linewidth=4, label=rf'$Im(\tau_{{2,3}})$')
    plotter.plot(theta_vals[1:]/np.pi, np.imag(tau[2, 1, 1:]),
                color=PlotStyle.COLORS['secondary'],
                linewidth=4, label=rf'$Im(\tau_{{3,2}})$')
    
    plotter.set_labels(r'$\theta/\pi$', r'$Im(\tau)$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='y', multiple=1.5)
    plotter.set_tick_locator(axis='x', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('tau_evolution_improved.png', save_dir=output_dir)
    print(f"Saved tau evolution plot to {output_dir}")

def generate_tau_abs_23_plot(data, output_dir):
    """
    Generate absolute tau[2,3] evolution plot using Plot2D.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'tau' not in data or 'theta_vals' not in data:
        print("Skipping tau[2,3] abs plot - missing data")
        return
    
    tau = data['tau']
    theta_vals = data['theta_vals']
    
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    # Plot absolute tau[2,3]
    plotter.plot(theta_vals[1:]/np.pi, np.abs(tau[1, 2, 1:]),
                color=PlotStyle.COLORS['primary'],
                linewidth=4, label=rf'$|\tau_{{2,3}}|$')
    
    plotter.set_labels(r'$\theta/\pi$', r'$|\tau_{2,3}|$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='y', multiple=1.5)
    plotter.set_tick_locator(axis='x', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('tau_abs_23_improved.png', save_dir=output_dir)
    print(f"Saved tau[2,3] abs plot to {output_dir}")

def generate_tau_abs_32_plot(data, output_dir):
    """
    Generate absolute tau[3,2] evolution plot using Plot2D.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'tau' not in data or 'theta_vals' not in data:
        print("Skipping tau[3,2] abs plot - missing data")
        return
    
    tau = data['tau']
    theta_vals = data['theta_vals']
    
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    # Plot absolute tau[3,2]
    plotter.plot(theta_vals[1:]/np.pi, np.abs(tau[2, 1, 1:]),
                color=PlotStyle.COLORS['secondary'],
                linewidth=4, label=rf'$|\tau_{{3,2}}|$')
    
    plotter.set_labels(r'$\theta/\pi$', r'$|\tau_{3,2}|$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='y', multiple=1.5)
    plotter.set_tick_locator(axis='x', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('tau_abs_32_improved.png', save_dir=output_dir)
    print(f"Saved tau[3,2] abs plot to {output_dir}")

def generate_gamma_heatmap(data, output_dir):
    """
    Generate improved gamma heatmap using matplotlib with scientific styling.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'gamma' not in data:
        print("Skipping gamma heatmap - missing data")
        return
    
    gamma = data['gamma']
    M, M, N = gamma.shape
    gamma_final = gamma[:, :, -1]
    
    plt.figure(figsize=(10, 8))
    
    # Use RdBu_r colormap with symmetric limits
    gamma_max = np.max(np.abs(gamma_final))
    im = plt.imshow(gamma_final/np.pi, cmap='RdBu_r', vmin=-gamma_max/np.pi, vmax=gamma_max/np.pi)
    
    # Apply scientific styling
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\gamma/\pi$', fontsize=PlotStyle.AXIS_LABELSIZE, labelpad=-10)
    cbar.ax.tick_params(labelsize=PlotStyle.TICKLABELSIZE)
    
    plt.xlabel('m', fontsize=PlotStyle.AXIS_LABELSIZE)
    plt.ylabel('n', fontsize=PlotStyle.AXIS_LABELSIZE)
    
    plt.tick_params(axis='both', which='major', labelsize=PlotStyle.TICKLABELSIZE, length=8, width=PlotStyle.AXIS_LINEWIDTH)
    plt.xticks([])
    plt.yticks([])
    
    # axis lkinewidth should be plotstyle.LINEWIDTH
    plt.gca().spines['bottom'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    plt.gca().spines['left'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    plt.gca().spines['top'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    plt.gca().spines['right'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    
    
    # Add text annotations
    for i in range(M):
        for j in range(M):
            value_text = f'{gamma_final[i,j]/np.pi:.3f}'
            text_color = 'w' if abs(gamma_final[i,j]/np.pi) > gamma_max/np.pi/2 else 'k'
            plt.text(j, i, value_text, ha="center", va="center", 
                    color=text_color, fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/gamma_heatmap_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gamma heatmap to {output_dir}")

def generate_eigenvector_components_plot(data, output_dir):
    """
    Generate eigenvector components plot using subplots.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'eigvecs' not in data or 'theta_vals' not in data:
        print("Skipping eigenvector components plot - missing data")
        return
    
    eigvecs = data['eigvecs']
    theta_vals = data['theta_vals']
    N, M, _ = eigvecs.shape  # N theta points, M states, M components
    
    plotter = Plot2D(figsize=(12, 12), dpi=300)
    fig, axes = plotter.create_subplots(2, 2, sharex='col')
    
    colors = [PlotStyle.COLORS['primary'], PlotStyle.COLORS['secondary'], 
              PlotStyle.COLORS['tertiary'], PlotStyle.COLORS['quaternary']]
    
    for component in range(4):
        ax = axes[component]
        plotter.ax = ax  # Set current axis for plotter methods
        
        for state in range(M):
            ax.plot(theta_vals, np.real(eigvecs[:, state, component]),
                   color=colors[state], linewidth=4, 
                   label=f'State {state+1}')
        
        if component >= 2:  # bottom row only
            ax.set_xlabel(r'$\theta/\pi$', fontsize=PlotStyle.AXIS_LABELSIZE)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)  # top row: hide, axis is shared
        
        ax.set_ylabel(rf'$Re(\psi_{{{component+1}}})$', fontsize=PlotStyle.AXIS_LABELSIZE)
        
        # axis lkinewidth should be plotstyle.LINEWIDTH
        ax.spines['bottom'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['left'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['top'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['right'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        if component == 0:  # Only show legend for first one
            ax.legend(loc='best', fontsize=PlotStyle.LEGEND_FONT_SIZE)
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA, linewidth=PlotStyle.GRID_LINEWIDTH)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/eigenvector_components_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvector components plot to {output_dir}")

def generate_combined_comparison_plot(all_data, output_dir):
    """
    Generate combined comparison plot for all CI configurations.
    Plots gamma[2,3] and gamma[3,2] for T, CI_1, CI_2, CI_3.
    
    Parameters:
    - all_data: Dictionary with CI names as keys and their data as values
    - output_dir: Directory to save the plot
    """
    ci_names = ['T (Trivial)', 'CI_1', 'CI_2', 'CI_3']
    
    # Plot gamma[2,3] comparison
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    colors = [PlotStyle.COLORS['primary'], PlotStyle.COLORS['secondary'], 
              PlotStyle.COLORS['tertiary'], PlotStyle.COLORS['quaternary']]
    
    for i, (ci_name, data) in enumerate(all_data.items()):
        if 'gamma' in data and 'theta_vals' in data:
            gamma = data['gamma']
            theta_vals = data['theta_vals']
            plotter.plot(theta_vals[1:]/np.pi, gamma[1, 2, 1:]/np.pi,
                        color=colors[i], linewidth=4, label=ci_name)
    
    plotter.set_labels(r'$\theta/\pi$', r'$\gamma_{2,3}/\pi$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='both', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('combined_gamma_23_comparison.png', save_dir=output_dir)
    
    # Plot gamma[3,2] comparison
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    for i, (ci_name, data) in enumerate(all_data.items()):
        if 'gamma' in data and 'theta_vals' in data:
            gamma = data['gamma']
            theta_vals = data['theta_vals']
            plotter.plot(theta_vals[1:]/np.pi, gamma[2, 1, 1:]/np.pi,
                        color=colors[i], linewidth=4, label=ci_name)
    
    plotter.set_labels(r'$\theta/\pi$', r'$\gamma_{3,2}/\pi$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='both', multiple=0.5)
    
    plotter.save('combined_gamma_32_comparison.png', save_dir=output_dir)
    
    # Plot tau[2,3] comparison
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    for i, (ci_name, data) in enumerate(all_data.items()):
        if 'tau' in data and 'theta_vals' in data:
            tau = data['tau']
            theta_vals = data['theta_vals']
            plotter.plot(theta_vals[1:]/np.pi, np.abs(tau[1, 2, 1:]),
                        color=colors[i], linewidth=4, label=ci_name)
    
    plotter.set_labels(r'$\theta/\pi$', r'$|\tau_{2,3}|$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='y', multiple=1.5)
    plotter.set_tick_locator(axis='x', multiple=0.5)
    
    plotter.save('combined_tau_23_comparison.png', save_dir=output_dir)
    
    # Plot tau[3,2] comparison
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    for i, (ci_name, data) in enumerate(all_data.items()):
        if 'tau' in data and 'theta_vals' in data:
            tau = data['tau']
            theta_vals = data['theta_vals']
            plotter.plot(theta_vals[1:]/np.pi, np.abs(tau[2, 1, 1:]),
                        color=colors[i], linewidth=4, label=ci_name)
    
    plotter.set_labels(r'$\theta/\pi$', r'$|\tau_{3,2}|$', use_latex=False)
    plotter.add_legend(loc='upper center')
    plotter.set_tick_locator(axis='y', multiple=1.5)
    plotter.set_tick_locator(axis='x', multiple=0.5)
    
    plotter.save('combined_tau_32_comparison.png', save_dir=output_dir)
    
    print(f"Saved combined comparison plots to {output_dir}")

def generate_eigenvalue_plot(data, output_dir):
    """
    Generate improved eigenvalue plot using Plot2D.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'theta_vals' not in data:
        print("Skipping eigenvalue plot - missing theta_vals")
        return
    
    theta_vals = data['theta_vals']
    
    # Try to get eigenvalues from different sources
    eigvals = None
    
    # First try eigvals if available
    if 'eigvals' in data:
        eigvals = data['eigvals']
        # Handle different eigenvalue data structures
        if eigvals.ndim == 3:  # Shape (N, M, M) or similar
            # Extract diagonal elements if it's a full matrix per theta
            if eigvals.shape[1] == eigvals.shape[2]:
                eigvals = np.array([np.linalg.eigvalsh(matrix) for matrix in eigvals])
            else:
                # Assume it's already in the right shape (N, M)
                pass
    # If eigvals not available, try to compute from Hamiltonians
    elif 'Hamiltonians' in data:
        print("Computing eigenvalues from Hamiltonians...")
        try:
            Hamiltonians = data['Hamiltonians']
            eigvals = np.array([np.linalg.eigvalsh(H) for H in Hamiltonians])
            print(f"Computed eigenvalues shape: {eigvals.shape}")
        except Exception as e:
            print(f"Error computing eigenvalues from Hamiltonians: {e}")
            return
    else:
        print("Skipping eigenvalue plot - no eigenvalue data available")
        return
    
    # Ensure we have the right shape (N, M) where N is number of theta points, M is number of eigenvalues
    if eigvals.ndim != 2:
        print(f"Unexpected eigenvalue shape: {eigvals.shape}, skipping eigenvalue plot")
        return
    
    N, M = eigvals.shape
    
    plotter = Plot2D(figsize=(12, 7), dpi=300)
    fig, ax = plotter.create_figure()
    
    colors = [PlotStyle.COLORS['primary'], PlotStyle.COLORS['secondary'], 
              PlotStyle.COLORS['tertiary'], PlotStyle.COLORS['quaternary']]
    
    for i in range(M):
        plotter.plot(theta_vals/np.pi, eigvals[:, i],
                    color=colors[i % len(colors)],
                    linewidth=2, label=f'State {i+1}')
    
    plotter.set_labels(r'$\theta/\pi$', 'Energy (eV)', use_latex=False)
    plotter.add_legend(loc='upper center')
    #plotter.set_tick_locator(axis='both', multiple=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    plotter.save('eigenvalues_improved.png', save_dir=output_dir)
    print(f"Saved eigenvalue plot to {output_dir}")

def generate_potential_components_plot(data, output_dir):
    """
    Generate improved Va and Vx components plot using Plot2D.
    
    Parameters:
    - data: Dictionary containing loaded CI data
    - output_dir: Directory to save the plot
    """
    if 'Va_values' not in data or 'Vx_values' not in data or 'theta_vals' not in data:
        print("Skipping potential components plot - missing data")
        return
    
    Va_values = data['Va_values']
    Vx_values = data['Vx_values']
    theta_vals = data['theta_vals']
    
    # Handle different data structures
    if Va_values.ndim != 2 or Va_values.shape[1] < 3:
        print(f"Unexpected Va_values shape: {Va_values.shape}, skipping potential components plot")
        return
    
    if Vx_values.ndim != 2 or Vx_values.shape[1] < 3:
        print(f"Unexpected Vx_values shape: {Vx_values.shape}, skipping potential components plot")
        return
    
    # Plot Va components
    try:
        plotter = Plot2D(figsize=(12, 7), dpi=300)
        fig, ax = plotter.create_figure()
        
        colors = [PlotStyle.COLORS['primary'], PlotStyle.COLORS['secondary'], 
                  PlotStyle.COLORS['tertiary']]
        
        # Use the actual number of components available
        num_components = min(3, Va_values.shape[1])
        
        for i in range(num_components):
            plotter.plot(theta_vals/np.pi, Va_values[:, i],
                        color=colors[i], linewidth=4, label=rf'$V_a[{i+1}]$')
        
        plotter.set_labels(r'$\theta/\pi$', r'$V_a Components (eV)$', use_latex=False)
        plotter.add_legend(loc='upper center')
        
        os.makedirs(output_dir, exist_ok=True)
        plotter.save('Va_components_improved.png', save_dir=output_dir)
        
        # Plot Vx components
        plotter = Plot2D(figsize=(15, 8), dpi=300)
        fig, ax = plotter.create_figure()
        
        num_components = min(3, Vx_values.shape[1])
        
        for i in range(num_components):
            plotter.plot(theta_vals/np.pi, Vx_values[:, i],
                        color=colors[i], linewidth=4, label=rf'$V_x[{i+1}]$')
        
        plotter.set_labels(r'$\theta/\pi$', r'$V_x Components (eV)$', use_latex=False)
        plotter.add_legend(loc='upper center')
        
        plotter.save('Vx_components_improved.png', save_dir=output_dir)

        #plot Vx-Va
        plotter = Plot2D(figsize=(12, 7), dpi=300)
        fig, ax = plotter.create_figure()
        
        for i in range(num_components):
            plotter.plot(theta_vals/np.pi, 100 *(Va_values[:, i] - Vx_values[:, i]),
                        color=colors[i], linewidth=4, label=rf'$V_a[{i+1}] - V_x[{i+1}]$')
        
        plotter.set_labels(r'$\theta/\pi$', r'$V_a - V_x$ (eV)', use_latex=False)
        plotter.add_legend(loc='upper center')
        plotter.set_tick_formatter(axis='y', format_str='%.5f')
        
        plotter.save('Vx_Va_comparison.png', save_dir=output_dir)
        
        print(f"Saved potential components plots to {output_dir}")
    except Exception as e:
        print(f"Error generating potential components plot: {e}")

def generate_r_thetas_projections_plot(data, output_dir):
    if 'R_thetas' not in data:
        return
        
    R_thetas = data['R_thetas']
    R_0 = np.mean(R_thetas, axis=0)
    
    basis1 = np.array([2.0, -1.0, -1.0])
    basis2 = np.array([0.0, -1.0, 1.0])
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    R_rel = R_thetas - R_0
    b1_proj = np.dot(R_rel, basis1)
    b2_proj = np.dot(R_rel, basis2)
    
    plotter = Plot2D(figsize=(12, 12), dpi=300)
    fig, axes = plotter.create_subplots(2, 2)
    
    projections = [
        (R_thetas[:, 0], R_thetas[:, 1], 'X', 'Y', 'XY Projection'),
        (R_thetas[:, 1], R_thetas[:, 2], 'Y', 'Z', 'YZ Projection'),
        (R_thetas[:, 0], R_thetas[:, 2], 'X', 'Z', 'XZ Projection'),
        (b1_proj, b2_proj, 'Basis 1', 'Basis 2', 'Orthogonal Plane')
    ]
    
    for i, (x_data, y_data, xlabel, ylabel, title) in enumerate(projections):
        ax = axes[i]
        plotter.ax = ax
        
        plotter.plot(x_data, y_data, color=PlotStyle.COLORS['primary'], linewidth=4)
        
        ax.set_xlabel(xlabel, fontsize=PlotStyle.AXIS_LABELSIZE)
        ax.set_ylabel(ylabel, fontsize=PlotStyle.AXIS_LABELSIZE)
        ax.set_title(title, fontsize=PlotStyle.AXIS_LABELSIZE)
        
        ax.spines['bottom'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['left'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['top'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        ax.spines['right'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
        
        ax.grid(True, alpha=PlotStyle.GRID_ALPHA, linewidth=PlotStyle.GRID_LINEWIDTH)
        ax.set_aspect('equal', adjustable='box')
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/R_thetas_projections_improved.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_ci_seam_3d_plot(x_prime, output_dir, r0_range=(-3.0, 3.0), r0_marker=0.0,
                              axis_title_size=24, axis_axistitle_size_xyz=None,
                              axis_labelsize=None, ticklabelsize=None,
                              legend_font_size=None, enlarge_axis_labelling_size_by=2):
    """
    Plot the trivial CI seam and the three nontrivial-CI branch lines in
    full 3D (r1, r2, r3) space, showing all three branches converge on
    the trivial CI. Saves the main view (with zoomed inset) plus two
    rotated views into output_dir.
    """
    if axis_axistitle_size_xyz is None:
        axis_axistitle_size_xyz = 0.8 * axis_title_size
    if axis_labelsize is None:
        axis_labelsize = 0.8 * axis_axistitle_size_xyz
    if ticklabelsize is None:
        ticklabelsize = 0.8 * axis_labelsize
    if legend_font_size is None:
        legend_font_size = ticklabelsize + 2

    axis_axistitle_size_xyz = int(axis_axistitle_size_xyz + enlarge_axis_labelling_size_by)
    ticklabelsize = int(ticklabelsize + enlarge_axis_labelling_size_by)

    plt.close('all')

    r0_vals = np.linspace(r0_range[0], r0_range[1], 200)
    r1_vals = 3 * r0_vals - 2 * x_prime
    r2_vals = 4 * x_prime - 3 * r0_vals

    branch1 = np.stack([r1_vals, r1_vals, r2_vals], axis=1)
    branch2 = np.stack([r1_vals, r2_vals, r1_vals], axis=1)
    branch3 = np.stack([r2_vals, r1_vals, r1_vals], axis=1)

    CI_0 = np.array([x_prime, x_prime, x_prime])
    seam_t = np.linspace(r0_range[0], r0_range[1], 50)
    seam = np.stack([seam_t, seam_t, seam_t], axis=1)

    limit_range = 2.0
    xlim = (x_prime - limit_range, x_prime + limit_range)
    ylim = (x_prime - limit_range, x_prime + limit_range)
    zlim = (x_prime - limit_range, x_prime + limit_range)

    def _clip_range(data):
        mask = ((data[:, 0] >= xlim[0]) & (data[:, 0] <= xlim[1]) &
                (data[:, 1] >= ylim[0]) & (data[:, 1] <= ylim[1]) &
                (data[:, 2] >= zlim[0]) & (data[:, 2] <= zlim[1]))
        return data[mask]

    seam = _clip_range(seam)
    branch1 = _clip_range(branch1)
    branch2 = _clip_range(branch2)
    branch3 = _clip_range(branch3)

    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.1, 0.1, 0.85, 0.8])

    if len(seam) > 0:
        ax.plot(*seam.T, '-', color='blue', label='CI_T', linewidth=4)
    for branch, color, name in [(branch1, 'tab:orange', 'CI_1'),
                                  (branch2, 'tab:green', 'CI_2'),
                                  (branch3, 'tab:red', 'CI_3')]:
        if len(branch) > 0:
            ax.plot(*branch.T, color=color, linewidth=4, label=name)

    

    r1_m, r2_m = 3 * r0_marker - 2 * x_prime, 4 * x_prime - 3 * r0_marker
    for pt in [(r1_m, r1_m, r2_m), (r1_m, r2_m, r1_m), (r2_m, r1_m, r1_m)]:
        if (xlim[0] <= pt[0] <= xlim[1] and ylim[0] <= pt[1] <= ylim[1] and
            zlim[0] <= pt[2] <= zlim[1]):
            ax.scatter(*pt, c='black', s=400, marker='X', zorder=5)

    ax.set_xlabel('$r_1$', fontsize=axis_axistitle_size_xyz + 20, labelpad=25)
    ax.set_ylabel('$r_2$', fontsize=axis_axistitle_size_xyz + 20, labelpad=40)
    ax.set_zlabel('$r_3$', fontsize=axis_axistitle_size_xyz + 20, labelpad=45)
    ax.xaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.label.set_position((0.5, 1.1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.view_init(azim=75, elev=20)

    ax.tick_params(axis='x', labelsize=PlotStyle.TICKLABELSIZE, pad=5, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='y', labelsize=PlotStyle.TICKLABELSIZE, pad=20, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='z', labelsize=PlotStyle.TICKLABELSIZE, pad=30, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=PlotStyle.LEGEND_FONT_SIZE)

    

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['tick']['inward_factor'] = 0.25
        axis._axinfo['tick']['outward_factor'] = 0.0
        lw = axis._axinfo['tick'].get('linewidth', 1.5)
        if isinstance(lw, dict):
            axis._axinfo['tick']['linewidth'] = {True: 2.5, False: 2.5}
        else:
            axis._axinfo['tick']['linewidth'] = 2.5

    inset_center = (x_prime, x_prime, x_prime)
    inset_halfwidth = 0.30
    ix0, iy0, iz0 = inset_center
    ixlim = (ix0 - inset_halfwidth, ix0 + inset_halfwidth)
    iylim = (iy0 - inset_halfwidth, iy0 + inset_halfwidth)
    izlim = (iz0 - inset_halfwidth, iz0 + inset_halfwidth)

    def _clip(data, xlim, ylim, zlim):
        mask = ((data[:, 0] >= xlim[0]) & (data[:, 0] <= xlim[1]) &
                (data[:, 1] >= ylim[0]) & (data[:, 1] <= ylim[1]) &
                (data[:, 2] >= zlim[0]) & (data[:, 2] <= zlim[1]))
        return data[mask]

    from itertools import product, combinations
    connector_cube_color = 'grey'

    corners = np.array(list(product(ixlim, iylim, izlim)))
    for s, e in combinations(corners, 2):
        if np.sum(np.abs(s - e) > 1e-9) == 1:
            ax.plot(*zip(s, e), color=connector_cube_color, linewidth=2, linestyle='--')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    inset_ax = fig.add_axes([0.33, 0.62, 0.25, 0.25], projection='3d')

    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, transform=inset_ax.transAxes,
                     edgecolor=connector_cube_color, facecolor='none',
                     linewidth=2.0, zorder=10)
    fig.add_artist(rect)

    seam_i = _clip(seam, ixlim, iylim, izlim)
    if len(seam_i):
        inset_ax.plot(*seam_i.T, color='blue', linestyle='-', alpha=1.0, linewidth=4)
    for data_full, color in [(branch1, 'tab:orange'), (branch2, 'tab:green'), (branch3, 'tab:red')]:
        data_i = _clip(data_full, ixlim, iylim, izlim)
        if len(data_i):
            inset_ax.plot(*data_i.T, color=color, linewidth=4)
            for spine_name in ['top', 'bottom', 'left', 'right']:
                spine = inset_ax.spines[spine_name]
                spine.set_visible(True)
                spine.set_color(connector_cube_color)
                spine.set_linewidth(PlotStyle.AXIS_LINEWIDTH)

    inset_ax.set_xlim(ixlim)
    inset_ax.set_ylim(iylim)
    inset_ax.set_zlim(izlim)
    inset_ax.view_init(azim=75)
    inset_ax.set_xticklabels([]); inset_ax.set_yticklabels([]); inset_ax.set_zticklabels([])
    inset_ax.set_xlabel(''); inset_ax.set_ylabel(''); inset_ax.set_zlabel('')

    for spine_axis in (inset_ax.xaxis, inset_ax.yaxis, inset_ax.zaxis):
        spine_axis.pane.set_edgecolor('black')
        spine_axis.pane.fill = False
    inset_ax.patch.set_alpha(0.55)

    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.lines as mlines

    def data_to_fig(point3d):
        x2, y2, _ = proj3d.proj_transform(*point3d, ax.get_proj())
        disp = ax.transData.transform((x2, y2))
        return tuple(fig.transFigure.inverted().transform(disp))

    def _dist2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    cube_corners_3d = list(product(ixlim, iylim, izlim))
    cube_corners_2d = [data_to_fig(c) for c in cube_corners_3d]

    inset_bbox = inset_ax.get_position()
    inset_corners_2d = [
        (inset_bbox.x0, inset_bbox.y0), (inset_bbox.x1, inset_bbox.y0),
        (inset_bbox.x0, inset_bbox.y1), (inset_bbox.x1, inset_bbox.y1),
    ]

    cube_center_2d = np.mean(cube_corners_2d, axis=0)
    near_inset_corners = sorted(inset_corners_2d, key=lambda c: _dist2(c, cube_center_2d))[:2]

    used_cube_idx = set()
    for ic in near_inset_corners:
        ranked = sorted(range(len(cube_corners_2d)), key=lambda i: _dist2(cube_corners_2d[i], ic))
        for idx in ranked:
            if idx not in used_cube_idx:
                used_cube_idx.add(idx)
                cx, cy = cube_corners_2d[idx]
                fig.add_artist(mlines.Line2D([cx, ic[0]], [cy, ic[1]], transform=fig.transFigure,
                                              color=connector_cube_color, linewidth=2.0, linestyle='--'))
                break

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/ci_seam_3d.png', dpi=300, bbox_inches='tight')

    plt.close('all')
    print(f"Saved 3D CI seam plots to {output_dir}")

def generate_ci_seam_3d_plot_no_inset(x_prime, output_dir, r0_range=(-3.0, 3.0), r0_marker=0.0,
                              axis_title_size=24, axis_axistitle_size_xyz=None,
                              axis_labelsize=None, ticklabelsize=None,
                              legend_font_size=12, enlarge_axis_labelling_size_by=2):
    """
    Plot the trivial CI seam and the three nontrivial-CI branch lines in
    full 3D (r1, r2, r3) space, showing all three branches converge on
    the trivial CI. Saves the main view (with zoomed inset) plus two
    rotated views into output_dir.
    """
    if axis_axistitle_size_xyz is None:
        axis_axistitle_size_xyz = 0.8 * axis_title_size
    if axis_labelsize is None:
        axis_labelsize = 0.8 * axis_axistitle_size_xyz
    if ticklabelsize is None:
        ticklabelsize = 0.8 * axis_labelsize
    if legend_font_size is None:
        legend_font_size = ticklabelsize + 2

    axis_axistitle_size_xyz = int(axis_axistitle_size_xyz + enlarge_axis_labelling_size_by)
    ticklabelsize = int(ticklabelsize + enlarge_axis_labelling_size_by)

    plt.close('all')

    r0_vals = np.linspace(r0_range[0], r0_range[1], 200)
    r1_vals = 3 * r0_vals - 2 * x_prime
    r2_vals = 4 * x_prime - 3 * r0_vals

    branch1 = np.stack([r1_vals, r1_vals, r2_vals], axis=1)
    branch2 = np.stack([r1_vals, r2_vals, r1_vals], axis=1)
    branch3 = np.stack([r2_vals, r1_vals, r1_vals], axis=1)

    CI_0 = np.array([x_prime, x_prime, x_prime])
    seam_t = np.linspace(r0_range[0], r0_range[1], 50)
    seam = np.stack([seam_t, seam_t, seam_t], axis=1)

    limit_range = 2.0
    xlim = (x_prime - limit_range, x_prime + limit_range)
    ylim = (x_prime - limit_range, x_prime + limit_range)
    zlim = (x_prime - limit_range, x_prime + limit_range)

    def _clip_range(data):
        mask = ((data[:, 0] >= xlim[0]) & (data[:, 0] <= xlim[1]) &
                (data[:, 1] >= ylim[0]) & (data[:, 1] <= ylim[1]) &
                (data[:, 2] >= zlim[0]) & (data[:, 2] <= zlim[1]))
        return data[mask]

    seam = _clip_range(seam)
    branch1 = _clip_range(branch1)
    branch2 = _clip_range(branch2)
    branch3 = _clip_range(branch3)

    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.1, 0.1, 0.85, 0.8])

    if len(seam) > 0:
        ax.plot(*seam.T, '-', color='blue', label='CI_T', linewidth=4)
    for branch, color, name in [(branch1, 'tab:orange', 'CI_1'),
                                  (branch2, 'tab:green', 'CI_2'),
                                  (branch3, 'tab:red', 'CI_3')]:
        if len(branch) > 0:
            ax.plot(*branch.T, color=color, linewidth=4, label=name)

    

    r1_m, r2_m = 3 * r0_marker - 2 * x_prime, 4 * x_prime - 3 * r0_marker
    for pt in [(r1_m, r1_m, r2_m), (r1_m, r2_m, r1_m), (r2_m, r1_m, r1_m)]:
        if (xlim[0] <= pt[0] <= xlim[1] and ylim[0] <= pt[1] <= ylim[1] and
            zlim[0] <= pt[2] <= zlim[1]):
            ax.scatter(*pt, c='black', s=400, marker='X', zorder=5)

    ax.set_xlabel('$r_1$', fontsize=axis_axistitle_size_xyz + 20, labelpad=25)
    ax.set_ylabel('$r_2$', fontsize=axis_axistitle_size_xyz + 20, labelpad=40)
    ax.set_zlabel('$r_3$', fontsize=axis_axistitle_size_xyz + 20, labelpad=45)
    ax.xaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.label.set_position((0.5, 1.1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.view_init(azim=75, elev=20)

    ax.tick_params(axis='x', labelsize=PlotStyle.TICKLABELSIZE, pad=5, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='y', labelsize=PlotStyle.TICKLABELSIZE, pad=20, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='z', labelsize=PlotStyle.TICKLABELSIZE, pad=30, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), fancybox=True, shadow=True, ncol=4, fontsize=PlotStyle.LEGEND_FONT_SIZE)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=4, fontsize=PlotStyle.LEGEND_FONT_SIZE)

    

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['tick']['inward_factor'] = 0.25
        axis._axinfo['tick']['outward_factor'] = 0.0
        lw = axis._axinfo['tick'].get('linewidth', 1.5)
        if isinstance(lw, dict):
            axis._axinfo['tick']['linewidth'] = {True: 2.5, False: 2.5}
        else:
            axis._axinfo['tick']['linewidth'] = 2.5
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/ci_seam_3d_no_inset.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Saved 3D CI seam plots to {output_dir}")

def generate_ci_orthogonal_plane_plot(ci_points, R_0, output_dir, R_thetas=None,
                                       show_coord_labels=False,
                                       axis_title_size=36, axis_axistitle_size_xyz=None,
                                       axis_labelsize=None, ticklabelsize=None,
                                       legend_font_size=None):
    #pasted from the new_bph w licis...
    if axis_axistitle_size_xyz is None:
        axis_axistitle_size_xyz = 0.8 * axis_title_size
    if axis_labelsize is None:
        axis_labelsize = 0.8 * axis_axistitle_size_xyz
    if ticklabelsize is None:
        ticklabelsize = 0.8 * axis_labelsize
    if legend_font_size is None:
        legend_font_size = ticklabelsize + 2

    basis1 = np.array([+2, -1, -1])
    basis2 = np.array([0, -1, +1])
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)

    def project_to_plane(point):
        rel_point = np.array(point) - np.array(R_0)
        return np.dot(rel_point, basis1), np.dot(rel_point, basis2)

    fig, ax = plt.subplots(figsize=(10, 8))

    trivial_ci = ci_points['trivial_lci']
    trivial_x, trivial_y = project_to_plane(trivial_ci)
    ax.scatter(trivial_x, trivial_y, c='red', s=200, label='CI_T', marker='o', zorder=5)

    aVx, aVa, x_shift = 1.0, 1.3, 0.1
    a_ratio = aVa / aVx
    x_prime_precise = (a_ratio / (a_ratio - 1)) * x_shift
    d_ci = 2 * np.sqrt(6) * abs(x_prime_precise)

    d_ci_circle = plt.Circle((trivial_x, trivial_y), d_ci, color='gray', fill=False, 
                             linestyle='--', linewidth=2, zorder=3, label=f'r = d_ci')
    ax.add_patch(d_ci_circle)

    all_points = [('CI_T', trivial_x, trivial_y, trivial_ci)]
    for i, ci in enumerate(ci_points['nontrivial_licis']):
        ci_x, ci_y = project_to_plane(ci)
        ax.scatter(ci_x, ci_y, c='blue', s=200, label=f'CI_{i+1}', marker='s', zorder=5)
        all_points.append((f'CI_{i+1}', ci_x, ci_y, ci))
    
    if show_coord_labels:
        lowest_y_point = min(all_points, key=lambda p: p[2])
        for label, x, y, coords_3d in all_points:
            coord_text = f'({coords_3d[0]:.3f}, {coords_3d[1]:.3f}, {coords_3d[2]:.3f})'
            va = 'bottom' if (label, x, y, coords_3d) == lowest_y_point else 'top'
            ax.text(x, y + (0.15 if va == 'bottom' else -0.15), coord_text, fontsize=9,
                    ha='center', va=va, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    if R_thetas is not None:
        r0_x, r0_y = project_to_plane(R_0)
        ax.scatter(r0_x, r0_y, c='blue', s=100, label='R_0 (center)', marker='*', zorder=6)
        R_thetas = np.asarray(R_thetas)
        circle_x, circle_y = zip(*[project_to_plane(pt) for pt in R_thetas])
        ax.plot(circle_x, circle_y, 'k--', alpha=0.5, label='Circle (R_thetas)', linewidth=4)

    ax.set_xlabel('Basis 1', fontsize=PlotStyle.AXIS_AXISTITLE_SIZE_XYZ * 0.90)
    ax.set_ylabel('Basis 2', fontsize=PlotStyle.AXIS_AXISTITLE_SIZE_XYZ * 0.90)
    # axis lkinewidth should be plotstyle.LINEWIDTH
    ax.spines['bottom'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.spines['top'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.spines['right'].set_linewidth(PlotStyle.AXIS_LINEWIDTH)

    ax.grid(True, alpha=PlotStyle.GRID_ALPHA, linewidth=PlotStyle.GRID_LINEWIDTH)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
               fancybox=True, shadow=True, ncol=3, fontsize=PlotStyle.LEGEND_FONT_SIZE * 0.75)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis='both', which='major', labelsize=PlotStyle.TICKLABELSIZE * 0.80)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/ci_points_orthogonal_plane.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CI orthogonal-plane plot to {output_dir}")

def generate_gamma_3d_plot(data, output_dir,
                            axis_title_size=24, axis_axistitle_size_xyz=None,
                            axis_labelsize=None, ticklabelsize=None,
                            legend_font_size=None, enlarge_axis_labelling_size_by=2):
    """
    Plot gamma_{2,3} and gamma_{3,2} as 3D surfaces over (R, theta).
    x = R (interatomic distance), y = theta/pi, z = gamma/pi.

    Parameters:
    - data: Dictionary containing loaded CI data with 'R_thetas', 'gamma', and 'theta_vals'
    - output_dir: directory to save the figure into
    """
    if 'R_thetas' not in data or 'gamma' not in data or 'theta_vals' not in data:
        print("Skipping gamma 3D plot - missing R_thetas, gamma, or theta_vals")
        return

    if axis_axistitle_size_xyz is None:
        axis_axistitle_size_xyz = 0.8 * axis_title_size
    if axis_labelsize is None:
        axis_labelsize = 0.8 * axis_axistitle_size_xyz
    if ticklabelsize is None:
        ticklabelsize = 0.8 * axis_labelsize
    if legend_font_size is None:
        legend_font_size = ticklabelsize + 2

    axis_axistitle_size_xyz = int(axis_axistitle_size_xyz + enlarge_axis_labelling_size_by)
    ticklabelsize = int(ticklabelsize + enlarge_axis_labelling_size_by)

    R_thetas = data['R_thetas']
    gamma = data['gamma']
    theta_vals = data['theta_vals']

    # Skip index 0 as in other plots
    # R_thetas has shape (5001, 3) - 3 different R values for each theta point
    R_vals = R_thetas[1:, :]  # Shape (5000, 3) - all R values
    theta = theta_vals[1:] / np.pi  # Shape (5000,)
    
    # Create meshgrid for surface plot
    # We'll use the 3 R values as the x-axis and theta as y-axis
    R_grid, Theta_grid = np.meshgrid(np.arange(3), theta, indexing='ij')
    
    # Get gamma values and create 2D arrays for surface
    gamma_23 = gamma[1, 2, 1:] / np.pi  # Shape (5000,)
    gamma_32 = gamma[2, 1, 1:] / np.pi  # Shape (5000,)
    
    # Reshape gamma values to create a surface (3 R values x theta points)
    # Since we only have one gamma value per theta, we'll replicate it across the 3 R values
    Z_23 = np.tile(gamma_23.reshape(-1, 1), (1, 3)).T  # Shape (3, 5000)
    Z_32 = np.tile(gamma_32.reshape(-1, 1), (1, 3)).T  # Shape (3, 5000)
    
    # For the x-axis, use the actual R values from R_thetas
    X_23 = R_vals.T  # Shape (3, 5000)
    X_32 = R_vals.T  # Shape (3, 5000)
    Y = np.tile(theta.reshape(1, -1), (3, 1))  # Shape (3, 5000)

    plt.close('all')
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.1, 0.1, 0.85, 0.8])

    # Plot as 3D surfaces
    ax.plot_surface(X_23, Y, Z_23, color=PlotStyle.COLORS['primary'],
                     alpha=0.85, linewidth=0, antialiased=True)
    ax.plot_surface(X_32, Y, Z_32, color=PlotStyle.COLORS['secondary'],
                     alpha=0.85, linewidth=0, antialiased=True)

    ax.set_xlabel('$R$', fontsize=axis_axistitle_size_xyz + 20, labelpad=25)
    ax.set_ylabel(r'$\theta/\pi$', fontsize=axis_axistitle_size_xyz + 20, labelpad=40)
    ax.set_zlabel(r'$\gamma/\pi$', fontsize=axis_axistitle_size_xyz + 20, labelpad=45)
    ax.xaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.label.set_position((0.5, 1.1))

    ax.view_init(azim=75, elev=20)

    ax.tick_params(axis='x', labelsize=PlotStyle.TICKLABELSIZE, pad=5, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='y', labelsize=PlotStyle.TICKLABELSIZE, pad=20, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='z', labelsize=PlotStyle.TICKLABELSIZE, pad=30, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # theta/pi spacing, same as 2D evolution plots

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PlotStyle.COLORS['primary'], label=r'$\gamma_{2,3}/\pi$'),
        Patch(facecolor=PlotStyle.COLORS['secondary'], label=r'$\gamma_{3,2}/\pi$'),
    ]
    ax.legend(handles=legend_handles, loc='best', fancybox=True, shadow=True,
              ncol=1, fontsize=PlotStyle.LEGEND_FONT_SIZE)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['tick']['inward_factor'] = 0.25
        axis._axinfo['tick']['outward_factor'] = 0.0
        lw = axis._axinfo['tick'].get('linewidth', 1.5)
        if isinstance(lw, dict):
            axis._axinfo['tick']['linewidth'] = {True: 2.5, False: 2.5}
        else:
            axis._axinfo['tick']['linewidth'] = 2.5

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/gamma_3d.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Saved gamma 3D plot to {output_dir}")

def generate_tau_3d_plot(data, output_dir,
                          axis_title_size=24, axis_axistitle_size_xyz=None,
                          axis_labelsize=None, ticklabelsize=None,
                          legend_font_size=None, enlarge_axis_labelling_size_by=2):
    """
    Plot |tau_{2,3}| and |tau_{3,2}| as 3D lines over (R, theta).
    x = R (interatomic distance), y = theta/pi, z = |tau|.

    Parameters:
    - data: Dictionary containing loaded CI data with 'R_thetas', 'tau', and 'theta_vals'
    - output_dir: directory to save the figure into
    """
    if 'R_thetas' not in data or 'tau' not in data or 'theta_vals' not in data:
        print("Skipping tau 3D plot - missing R_thetas, tau, or theta_vals")
        return

    if axis_axistitle_size_xyz is None:
        axis_axistitle_size_xyz = 0.8 * axis_title_size
    if axis_labelsize is None:
        axis_labelsize = 0.8 * axis_axistitle_size_xyz
    if ticklabelsize is None:
        ticklabelsize = 0.8 * axis_labelsize
    if legend_font_size is None:
        legend_font_size = ticklabelsize + 2

    axis_axistitle_size_xyz = int(axis_axistitle_size_xyz + enlarge_axis_labelling_size_by)
    ticklabelsize = int(ticklabelsize + enlarge_axis_labelling_size_by)

    R_thetas = data['R_thetas']
    tau = data['tau']
    theta_vals = data['theta_vals']

    # Skip index 0 as in other plots
    # R_thetas has shape (5001, 3) - 3 different R values for each theta point
    R_vals = R_thetas[1:, :]  # Shape (5000, 3) - all R values
    theta = theta_vals[1:] / np.pi  # Shape (5000,)
    
    # Create meshgrid for surface plot
    # We'll use the 3 R values as the x-axis and theta as y-axis
    R_grid, Theta_grid = np.meshgrid(np.arange(3), theta, indexing='ij')
    
    # Get tau values and create 2D arrays for surface
    tau_23 = np.abs(tau[1, 2, 1:])  # Shape (5000,)
    tau_32 = np.abs(tau[2, 1, 1:])  # Shape (5000,)
    
    # Reshape tau values to create a surface (3 R values x theta points)
    # Since we only have one tau value per theta, we'll replicate it across the 3 R values
    Z_23 = np.tile(tau_23.reshape(-1, 1), (1, 3)).T  # Shape (3, 5000)
    Z_32 = np.tile(tau_32.reshape(-1, 1), (1, 3)).T  # Shape (3, 5000)
    
    # For the x-axis, use the actual R values from R_thetas
    X_23 = R_vals.T  # Shape (3, 5000)
    X_32 = R_vals.T  # Shape (3, 5000)
    Y = np.tile(theta.reshape(1, -1), (3, 1))  # Shape (3, 5000)

    plt.close('all')
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.1, 0.1, 0.85, 0.8])

    # Plot as 3D surfaces
    ax.plot_surface(X_23, Y, Z_23, color=PlotStyle.COLORS['primary'],
                     alpha=0.85, linewidth=0, antialiased=True)
    ax.plot_surface(X_32, Y, Z_32, color=PlotStyle.COLORS['secondary'],
                     alpha=0.85, linewidth=0, antialiased=True)

    ax.set_xlabel('$R$', fontsize=axis_axistitle_size_xyz + 20, labelpad=25)
    ax.set_ylabel(r'$\theta/\pi$', fontsize=axis_axistitle_size_xyz + 20, labelpad=40)
    ax.set_zlabel(r'$|\tau|$', fontsize=axis_axistitle_size_xyz + 20, labelpad=45)
    ax.xaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.line.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    ax.zaxis.label.set_position((0.5, 1.1))

    ax.view_init(azim=75, elev=20)

    ax.tick_params(axis='x', labelsize=PlotStyle.TICKLABELSIZE, pad=5, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='y', labelsize=PlotStyle.TICKLABELSIZE, pad=20, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.tick_params(axis='z', labelsize=PlotStyle.TICKLABELSIZE, pad=30, length=20, width=PlotStyle.AXIS_LINEWIDTH)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PlotStyle.COLORS['primary'], label=r'$|\tau_{2,3}|$'),
        Patch(facecolor=PlotStyle.COLORS['secondary'], label=r'$|\tau_{3,2}|$'),
    ]
    ax.legend(handles=legend_handles, loc='best', fancybox=True, shadow=True,
              ncol=1, fontsize=PlotStyle.LEGEND_FONT_SIZE)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['tick']['inward_factor'] = 0.25
        axis._axinfo['tick']['outward_factor'] = 0.0
        lw = axis._axinfo['tick'].get('linewidth', 1.5)
        if isinstance(lw, dict):
            axis._axinfo['tick']['linewidth'] = {True: 2.5, False: 2.5}
        else:
            axis._axinfo['tick']['linewidth'] = 2.5

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/tau_3d.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Saved tau 3D plot to {output_dir}")

def process_ci_folder(base_dir, folder_name):
    """
    Process a single CI folder and generate improved plots.
    
    Parameters:
    - base_dir: Base directory containing CI folders
    - folder_name: Name of the CI folder to process
    """
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*60}")
    
    # Extract CI coordinates
    coords = extract_ci_coords(folder_name)
    d, coord1, coord2, coord3 = coords
    
    # Define paths - try multiple possible structures
    ci_folder = os.path.join(base_dir, folder_name)
    
    # Try different possible npy directory structures
    possible_npy_dirs = [
        os.path.join(ci_folder, 'd_0.001', 'npy'),  # Structure like d_0.001_-0.867_-0.867_1.733/d_0.001/npy
        os.path.join(ci_folder, 'npy'),              # Direct npy subfolder
        ci_folder                                   # Check if npy files are directly in the folder
    ]
    
    npy_dir = None
    for possible_dir in possible_npy_dirs:
        if os.path.exists(possible_dir):
            # Check if it contains .npy files
            npy_files = glob.glob(os.path.join(possible_dir, '*.npy'))
            if npy_files:
                npy_dir = possible_dir
                print(f"Found npy directory: {npy_dir}")
                break
    
    if npy_dir is None:
        print(f"Could not find npy directory in {ci_folder}")
        print("Tried these paths:")
        for possible_dir in possible_npy_dirs:
            print(f"  - {possible_dir}")
        return
    
    if not os.path.exists(npy_dir):
        print(f"Could not find npy directory in {ci_folder}")
        return
    
    # Create output directory
    coords_str = f"{coord1:.3f}_{coord2:.3f}_{coord3:.3f}"
    output_dir = os.path.join(base_dir, 'ci_analysis', coords_str, 'results')
    
    # Load data
    data = load_ci_data(npy_dir)
    
    if not data:
        print("No data loaded, skipping this folder")
        return
    
    # Generate plots
    generate_gamma_evolution_plot(data, output_dir)
    generate_tau_evolution_plot(data, output_dir)
    generate_gamma_heatmap(data, output_dir)
    generate_potential_components_plot(data, output_dir)
    generate_eigenvector_components_plot(data, output_dir)
    generate_eigenvalue_plot(data, output_dir)
    #generate_gamma_3d_plot(data, output_dir) # deprecated
    #generate_tau_3d_plot(data, output_dir) # deprecated
    generate_tau_abs_23_plot(data, output_dir)
    generate_tau_abs_32_plot(data, output_dir)

    generate_r_thetas_projections_plot(data, output_dir)

    
    print(f"Completed processing {folder_name}")
    
    return data

def main():
    """
    Main function to process all CI folders.
    """
    # Base directory (where the script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the four CI folders to process
    ci_folders = [
        'd_0.001_0.433_0.433_0.433',
        'd_0.001_-0.867_-0.867_1.733',
        'd_0.001_-0.867_1.733_-0.867',
        'd_0.001_1.733_-0.867_-0.867'
    ]
    
    # Map folder names to CI identifiers
    ci_identifiers = {
        'd_0.001_0.433_0.433_0.433': 'CI_T',
        'd_0.001_-0.867_-0.867_1.733': 'CI_1',
        'd_0.001_-0.867_1.733_-0.867': 'CI_2',
        'd_0.001_1.733_-0.867_-0.867': 'CI_3'
    }
    
    print("="*60)
    print("CI Analysis Plot Generator")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Processing {len(ci_folders)} CI folders")
    
    # Collect data from all folders for combined plots
    all_ci_data = {}
    
    # Process each folder
    num_processes = multiprocessing.cpu_count() - 5

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_ci_folder, [(base_dir, f) for f in ci_folders])

    # Map the multiprocessing results back into the all_ci_data dictionary
    for folder_name, data in zip(ci_folders, results):
        if data:  # Ensure data was actually loaded and returned
            ci_name = ci_identifiers[folder_name]
            all_ci_data[ci_name] = data
    
    # Generate combined comparison plots in 'together' folder
    if all_ci_data:
        together_dir = os.path.join(base_dir, 'ci_analysis', 'together')
        os.makedirs(together_dir, exist_ok=True)
        generate_combined_comparison_plot(all_ci_data, together_dir)

    
    # Generate 3D CI seam overview + orthogonal-plane plot
    trivial_lci = None
    nontrivial_licis = []
    for folder_name in ci_folders:
        d, coord1, coord2, coord3 = extract_ci_coords(folder_name)
        point = (coord1, coord2, coord3)
        if ci_identifiers[folder_name] == 'CI_T':
            trivial_lci = point
        else:
            nontrivial_licis.append(point)

    if trivial_lci is not None:
        x_prime = trivial_lci[0]  # coord1 == coord2 == coord3 for the trivial CI
        seam_3d_dir = os.path.join(base_dir, 'ci_analysis', '3d_and_CIs')

        generate_ci_seam_3d_plot(x_prime, seam_3d_dir)
        generate_ci_seam_3d_plot_no_inset(x_prime, seam_3d_dir)

        ci_points = {'trivial_lci': trivial_lci, 'nontrivial_licis': nontrivial_licis}
        generate_ci_orthogonal_plane_plot(ci_points, R_0=trivial_lci, output_dir=seam_3d_dir)


    print("\n" + "="*60)
    print("All processing completed!")
    print(f"Results saved to: {os.path.join(base_dir, 'ci_analysis')}")
    print("="*60)


if __name__ == '__main__':
    main()