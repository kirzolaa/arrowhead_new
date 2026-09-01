"""
Scientific Plotting Library
============================
Provides 3D and 2D plotting classes with LaTeX-quality scientific visualization.
Inspired by LaTeX/tikz style plotting conventions.

Classes:
- Plot3D: For 3D scientific visualizations --->  future updates (maybe...)
- Plot2D: For 2D scientific visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os


# Global plotting style parameters
class PlotStyle:
    """Global style configuration for scientific plots."""
    
    # Font sizes (globally defined, not in functions)
    AXIS_TITLE_SIZE = 48
    AXIS_AXISTITLE_SIZE_XYZ = int(0.6 * AXIS_TITLE_SIZE)
    AXIS_LABELSIZE = int(0.9 * AXIS_AXISTITLE_SIZE_XYZ)
    TICKLABELSIZE = int(0.9 * AXIS_LABELSIZE)
    LEGEND_FONT_SIZE = TICKLABELSIZE
    
    # Line and grid settings
    AXIS_LINEWIDTH = 2
    GRID_ALPHA = 0.75
    GRID_LINEWIDTH = 1.0
    
    # 3D specific settings
    DEFAULT_AZIM = 75
    DEFAULT_ELEV = 30
    
    # Color scheme (scientific/publication quality)
    COLORS = {
        'primary': '#1f77b4',      # blue
        'secondary': '#ff7f0e',    # orange
        'tertiary': '#2ca02c',     # green
        'quaternary': '#d62728',   # red
        'quinary': '#9467bd',      # purple
        'black': '#000000',
        'gray': '#7f7f7f'
    }


class Plot2D:
    """
    2D Scientific Plotting Class
    
    Provides methods for creating high-quality 2D visualizations with
    publication-ready formatting, following LaTeX/tikz conventions.
    """
    
    def __init__(self, figsize=(10, 8), dpi=300):
        """
        Initialize the 2D plotter.
        
        Parameters:
        - figsize: Tuple of (width, height) in inches
        - dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        
    def create_figure(self):
        """Create a new 2D figure with proper styling."""
        plt.close('all')
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._apply_styling()
        return self.fig, self.ax
    
    def _apply_styling(self):
        """Apply global styling to the 2D axes."""
        self.ax.tick_params(
            axis='both',
            which='major',
            labelsize=PlotStyle.TICKLABELSIZE,
            length=8,
            width=PlotStyle.AXIS_LINEWIDTH,
            direction='in'
        )

        self.ax.grid(True, alpha=PlotStyle.GRID_ALPHA, linewidth=PlotStyle.GRID_LINEWIDTH)
        
        for spine in self.ax.spines.values():
            spine.set_linewidth(PlotStyle.AXIS_LINEWIDTH)
    
    def set_labels(self, xlabel, ylabel, use_latex=False):
        """
        Set axis labels with LaTeX formatting if enabled.
        
        Parameters:
        - xlabel, ylabel: Axis label strings
        - use_latex: Whether to use LaTeX formatting (wrap in $)
        """
        xlabel_has_latex = '$' in xlabel
        ylabel_has_latex = '$' in ylabel
        
        if use_latex and not xlabel_has_latex:
            xlabel = f'${xlabel}$'
        if use_latex and not ylabel_has_latex:
            ylabel = f'${ylabel}$'
        
        self.ax.set_xlabel(
            xlabel,
            fontsize=PlotStyle.AXIS_LABELSIZE,
            labelpad=10
        )
        self.ax.set_ylabel(
            ylabel,
            fontsize=PlotStyle.AXIS_LABELSIZE,
            labelpad=10
        )
    
    def set_limits(self, xlim=None, ylim=None):
        """
        Set axis limits.
        
        Parameters:
        - xlim: Tuple of (min, max) for x-axis (optional)
        - ylim: Tuple of (min, max) for y-axis (optional)
        """
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
    
    def plot(self, x, y, color=None, linewidth=4, label=None, 
             linestyle='-', alpha=1.0, marker=None):
        """
        Plot a 2D line.
        
        Parameters:
        - x, y: Arrays of coordinates
        - color: Color specification (uses default if None)
        - linewidth: Line width
        - label: Legend label
        - linestyle: Line style ('-', '--', ':', etc.)
        - alpha: Transparency (0-1)
        - marker: Marker style ('o', 's', etc.)
        """
        if color is None:
            color = PlotStyle.COLORS['primary']
        
        self.ax.plot(
            x, y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            alpha=alpha,
            marker=marker
        )
    
    def scatter(self, x, y, color=None, s=100, marker='o', label=None, alpha=1.0):
        """
        Plot 2D scatter points.
        
        Parameters:
        - x, y: Arrays of coordinates
        - color: Color specification
        - s: Marker size
        - marker: Marker style ('o', 's', '*', 'x', etc.)
        - label: Legend label
        - alpha: Transparency (0-1)
        """
        if color is None:
            color = PlotStyle.COLORS['primary']
        
        self.ax.scatter(
            x, y,
            c=color,
            s=s,
            marker=marker,
            label=label,
            alpha=alpha,
            zorder=5
        )
    
    def fill_between(self, x, y1, y2=None, color=None, alpha=0.3, label=None):
        """
        Fill area between curves.
        
        Parameters:
        - x: X coordinates
        - y1: First Y curve (or lower curve if y2 provided)
        - y2: Second Y curve (upper curve)
        - color: Fill color
        - alpha: Transparency
        - label: Legend label
        """
        if color is None:
            color = PlotStyle.COLORS['primary']
        
        if y2 is None:
            y2 = np.zeros_like(x)
        
        self.ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=label)
    
    def set_title(self, title, use_latex=False):
        """
        Set plot title.
        
        Parameters:
        - title: Title string
        - use_latex: Whether to use LaTeX formatting
        """
        # Check if title already contains LaTeX syntax ($...$)
        title_has_latex = '$' in title
        
        if use_latex and not title_has_latex:
            title = f'${title}$'
        
        self.ax.set_title(title, fontsize=PlotStyle.AXIS_TITLE_SIZE, pad=15)
    
    def add_legend(self, loc='upper center', fontsize=None):
        """
        Add legend to the plot.
        
        Parameters:
        - loc: Legend location ('best', 'upper right', etc.)
        - fontsize: Font size (uses global default if None)
        """
        if fontsize is None:
            fontsize = PlotStyle.LEGEND_FONT_SIZE
        self.ax.legend(loc=loc, bbox_to_anchor=(0.5, 1.20),
          fancybox=True, shadow=True, ncol=4, fontsize=fontsize)
    
    def set_tick_locator(self, axis='both', multiple=0.5):
        """
        Set tick locators for consistent spacing.
        
        !!! pay attention to this as the theta axis was wrong in the plots but I fixed it !!!

        Parameters:
        - axis: Which axis to set ('x', 'y', or 'both')
        - multiple: Spacing between ticks
        """
        if axis in ['x', 'both']:
            self.ax.xaxis.set_major_locator(MultipleLocator(multiple))
        if axis in ['y', 'both']:
            self.ax.yaxis.set_major_locator(MultipleLocator(multiple))
    
    def set_tick_formatter(self, axis='both', format_str='%.1f'):
        """
        Set tick formatter to control decimal places.
        
        Parameters:
        - axis: Which axis to set ('x', 'y', or 'both')
        - format_str: Format string for tick labels (e.g., '%.1f' for 1 decimal)
        """
        if axis in ['x', 'both']:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(format_str))
        if axis in ['y', 'both']:
            self.ax.yaxis.set_major_formatter(FormatStrFormatter(format_str))
        
    def set_aspect(self, aspect='equal'):
        """
        Set aspect ratio of the plot.
        
        Parameters:
        - aspect: Aspect ratio ('equal', 'auto', or numeric value)
        """
        self.ax.set_aspect(aspect, adjustable='box')
    
    def create_subplots(self, rows, cols, figsize=None, sharex=False, sharey=False):
        """
        Create a figure with multiple subplots.
        
        Parameters:
        - rows: Number of subplot rows
        - cols: Number of subplot columns
        - figsize: Figure size (uses default if None)
        - sharex: bool or {'none','all','row','col'} — passed straight to plt.subplots
        - sharey: bool or {'none','all','row','col'} — passed straight to plt.subplots
        """
        plt.close('all')
        if figsize is None:
            figsize = (self.figsize[0] * cols, self.figsize[1] * rows)
        
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex, sharey=sharey)
        self.axes = self.axes.flatten() if rows * cols > 1 else [self.axes]
        
        for ax in self.axes:
            original_ax = self.ax
            self.ax = ax
            self._apply_styling()
            self.ax = original_ax
        
        return self.fig, self.axes

    def add_hline(self, y, color=None, linestyle='--', linewidth=2, alpha=0.7, label=None):
        """Add a horizontal line."""
        if color is None:
            color = PlotStyle.COLORS['gray']
        self.ax.axhline(y=y, color=color, linestyle=linestyle, 
                       linewidth=linewidth, alpha=alpha, label=label)
    
    def add_vline(self, x, color=None, linestyle='--', linewidth=2, alpha=0.7, label=None):
        """Add a vertical line."""
        if color is None:
            color = PlotStyle.COLORS['gray']
        self.ax.axvline(x=x, color=color, linestyle=linestyle,
                       linewidth=linewidth, alpha=alpha, label=label)
    
    def save(self, filename, save_dir=None, bbox_inches='tight'):
        """
        Save the figure to file.
        
        Parameters:
        - filename: Name of the output file
        - save_dir: Directory to save to (uses current if None)
        - bbox_inches: Bounding box setting
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename
        
        self.fig.tight_layout()

        plt.savefig(filepath, dpi=self.dpi, bbox_inches=bbox_inches)
        print(f"2D plot saved to {filepath}")
        plt.close()
    
    def show(self):
        """Display the plot interactively."""
        plt.show()


if __name__ == '__main__':
    """
    Test section using sqrt(x) function as a dummy test.
    Demonstrates both 3D and 2D plotting capabilities.
    """
    
    print("=" * 60)
    print("Testing Scientific Plotter Library")
    print("=" * 60)
    
    # Test data: sqrt(x) function
    x = np.linspace(0, 10, 100)
    y_sqrt = np.sqrt(x)
    
    # ========================================
    # Test 2D Plotting
    # ========================================
    print("\nTesting 2D Plotter...")
    plotter_2d = Plot2D(figsize=(12, 7), dpi=150)
    fig, ax = plotter_2d.create_figure()
    
    # Plot sqrt(x)
    plotter_2d.plot(x, y_sqrt, color=PlotStyle.COLORS['primary'], 
                   linewidth=3, label=r'\sqrt{x}')
    
    # Add reference line y = x/3 for comparison
    plotter_2d.plot(x, x/3, color=PlotStyle.COLORS['secondary'], 
                   linewidth=2, linestyle='--', label='x/3')
    
    # Styling
    plotter_2d.set_labels('x', r'\sqrt{x}', use_latex=True)
    plotter_2d.set_title('Square Root Function Test', use_latex=True)
    plotter_2d.set_limits(xlim=(0, 6), ylim=(0, 2.5))
    plotter_2d.add_legend(loc='best')
    plotter_2d.set_tick_locator(axis='x', multiple=1.0)
    plotter_2d.set_tick_locator(axis='y', multiple=0.5)
    
    # Save test plot
    plotter_2d.save('test_2d_sqrt.png', save_dir='test_outputs')
    
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check 'test_outputs' directory for generated plots.")
    print("=" * 60)
