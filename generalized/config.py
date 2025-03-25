#!/usr/bin/env python3
import numpy as np
import math
import json
import os

class VectorConfig:
    """
    Configuration class for orthogonal vector generation and visualization
    """
    def __init__(self, 
                 R_0=(0, 0, 0), 
                 d=1, 
                 theta=math.pi/4,
                 plot_type="3d",
                 title=None,
                 show_plot=True,
                 save_path=None,
                 enhanced_visualization=True,
                 axis_colors=["r", "g", "b"],
                 show_coordinate_labels=True,
                 equal_aspect_ratio=True,
                 buffer_factor=0.2,
                 show_r0_plane=True,
                 figsize_3d=(10, 8),
                 figsize_2d=(8, 8),
                 show_legend=True,
                 show_grid=True,
                 perfect=False):
        """
        Initialize the configuration
        
        Parameters:
        R_0 (tuple or list): The origin vector
        d (float): The distance parameter
        theta (float): The angle parameter in radians
        plot_type (str): Type of plot, either "3d" or "2d"
        title (str): Title of the plot
        show_plot (bool): Whether to display the plot interactively
        save_path (str): Path to save the plot
        enhanced_visualization (bool): Whether to use enhanced visualization features
        axis_colors (list): Custom colors for the X, Y, and Z axes
        show_coordinate_labels (bool): Whether to show coordinate labels on the axes
        equal_aspect_ratio (bool): Whether to use equal aspect ratio for 3D plots
        buffer_factor (float): Buffer factor for axis limits
        show_r0_plane (bool): Whether to show the R_0 plane projection
        figsize_3d (tuple): Figure size for 3D plot
        figsize_2d (tuple): Figure size for 2D plots
        show_legend (bool): Whether to show the legend
        show_grid (bool): Whether to show the grid
        perfect (bool): Whether to use perfect circle generation method
        """
        self.R_0 = np.array(R_0)
        self.d = d
        self.theta = theta
        self.plot_type = plot_type
        self.title = title
        self.show_plot = show_plot
        self.save_path = save_path
        self.enhanced_visualization = enhanced_visualization
        self.axis_colors = axis_colors
        self.show_coordinate_labels = show_coordinate_labels
        self.equal_aspect_ratio = equal_aspect_ratio
        self.buffer_factor = buffer_factor
        self.show_r0_plane = show_r0_plane
        self.figsize_3d = figsize_3d
        self.figsize_2d = figsize_2d
        self.show_legend = show_legend
        self.show_grid = show_grid
        self.perfect = perfect
    
    def to_dict(self):
        """
        Convert the configuration to a dictionary
        
        Returns:
        dict: Dictionary representation of the configuration
        """
        return {
            'origin': self.R_0.tolist(),
            'd': self.d,
            'theta': self.theta,
            'plot_type': self.plot_type,
            'title': self.title,
            'show_plot': self.show_plot,
            'save_path': self.save_path,
            'enhanced_visualization': self.enhanced_visualization,
            'axis_colors': self.axis_colors,
            'show_coordinate_labels': self.show_coordinate_labels,
            'equal_aspect_ratio': self.equal_aspect_ratio,
            'buffer_factor': self.buffer_factor,
            'show_r0_plane': self.show_r0_plane,
            'figsize_3d': self.figsize_3d,
            'figsize_2d': self.figsize_2d,
            'show_legend': self.show_legend,
            'show_grid': self.show_grid,
            'perfect': self.perfect
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a configuration from a dictionary
        
        Parameters:
        config_dict (dict): Dictionary containing configuration parameters
        
        Returns:
        VectorConfig: Configuration object
        """
        return cls(
            R_0=config_dict.get('origin', (0, 0, 0)),
            d=config_dict.get('d', 1),
            theta=config_dict.get('theta', math.pi/4),
            plot_type=config_dict.get('plot_type', '3d'),
            title=config_dict.get('title', None),
            show_plot=config_dict.get('show_plot', True),
            save_path=config_dict.get('save_path', None),
            enhanced_visualization=config_dict.get('enhanced_visualization', True),
            axis_colors=config_dict.get('axis_colors', ['r', 'g', 'b']),
            show_coordinate_labels=config_dict.get('show_coordinate_labels', True),
            equal_aspect_ratio=config_dict.get('equal_aspect_ratio', True),
            buffer_factor=config_dict.get('buffer_factor', 0.2),
            show_r0_plane=config_dict.get('show_r0_plane', True),
            figsize_3d=config_dict.get('figsize_3d', (10, 8)),
            figsize_2d=config_dict.get('figsize_2d', (8, 8)),
            show_legend=config_dict.get('show_legend', True),
            show_grid=config_dict.get('show_grid', True),
            perfect=config_dict.get('perfect', False)
        )
    
    def save_to_file(self, filename):
        """
        Save the configuration to a JSON file
        
        Parameters:
        filename (str): Path to the output file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Load a configuration from a JSON file
        
        Parameters:
        filename (str): Path to the input file
        
        Returns:
        VectorConfig: Configuration object
        """
        if not os.path.exists(filename):
            print(f"Warning: Config file {filename} not found. Using default configuration.")
            return cls()
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# Default configuration
default_config = VectorConfig()
