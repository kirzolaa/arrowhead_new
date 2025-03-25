"""
Orthogonal Vectors Generator and Visualizer

This package provides tools for creating and visualizing three orthogonal vectors
from a given origin point.
"""

from .vector_utils import create_orthogonal_vectors, check_vector_components
from .visualization import (
    plot_vectors_3d, 
    plot_vectors_2d_projection, 
    plot_all_projections
)
from .config import VectorConfig, default_config

__all__ = [
    'create_orthogonal_vectors',
    'check_vector_components',
    'plot_vectors_3d',
    'plot_vectors_2d_projection',
    'plot_all_projections',
    'VectorConfig',
    'default_config'
]
