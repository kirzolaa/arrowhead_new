# Orthogonal Vector Visualization System

A flexible Python tool for generating and visualizing complex orthogonal vector configurations with advanced plotting capabilities.

## Overview

This system generates a single R vector using scalar formulas and provides comprehensive visualization options for both single and multiple vectors. It supports various projection methods, parameter ranges, and visualization styles.

## Features

- Single R vector generation using scalar formulas
- Multiple vector generation with parameter ranges
- Perfect orthogonal circle generation in the plane orthogonal to the x=y=z line
- Enhanced 3D visualization with color-coded axes, coordinate labels, and data-driven scaling
- 2D projection visualization with improved clarity
- Endpoints-only plotting option
- Configurable visualization parameters
- Command-line interface with extensive options
- Configuration file support
- Circle/sphere pattern generation examples

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd arrowhead

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Generate a single vector
python generalized/main.py -R 1 1 1 -d 2 -a 1.047

# Generate multiple vectors with distance range
python generalized/main.py -R 0 0 0 --d-range 1 5 3 -a 0.7854

# Generate multiple vectors with angle range
python generalized/main.py -R 0 0 0 -d 1.5 --theta-range 0 10 3.14159

# Endpoints-only plotting
python generalized/main.py --endpoints true
```

### Command-line Options

```
-R, --origin X Y Z    : Set the origin vector R_0 coordinates (default: 0 0 0)
-d, --distance VALUE  : Set the distance parameter (default: 1)
--d-range START STEPS END : Generate multiple vectors with distance values from START to END with STEPS steps
-a, --angle VALUE     : Set the angle parameter in radians (default: π/4)
--theta-range START STEPS END : Generate multiple vectors with angle values from START to END with STEPS steps
--endpoints true/false : Only plot the endpoints of vectors, not the arrows (default: false)
--no-r0-plane        : Do not show the R_0 plane projection
--no-legend          : Do not show the legend
--no-grid            : Do not show the grid
--save-plots         : Save plots to files instead of displaying them
--output-dir DIR     : Directory to save plots to (default: 'plots')
--config FILE        : Load configuration from a JSON file
--save-config FILE   : Save current configuration to a JSON file
```

## Package Structure

- `vector_utils.py`: Vector generation and component calculation functions
- `visualization.py`: Comprehensive visualization functions for 2D and 3D plotting
- `config.py`: Configuration management and serialization
- `main.py`: Command-line interface and main program logic
- `example_circle.py`: Example generating a sphere-like pattern using orthogonal vectors
- `example_circle_xy.py`: Example generating a traditional circle in the XY plane
- `example_orthogonal_circle.py`: Example with improved visualization of orthogonal vectors
- `CIRCLE_EXAMPLES.md`: Documentation for the circle examples

## Vector Generation

The system generates a single R vector using the following scalar formulas:

```
R_1 = R_0 + d * (cos(theta))*sqrt(2/3)
R_2 = R_0 + d * (cos(theta)/sqrt(3) + sin(theta))/sqrt(2)
R_3 = R_0 + d * (sin(theta) - cos(theta)/sqrt(3))/sqrt(2)
R = R_1 + R_2 + R_3 - 2 * R_0
```

Where:
- `R_0` is the origin vector
- `d` is the distance parameter
- `theta` is the angle parameter in radians
- `R_1`, `R_2`, `R_3` are component vectors
- `R` is the resulting vector

## Visualization

The system provides several visualization functions with enhanced features:

- `plot_vectors_3d`: Plot vectors in 3D space with color-coded axes and coordinate labels
- `plot_vectors_2d_projection`: Plot 2D projections (xy, xz, yz, r0 planes)
- `plot_all_projections`: Plot all projections of a single vector
- `plot_multiple_vectors_3d`: Plot multiple vectors in 3D with optional endpoints-only mode
- `plot_multiple_vectors_2d`: Plot 2D projections of multiple vectors
- `plot_multiple_vectors`: Plot multiple vectors in all projections

### Enhanced Visualization Features

The latest version includes several visualization enhancements for improved clarity and spatial understanding:

- **Color-coded Axes**: The X (red), Y (green), and Z (blue) axes are color-coded for easy identification
- **Coordinate Labels**: Integer coordinate values are displayed along each axis, color-matched to the axis color
- **Tick Marks**: Small tick marks along each axis for better spatial reference
- **Data-driven Scaling**: The axis limits are dynamically adjusted based on the actual data points
- **Equal Aspect Ratio**: The 3D plots maintain an equal aspect ratio for accurate spatial representation
- **Buffer Zones**: Small buffer zones are added around the data points for better visibility

These enhancements significantly improve the visual representation of the orthogonal vectors, making it easier to understand their spatial relationships and properties.

## Circle Examples

Three circle examples demonstrate different visualization approaches:

1. `example_circle.py`: Generates points using orthogonal vector formulas, creating a sphere-like pattern
2. `example_circle_xy.py`: Creates a traditional circle in the XY plane
3. `example_orthogonal_circle.py`: Similar to the first example but with improved visualization

See `CIRCLE_EXAMPLES.md` for detailed information about these examples.

## Configuration

The `VectorConfig` class provides configuration management with JSON serialization support. Use the `--save-config` and `--config` options to save and load configurations.
