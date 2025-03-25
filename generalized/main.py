#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import os
import sys
import importlib.util

from vector_utils import create_orthogonal_vectors, check_vector_components, generate_R_vector
from visualization import plot_vectors_3d, plot_vectors_2d_projection, plot_all_projections, plot_multiple_vectors
from config import VectorConfig, default_config

# Import the ArrowheadMatrixAnalyzer class from the arrowhead.py module
arrowhead_path = os.path.join(os.path.dirname(__file__), 'example_use', 'arrowhead_matrix', 'arrowhead.py')
spec = importlib.util.spec_from_file_location("arrowhead", arrowhead_path)
arrowhead = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arrowhead)
ArrowheadMatrixAnalyzer = arrowhead.ArrowheadMatrixAnalyzer

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generalized Arrowhead Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Vector generation command
    vector_parser = subparsers.add_parser('vector', help='Generate and visualize orthogonal vectors')
    
    # Vector parameters
    vector_parser.add_argument('--origin', '-R', type=float, nargs=3, default=[0, 0, 0],
                        help='Origin vector R_0 (x y z)')
    
    # Distance parameter with range support
    d_group = vector_parser.add_mutually_exclusive_group()
    d_group.add_argument('--distance', '-d', type=float, default=1,
                        help='Distance parameter d')
    d_group.add_argument('--d-range', type=float, nargs=3, metavar=('START', 'STEPS', 'END'),
                        help='Distance parameter range: start steps end')
    
    # Angle parameter with range support
    theta_group = vector_parser.add_mutually_exclusive_group()
    theta_group.add_argument('--angle', '-a', '--theta', type=float, default=math.pi/4,
                        help='Angle parameter theta in radians')
    theta_group.add_argument('--theta-range', type=float, nargs=3, metavar=('START', 'STEPS', 'END'),
                        help='Angle parameter range: start steps end')
    
    # Perfect circle generation option
    vector_parser.add_argument('--perfect', action='store_true',
                        help='Use perfect circle generation method')
    
    # Visualization parameters
    vector_parser.add_argument('--plot-type', type=str, choices=['3d', '2d'], default='3d',
                        help='Type of plot to generate (3d or 2d)')
    vector_parser.add_argument('--title', type=str, default=None,
                        help='Title for the plot')
    vector_parser.add_argument('--no-show', action='store_false', dest='show_plot',
                        help='Prevents the plot from being displayed interactively')
    vector_parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the plot')
    vector_parser.add_argument('--no-enhanced-visualization', action='store_false', dest='enhanced_visualization',
                        help='Disables enhanced visualization features')
    vector_parser.add_argument('--axis-colors', type=str, nargs=3, default=['r', 'g', 'b'],
                        help='Custom colors for the X, Y, and Z axes as three space-separated values')
    vector_parser.add_argument('--no-coordinate-labels', action='store_false', dest='show_coordinate_labels',
                        help='Disables coordinate labels on the axes')
    vector_parser.add_argument('--no-equal-aspect-ratio', action='store_false', dest='equal_aspect_ratio',
                        help='Disables equal aspect ratio for 3D plots')
    vector_parser.add_argument('--buffer-factor', type=float, default=0.2,
                        help='Sets the buffer factor for axis limits. Default: 0.2')
    
    # Existing visualization parameters
    vector_parser.add_argument('--no-r0-plane', action='store_false', dest='show_r0_plane',
                        help='Do not show the R_0 plane projection')
    vector_parser.add_argument('--no-legend', action='store_false', dest='show_legend',
                        help='Do not show the legend')
    vector_parser.add_argument('--no-grid', action='store_false', dest='show_grid',
                        help='Do not show the grid')
    vector_parser.add_argument('--endpoints', type=lambda x: x.lower() == 'true', default=False,
                        help='Only plot the endpoints of vectors, not the arrows')
    
    # Output parameters
    vector_parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying them')
    vector_parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots to')
    vector_parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    vector_parser.add_argument('--save-config', type=str,
                        help='Save configuration to file')
    
    # Arrowhead matrix command
    arrowhead_parser = subparsers.add_parser('arrowhead', help='Generate and analyze arrowhead matrices')
    
    # Matrix parameters
    arrowhead_parser.add_argument('--r0', type=float, nargs=3, default=[0, 0, 0],
                        help='Origin vector (x, y, z)')
    arrowhead_parser.add_argument('--d', type=float, default=0.5,
                        help='Distance parameter')
    arrowhead_parser.add_argument('--theta-start', type=float, default=0,
                        help='Starting theta value in radians')
    arrowhead_parser.add_argument('--theta-end', type=float, default=2*np.pi,
                        help='Ending theta value in radians')
    arrowhead_parser.add_argument('--theta-steps', type=int, default=72,
                        help='Number of theta values to generate matrices for')
    arrowhead_parser.add_argument('--coupling', type=float, default=0.1,
                        help='Coupling constant for off-diagonal elements')
    arrowhead_parser.add_argument('--omega', type=float, default=1.0,
                        help='Angular frequency for the energy term h*ω')
    arrowhead_parser.add_argument('--size', type=int, default=4,
                        help='Size of the matrix to generate')
    arrowhead_parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results')
    arrowhead_parser.add_argument('--load-only', action='store_true',
                        help='Only load existing results and create plots')
    arrowhead_parser.add_argument('--plot-only', action='store_true',
                        help='Only create plots from existing results')
    arrowhead_parser.add_argument('--perfect', action='store_true', default=True,
                        help='Whether to use perfect circle generation method')
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def display_help():
    """
    Display detailed help information
    """
    help_text = """
    Generalized Arrowhead Framework
    =======================================
    
    This tool provides a unified interface for generating orthogonal vectors and arrowhead matrices.
    
    Basic Usage:
    -----------
    python main.py vector                      # Generate and visualize orthogonal vectors
    python main.py arrowhead                   # Generate and analyze arrowhead matrices
    python main.py help                        # Show detailed help
    
    Vector Generation Command:
    ------------------------
    python main.py vector [OPTIONS]            # Generate and visualize orthogonal vectors
    
    Vector Parameters:
    ----------------
    -R, --origin X Y Z    : Set the origin vector R_0 coordinates (default: 0 0 0)
    -d, --distance VALUE  : Set the distance parameter (default: 1)
    --d-range START STEPS END : Generate multiple vectors with distance values from START to END with STEPS steps
    -a, --angle, --theta VALUE : Set the angle parameter in radians (default: π/4)
    --theta-range START STEPS END : Generate multiple vectors with angle values from START to END with STEPS steps
    --perfect            : Use perfect circle generation method with normalized basis vectors
    
    Vector Visualization Options:
    --------------------------
    --plot-type          : Specifies the type of plot, either "3d" or "2d" (default: "3d")
    --title              : Specifies the title of the plot
    --no-show            : Prevents the plot from being displayed interactively
    --save-path          : Specifies the path to save the plot
    --no-enhanced-visualization : Disables enhanced visualization features
    --axis-colors        : Specifies custom colors for the X, Y, and Z axes as three space-separated values
    --no-coordinate-labels : Disables coordinate labels on the axes
    --no-equal-aspect-ratio : Disables equal aspect ratio for 3D plots
    --buffer-factor VALUE : Sets the buffer factor for axis limits (default: 0.2)
    --no-r0-plane        : Do not show the R_0 plane projection
    --no-legend          : Do not show the legend
    --no-grid            : Do not show the grid
    --endpoints true/false : Only plot the endpoints of vectors, not the arrows (default: false)
    
    Vector Output Options:
    -------------------
    --save-plots         : Save plots to files instead of displaying them
    --output-dir DIR     : Directory to save plots to (default: 'plots')
    --config FILE        : Load configuration from a JSON file
    --save-config FILE   : Save current configuration to a JSON file
    
    Vector Examples:
    -------------
    # Generate vector with origin at (1,1,1), distance 2, and angle π/3
    python main.py vector -R 1 1 1 -d 2 -a 1.047
    
    # Generate multiple vectors with distance range from 1 to 3 with 5 steps
    python main.py vector -R 0 0 0 --d-range 1 5 3 -a 0.7854
    
    # Generate multiple vectors with angle range from 0 to π with 10 steps
    python main.py vector -R 0 0 0 -d 1.5 --theta-range 0 10 3.14159
    
    # Generate a perfect circle orthogonal to the x=y=z line
    python main.py vector -R 0 0 0 -d 1 --theta-range 0 36 6.28 --perfect
    
    # Save plots to a custom directory
    python main.py vector -R 0 0 2 --save-plots --output-dir my_plots
    
    # Load configuration from a file
    python main.py vector --config my_config.json
    
    # Use custom plot type and title
    python main.py vector -R 0 0 0 -d 1.5 --plot-type 2d --title "Custom Plot Title"
    
    # Customize visualization with axis colors
    python main.py vector -R 0 0 0 -d 1 --axis-colors blue green red
    
    Arrowhead Matrix Command:
    ----------------------
    python main.py arrowhead [OPTIONS]         # Generate and analyze arrowhead matrices
    
    Arrowhead Matrix Parameters:
    ------------------------
    --r0 X Y Z           : Origin vector coordinates (default: 0 0 0)
    --d VALUE            : Distance parameter (default: 0.5)
    --theta-start VALUE  : Starting theta value in radians (default: 0)
    --theta-end VALUE    : Ending theta value in radians (default: 2π)
    --theta-steps VALUE  : Number of theta values to generate matrices for (default: 72)
    --coupling VALUE     : Coupling constant for off-diagonal elements (default: 0.1)
    --omega VALUE        : Angular frequency for the energy term h*ω (default: 1.0)
    --size VALUE         : Size of the matrix to generate (default: 4)
    --perfect            : Use perfect circle generation method (default: True)
    
    Arrowhead Matrix Options:
    ----------------------
    --output-dir DIR     : Directory to save results (default: './results')
    --load-only          : Only load existing results and create plots
    --plot-only          : Only create plots from existing results
    
    Arrowhead Matrix Examples:
    -----------------------
    # Generate matrices with default parameters
    python main.py arrowhead
    
    # Generate matrices with custom parameters
    python main.py arrowhead --r0 1 1 1 --d 0.8 --theta-steps 36 --size 6
    
    # Generate matrices with perfect circle generation
    python main.py arrowhead --perfect --theta-steps 12
    
    # Only create plots from existing results
    python main.py arrowhead --plot-only --output-dir my_results
    
    # Load existing results and create plots
    python main.py arrowhead --load-only --output-dir my_results
    """
    print(help_text)
    sys.exit(0)

def run_vector_command(args):
    """
    Run the vector generation and visualization command
    """
    # Load configuration
    if args.config:
        config = VectorConfig.load_from_file(args.config)
        # Generate a single R vector
        R_0 = config.R_0
        perfect = getattr(config, 'perfect', False)
        
        # Check if theta is a single value or multiple values
        if isinstance(config.theta, list):
            # Multiple values, use create_orthogonal_vectors with num_points
            R = create_orthogonal_vectors(R_0, config.d, len(config.theta), perfect=perfect)
        else:
            # Single value, use generate_R_vector
            R = generate_R_vector(R_0, config.d, config.theta, perfect=perfect)
        
        # Print vector information
        print("R_0:", R_0)
        print("R:", R)
        print("Perfect circle generation:", perfect)
        
        # Check vector components
        components = check_vector_components(R_0, R, config.d, config.theta, perfect=perfect)
        print("\nVector components:")
        for key, value in components.items():
            print(f"{key}: {value}")
        
        # Plot the vector
        plots = plot_all_projections(
            R_0, R,
            show_r0_plane=config.show_r0_plane,
            figsize_3d=config.figsize_3d,
            figsize_2d=config.figsize_2d
        )
        
        # Save or show the plots
        if args.save_plots:
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save each plot
            for name, (fig, _) in plots.items():
                filename = os.path.join(args.output_dir, f"{name}.png")
                fig.savefig(filename)
                print(f"Saved plot to {filename}")
        else:
            # Show the plots
            plt.show()
    else:
        # Create configuration from command line arguments
        R_0 = np.array(args.origin)
        
        # Handle distance range
        if args.d_range is not None:
            d_start, d_steps, d_end = args.d_range
            d_values = np.linspace(d_start, d_end, int(d_steps))
        else:
            d_values = [args.distance]
        
        # Handle theta range
        if args.theta_range is not None:
            theta_start, theta_steps, theta_end = args.theta_range
            theta_values = np.linspace(theta_start, theta_end, int(theta_steps))
        else:
            theta_values = [args.angle]
        
        # Generate all combinations of d and theta
        all_vectors = []
        
        # If there's only one value for each parameter, we can use either method
        if len(d_values) == 1 and len(theta_values) == 1:
            # Single vector case
            d = d_values[0]
            theta = theta_values[0]
            
            # Create a vector for this combination
            R = generate_R_vector(R_0, d, theta, perfect=args.perfect)
            all_vectors.append((d, theta, R))
            
            # Print vector information
            print(f"\nR_0: {R_0}, d: {d}, theta: {theta}")
            print(f"R: {R}")
            print(f"Perfect circle generation: {args.perfect}")
            
            # Check vector components
            components = check_vector_components(R_0, R, d, theta, perfect=args.perfect)
            print("Vector components:")
            for key, value in components.items():
                if key != "Combined R":
                    print(f"{key}: {value}")
        elif len(theta_values) > 1 and len(d_values) == 1:
            # Multiple angles, single distance - can use create_orthogonal_vectors for the circle
            d = d_values[0]
            
            # Get the start and end theta values from the theta range
            start_theta = theta_values[0]
            end_theta = theta_values[-1]
            
            # Generate the circle of vectors
            vectors = create_orthogonal_vectors(R_0, d, len(theta_values), perfect=args.perfect, 
                                               start_theta=start_theta, end_theta=end_theta)
            
            # Add each vector to the list
            for i, theta in enumerate(theta_values):
                R = vectors[i]
                all_vectors.append((d, theta, R))
                
                # Print vector information
                print(f"\nR_0: {R_0}, d: {d}, theta: {theta}")
                print(f"R: {R}")
                print(f"Perfect circle generation: {args.perfect}")
                
                # Check vector components
                components = check_vector_components(R_0, R, d, theta, perfect=args.perfect)
                print("Vector components:")
                for key, value in components.items():
                    if key != "Combined R":
                        print(f"{key}: {value}")
        else:
            # Multiple combinations - generate each vector individually
            for d in d_values:
                for theta in theta_values:
                    # Create a vector for this combination
                    R = generate_R_vector(R_0, d, theta, perfect=args.perfect)
                    all_vectors.append((d, theta, R))
                    
                    # Print vector information
                    print(f"\nR_0: {R_0}, d: {d}, theta: {theta}")
                    print(f"R: {R}")
                    print(f"Perfect circle generation: {args.perfect}")
                    
                    # Check vector components
                    components = check_vector_components(R_0, R, d, theta, perfect=args.perfect)
                    print("Vector components:")
                    for key, value in components.items():
                        if key != "Combined R":
                            print(f"{key}: {value}")
        
        # Save configuration if requested
        if args.save_config:
            config = VectorConfig(
                R_0=args.origin,
                d=args.distance if args.d_range is None else d_values.tolist(),
                theta=args.angle if args.theta_range is None else theta_values.tolist(),
                show_r0_plane=args.show_r0_plane,
                show_legend=args.show_legend,
                show_grid=args.show_grid,
                perfect=args.perfect
            )
            config.save_to_file(args.save_config)
        
        # Plot all vectors
        if len(all_vectors) == 1:
            # Only one vector, use the standard plotting function
            d, theta, R = all_vectors[0]
            plots = plot_all_projections(
                R_0, R,
                show_r0_plane=args.show_r0_plane,
                figsize_3d=(10, 8),
                figsize_2d=(8, 8)
            )
            # Note: endpoints_only is not applicable for single vector in plot_all_projections
        else:
            # Multiple vectors, create a special plot
            plots = plot_multiple_vectors(
                R_0, all_vectors,
                show_r0_plane=args.show_r0_plane,
                figsize_3d=(12, 10),
                figsize_2d=(10, 10),
                endpoints_only=args.endpoints
            )
        
        # Save or show the plots
        if args.save_plots:
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save each plot
            for name, (fig, _) in plots.items():
                filename = os.path.join(args.output_dir, f"{name}.png")
                fig.savefig(filename)
                print(f"Saved plot to {filename}")
        else:
            # Show the plots
            plt.show()

def run_arrowhead_command(args):
    """
    Run the arrowhead matrix generation and analysis command
    """
    # Create the analyzer
    analyzer = ArrowheadMatrixAnalyzer(
        R_0=tuple(args.r0),
        d=args.d,
        theta_start=args.theta_start,
        theta_end=args.theta_end,
        theta_steps=args.theta_steps,
        coupling_constant=args.coupling,
        omega=args.omega,
        matrix_size=args.size,
        perfect=args.perfect,
        output_dir=args.output_dir
    )
    
    if args.plot_only:
        # Only create plots
        analyzer.load_results()
        analyzer.create_plots()
        analyzer.plot_r_vectors()
    elif args.load_only:
        # Load results and create plots
        analyzer.load_results()
        analyzer.create_plots()
        analyzer.plot_r_vectors()
    else:
        # Run the complete analysis
        analyzer.run_all()

def main():
    """
    Main function
    """
    # Check for detailed help command
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        display_help()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Dispatch to the appropriate command handler
    if args.command == 'vector':
        run_vector_command(args)
    elif args.command == 'arrowhead':
        run_arrowhead_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
