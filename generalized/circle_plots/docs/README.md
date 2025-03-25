# Documentation for Orthogonal Vectors Generator

This directory contains LaTeX-based documentation for the Orthogonal Vectors Generator and Visualizer.

## Files

- `orthogonal_vectors.tex`: Main LaTeX document with mathematical explanations and implementation details
- `Makefile`: Utility for compiling the LaTeX document

## Compiling the Documentation

To compile the documentation into a PDF, you need to have LaTeX installed on your system. Then, you can use the provided Makefile:

```bash
# Navigate to the docs directory
cd docs

# Compile the documentation
make

# Clean up auxiliary files
make clean
```

## Documentation Contents

The documentation covers:

1. Mathematical formulation of the orthogonal vectors
2. Implementation details in Python
3. Visualization techniques (3D and 2D projections)
4. Usage examples
5. Mathematical properties of the vectors

## Requirements for Compilation

- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- The following LaTeX packages:
  - amsmath
  - amssymb
  - graphicx
  - hyperref
  - listings
  - xcolor
  - tikz
  - float
