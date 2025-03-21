#!/usr/bin/env python3
import os
import sys

# Print current directory
print(f"Current directory: {os.getcwd()}")

# Print Python path
print(f"Python path: {sys.path}")

# Try to import the module
try:
    import energy_gap_analysis
    print("Successfully imported energy_gap_analysis module")
    print(f"Module location: {energy_gap_analysis.__file__}")
except ImportError as e:
    print(f"Error importing energy_gap_analysis module: {e}")

# Try to import the module with an absolute import
try:
    sys.path.append('/home/zoli/arrowhead')
    import energy_gap_analysis
    print("Successfully imported energy_gap_analysis module with absolute path")
    print(f"Module location: {energy_gap_analysis.__file__}")
except ImportError as e:
    print(f"Error importing energy_gap_analysis module with absolute path: {e}")
