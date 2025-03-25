#!/usr/bin/env python3
"""
Script to organize the results directory by file type.
Creates subdirectories for different file types and moves files accordingly.
"""

import os
import shutil
import glob

def organize_results_directory(results_dir):
    """
    Organize the results directory by file type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    """
    # Create subdirectories if they don't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Move files to appropriate subdirectories
    for file_path in glob.glob(os.path.join(results_dir, '*')):
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip('.')
        
        # Skip if extension not in our list
        if ext not in subdirs:
            continue
        
        # Get destination directory
        dest_dir = subdirs[ext]
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Move file to destination directory
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"Moved {filename} to {os.path.relpath(dest_path, results_dir)}")

def main():
    """
    Main function to organize the results directory.
    """
    # Get the results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # Check if the directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return
    
    print(f"Organizing results directory: {results_dir}")
    organize_results_directory(results_dir)
    print("Organization complete.")

if __name__ == "__main__":
    main()
