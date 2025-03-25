#!/usr/bin/env python3
"""
Utility functions for file operations, including directory organization.
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
    
    Returns:
    --------
    dict
        Dictionary mapping file extensions to subdirectories
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
    
    return subdirs

def get_file_path(results_dir, filename, file_type=None):
    """
    Get the appropriate path for a file based on its type.
    
    Parameters:
    -----------
    results_dir : str
        Path to the results directory
    filename : str
        Name of the file
    file_type : str, optional
        Type of the file (extension without the dot)
        If None, will be determined from the filename
    
    Returns:
    --------
    str
        Path to the file
    """
    # Create the directory structure if it doesn't exist
    subdirs = {
        'png': os.path.join(results_dir, 'plots'),
        'txt': os.path.join(results_dir, 'text'),
        'csv': os.path.join(results_dir, 'csv'),
        'npy': os.path.join(results_dir, 'numpy')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Determine file type if not provided
    if file_type is None:
        _, ext = os.path.splitext(filename)
        file_type = ext.lstrip('.')
    
    # Get the appropriate directory
    if file_type in subdirs:
        return os.path.join(subdirs[file_type], filename)
    else:
        return os.path.join(results_dir, filename)
