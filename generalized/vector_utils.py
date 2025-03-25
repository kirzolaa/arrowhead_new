#!/usr/bin/env python3
import numpy as np

def create_perfect_orthogonal_vectors(R_0=(0, 0, 0), d=1, theta=0):
    """
    Create a single R vector that forms a perfect circle orthogonal to the x=y=z line
    using normalized basis vectors.
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    theta (float): The angle parameter in radians, default is 0
    
    Returns:
    numpy.ndarray: The resulting R vector orthogonal to the x=y=z line
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])  # First basis vector
    basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
    
    # Normalize the basis vectors
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # Create a point at distance d from the origin in the plane spanned by basis1 and basis2
    R = R_0 + d * (np.cos(theta) * basis1 + np.sin(theta) * basis2)
    
    return R

def create_perfect_orthogonal_circle(R_0=(0, 0, 0), d=1, num_points=36, start_theta=0, end_theta=2*np.pi):
    """
    Create multiple vectors that form a perfect circle orthogonal to the x=y=z line
    using normalized basis vectors.
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    num_points (int): The number of points to generate, default is 36
    start_theta (float): Starting angle in radians, default is 0
    end_theta (float): Ending angle in radians, default is 2*pi
    
    Returns:
    numpy.ndarray: Array of shape (num_points, 3) containing the generated vectors
    """
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Generate equally spaced angles between start_theta and end_theta
    thetas = np.linspace(start_theta, end_theta, num_points, endpoint=False)
    
    # Initialize the array to store the vectors
    vectors = np.zeros((num_points, 3))
    
    # Generate vectors for each angle
    for i, theta in enumerate(thetas):
        vectors[i] = create_perfect_orthogonal_vectors(R_0, d, theta)
    
    return vectors

def generate_R_vector(R_0, d, theta, perfect=False):
    """
    Generate a single R vector orthogonal to the x=y=z line
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector
    d (float): The distance parameter
    theta (float): The angle parameter in radians
    perfect (bool): If True, use the perfect circle generation method, default is False
    
    Returns:
    numpy.ndarray: The resulting R vector orthogonal to the x=y=z line
    """
    if perfect:
        return create_perfect_orthogonal_vectors(R_0, d, theta)
    
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Define the basis vectors orthogonal to the (1,1,1) direction
    basis1 = np.array([1, -1/2, -1/2])  # First basis vector
    basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
    
    # Calculate the components using the basis vectors
    component1 = d * np.cos(theta) * np.sqrt(2/3) * basis1
    component2 = d * (np.cos(theta)/np.sqrt(3) + np.sin(theta))/np.sqrt(2) * basis1
    component3 = d * (np.sin(theta) - np.cos(theta)/np.sqrt(3))/np.sqrt(2) * basis2 * np.sqrt(2)
    
    # Calculate the R vector using the scalar formula
    R = R_0 + component1 + component2 + component3
    
    return R

def create_orthogonal_vectors(R_0=(0, 0, 0), d=1, num_points=36, perfect=False, start_theta=0, end_theta=2*np.pi):
    """
    Create multiple vectors that form a circle orthogonal to the x=y=z line
    
    Parameters:
    R_0 (tuple or numpy.ndarray): The origin vector, default is (0, 0, 0)
    d (float): The distance parameter, default is 1
    num_points (int): The number of points to generate, default is 36
    perfect (bool): If True, use the perfect circle generation method, default is False
    start_theta (float): Starting angle in radians, default is 0
    end_theta (float): Ending angle in radians, default is 2*pi
    
    Returns:
    numpy.ndarray: Array of shape (num_points, 3) containing the generated vectors
    """
    if perfect:
        return create_perfect_orthogonal_circle(R_0, d, num_points, start_theta, end_theta)
    
    # Convert R_0 to numpy array for vector operations
    R_0 = np.array(R_0)
    
    # Generate equally spaced angles
    thetas = np.linspace(start_theta, end_theta, num_points, endpoint=False)
    
    # Initialize the array to store the vectors
    vectors = np.zeros((num_points, 3))
    
    # Generate vectors for each angle
    for i, theta in enumerate(thetas):
        vectors[i] = generate_R_vector(R_0, d, theta)
    
    return vectors

def calculate_components(R_0, d, theta, perfect=False):
    """
    Calculate the individual components used to generate the R vector
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    d (float): The distance parameter
    theta (float): The angle parameter in radians
    perfect (bool): If True, use the perfect circle generation method, default is False
    
    Returns:
    tuple: The component vectors
    """
    if perfect:
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([1, -1/2, -1/2])  # First basis vector
        basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
        
        # Normalize the basis vectors
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Calculate the components using the normalized basis vectors
        component1 = d * np.cos(theta) * basis1
        component2 = d * np.sin(theta) * basis2
        
        return component1, component2
    else:
        # Define the basis vectors orthogonal to the (1,1,1) direction
        basis1 = np.array([1, -1/2, -1/2])  # First basis vector
        basis2 = np.array([0, -1/2, 1/2])   # Second basis vector
        
        # Calculate the components using the basis vectors
        component1 = d * np.cos(theta) * np.sqrt(2/3) * basis1
        component2 = d * (np.cos(theta)/np.sqrt(3) + np.sin(theta))/np.sqrt(2) * basis1
        component3 = d * (np.sin(theta) - np.cos(theta)/np.sqrt(3))/np.sqrt(2) * basis2 * np.sqrt(2)
        
        return component1, component2, component3

def check_vector_components(R_0, R, d, theta, perfect=False):
    """
    Calculate and return the individual components of the R vector for verification
    
    Parameters:
    R_0 (numpy.ndarray): The origin vector
    R (numpy.ndarray): The generated R vector
    d (float): The distance parameter
    theta (float): The angle parameter in radians
    perfect (bool): If True, use the perfect circle generation method, default is False
    
    Returns:
    dict: Dictionary containing the component vectors and the combined vector
    """
    # Calculate the individual components
    if perfect:
        component1, component2 = calculate_components(R_0, d, theta, perfect=True)
        
        # Calculate the expected combined vector
        R_expected = R_0 + component1 + component2
        
        # Calculate the difference between expected and actual R
        diff = np.linalg.norm(R - R_expected)
        
        # Check orthogonality to the (1,1,1) direction
        unit_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized (1,1,1) vector
        orthogonality = np.abs(np.dot(R - R_0, unit_111))
        
        # Check if the distance from R_0 is exactly d
        distance = np.linalg.norm(R - R_0)
        distance_error = np.abs(distance - d)
        
        return {
            "Component 1 (cos term)": component1,
            "Component 2 (sin term)": component2,
            "Combined R": R,
            "Verification (should be close to 0)": diff,
            "Orthogonality to (1,1,1) (should be close to 0)": orthogonality,
            "Distance from R_0": distance,
            "Distance Error (should be close to 0)": distance_error
        }
    else:
        component1, component2, component3 = calculate_components(R_0, d, theta)
        
        # Calculate the expected combined vector
        R_expected = R_0 + component1 + component2 + component3
        
        # Calculate the difference between expected and actual R
        diff = np.linalg.norm(R - R_expected)
        
        # Check orthogonality to the (1,1,1) direction
        unit_111 = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized (1,1,1) vector
        orthogonality = np.abs(np.dot(R - R_0, unit_111))
        
        return {
            "Component 1": component1,
            "Component 2": component2,
            "Component 3": component3,
            "Combined R": R,
            "Verification (should be close to 0)": diff,
            "Orthogonality to (1,1,1) (should be close to 0)": orthogonality
        }
