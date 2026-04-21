import numpy as np

def check_stability(J):
    """
    Analyzes the 2x2 interaction sub-matrix of the Jacobian.
    Returns Trace and Determinant.
    """
    sub_matrix = J[1:3, 1:3]
    
    trace = np.trace(sub_matrix)
    determinant = np.linalg.det(sub_matrix)
    
    is_stable = (trace < 0 and determinant > 0)
    
    return trace, determinant, is_stable 