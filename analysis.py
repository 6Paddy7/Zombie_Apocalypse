import numpy as np

def check_stability(J):
    """
    Analyzes the 2x2 interaction sub-matrix of the Jacobian.
    Returns Trace and Determinant.
    """
    #Extract the bottom-right 2x2 sub-matrix
    sub_matrix = J[1:3, 1:3]
    
    trace = np.trace(sub_matrix)
    determinant = np.linalg.det(sub_matrix)
    
    #Logic: Survival if Trace < 0 and Det > 0
    is_stable = (trace < 0 and determinant > 0)
    
    return trace, determinant, is_stable 