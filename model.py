import numpy as np

def szr_system(t, y, P, beta, alpha, delta, zeta):
    """The SZR differential equations."""
    S, Z, R = y
    dSdt = P - beta * S * Z - delta * S
    dZdt = beta * S * Z + zeta * R - alpha * S * Z
    dRdt = delta * S + alpha * S * Z - zeta * R
    return [dSdt, dZdt, dRdt]

def get_jacobian_dfe(P, beta, alpha, delta, zeta):
    """
    Returns the Jacobian matrix evaluated at the 
    Disease-Free Equilibrium (S=P/delta, Z=0, R=0).
    """
    S_eq = P / delta
    #The matrix we derived on paper
    J = np.array([
        [P - delta, -beta * S_eq, 0],
        [0, (beta - alpha) * S_eq, zeta],
        [delta, alpha * S_eq, -zeta]
    ])
    return J