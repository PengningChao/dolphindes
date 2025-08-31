"""
useful mathematical operations
"""

import numpy as np


def Sym(A):
    """
    Compute the symmetric Hermitian part of a matrix A
    TODO: add type hinting indicating that input is a generic matrix
    """
    return (A + A.T.conj()) / 2


def CRdot(v1: np.ndarray, v2: np.ndarray):
    """
    Computes the inner product of two complex vectors over a real field.
    In other words, the vectors have complex numbers but linear combination coefficients have to be real.
    This is the vector space for the complex QCQP constraints since Lagrangian multipliers are real.

    Parameters
    ----------
    v1 : np.ndarray
        vector1
    v2 : np.ndarray
        vector2

    Returns
    -------
    The inner product
    """
    return np.real(np.vdot(v1, v2))
