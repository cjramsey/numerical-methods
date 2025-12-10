import numpy as np

from linear_systems.exceptions import ImpossibleFactorizationError

def LU_factorization(A: np.matrix) -> tuple[np.matrix, np.matrix]:
    '''
    Factorize n-by-n matrix A with LU factorization.
    L is a lower triangular matrix.
    U is a unit upper triangular matrix (1 along diagonal).

    Parameters:
        A: np.matrix
            n-by-n matrix being factorized. 

    Returns:
        L: np.matrix
            n-by-b lower triangular matrix such as A = LU. 
        U: np.matrix
            n-by-n unit upper triangular matrix such that A = LU.
    '''

    n = A.shape[0]

    # Initialize L with 0s, U has 1s along the diagonal
    L = np.zeros_like(A)
    U = np.eye(n)

    L[0, 0] = A[0, 0] / U[0, 0]

    if L[0, 0] * U[0, 0] == 0:
        raise ImpossibleFactorizationError("Factorization impossible.")
    
    # Calculate first row of U and first column of L
    for j in range(1, n):
        U[0, j] = A[0, j] / L[0, 0]
        L[j, 0] = A[j, 0] / U[0, 0]

    for i in range(1, n - 1):

        # Calculate diagonal element of L 
        L[i, i] = A[i, i]
        for k in range(0, i - 1):
            L[i, i] -= L[i, k] * U[k, i]
        
        if L[i, i] * U[i, i] == 0:
            raise ImpossibleFactorizationError("Factorization impossible.")
    
        for j in range(i + 1, n):

            # Calculate ith row of U
            U[i, j] = A[i, j]
            for m in range(0, i - 1):
                U[i, j] -= L[i, m] * U[m, j]
            U[i, j] /= L[i, i]

            # Calculate ith column of L
            L[j, i] = A[j, i]
            for m in range(0, i - 1):
                L[j, i] -= L[j, m] * U[m, i]
            L[j, i] /= U[i, i]
    
    # Bottom right element of L
    L[n - 1, n - 1] = A[n - 1, n - 1]
    for k in range(0, n - 1):
        L[n - 1, n - 1] -= L[n - 1, k] * U[k, n - 1] 

    return [L, U]
