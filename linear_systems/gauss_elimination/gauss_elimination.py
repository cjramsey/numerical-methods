import numpy as np

from linear_systems.exceptions import DimensionError, NoUniqueSolutionError

def gauss_elimination(A: np.ndarray, b: np.array) -> np.array:
    '''
    Solve system of linear equations of the form Ax = b using Gauss Elimination.

    Parameters:
        A: np.ndarray[n][n]
            Matrix of coefficients of system of linear equations.
        B: np.array[n]
            Column vector of right-hand side of system of linear equations.
        
    Returns:
        x: np.array[n]
            Solution to linear system Ax = b.
    '''
    
    # Matrix A must be square, n equations in n unknowns
    if (A.ndim != 2):
        raise DimensionError(f"A must be a 2-dimensional matrix, got {A.ndim}")
    if (A.shape[0] != A.shape[1]):
        raise DimensionError(f"A must be a square matrix, got {A.shape}")
    
    n = A.shape[0]

    # b must have same number of elements as number of equations
    if (n != b.shape[0]):
        raise DimensionError(f"Length of b must equal dimension of A.")
    
    b = b.reshape(-1, 1)
    augmented = np.concatenate((A, b), axis=1, dtype=np.float64)

    for i in range(0, n - 1):

        # Find smallest row index with element in ith column non-zero
        p = None
        for j in range(i, n):
            if augmented[j, i] != 0:
                p = j
                break

            # If entire column is zero from ith row index onwards, there are infintely many solutions
            if p is None:
                raise NoUniqueSolutionError("No unique solution exists.")
        
        # Swap pivot so that it lies on the diagonal
        if p != i:
            augmented[p], augmented[i] = augmented[i], augmented[p]

        # Transform into row echelon form
        for j in range(i + 1, n):

            m = augmented[j, i] / augmented[i, i]

            augmented[j] -= m * augmented[i]    
    
    # If bottom row is all zero, infinitely many solutions exists
    if augmented[n - 1, n - 1] == 0:
        raise NoUniqueSolutionError("No unique solution exists.")
    
    x = np.zeros(n)

    # Backward substitution
    x[n - 1] = augmented[n - 1, n] / augmented[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (augmented[i, n] - np.sum(augmented[i, i+1:n-1] * x[i+1:n-1])) / augmented[i, i]
    
    return x


