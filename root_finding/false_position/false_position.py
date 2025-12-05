from typing import Callable

from exceptions import PreconditionError

def false_position(f: Callable[[float], float], a: float, 
                   b: float, tol: float=1e-8, max_iter: int=100) -> list:
    '''
    Find a root of f(x) = 0  on the interval [a,b] using false position (regular-falsi) method.
    Order of convergence is linear.
    Error measured by absolute difference between successive approximations.

    Preconditions:
        f(a) and f(b) must have opposite signs.

    Parameters:
        f: Callable[[float], float]
            Continuous function f(x) whose root is being found.
        a: float
            Start point of interval [a,b].
        b: float
            End point of interval [a,b].
        tol: float
            Absolute tolerance for convergence.
        max_iter: int
            Maximum number of iterations.

    Returns:
        list:
            List of consecutive approximations to the root of f(x).
    '''

     # Check preconditon for algorithm to hold
    if f(a) * f(b) > 0:
        raise PreconditionError("Bisection requires f(a) and f(b) to be opposite signs.")

    Pn = [a, b]
    iterations = 0

    # Iterate until method is converged or iterations exceed max_iter
    while (iterations <= max_iter):

        p = (a * f(b) - b * f(a)) / (f(b) - f(a))
        Pn.append(p)

        if abs(Pn[-1] - Pn[-2] < tol):
            # Return array to allow for Aitken's delta-squared process (Exponential extrapolation)
            return Pn

        # Move a or b depending on signage
        if f(a) * f(p) > 0:
            a = p
        else:
            b = p
        
        iterations += 1
    
    if iterations > max_iter:
        raise RuntimeError(f"Method failed to converge within max_iter={max_iter} iterations.")
