from typing import Callable

from exceptions import PreconditionError

def bisection(f: Callable[[float], float], a: float, b: float, 
              tol: float=1e-8, max_iter: int=100) -> float:
    '''
    Find a root of f(x) = 0  on the interval [a,b] using bisection method.
    Order of convergence is linear.
    Error measured by length of current interval = b - a.

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
        float:
            Approximation to the root of f(x).
    '''

    # Check preconditon for algorithm to hold
    if f(a) * f(b) > 0:
        raise PreconditionError("Bisection requires f(a) and f(b) to be opposite signs.")
    
    iterations = 1
    
    while (iterations <= max_iter):
        # Python integers cannot overflow
        p = (a + b) / 2

        # Root found
        if f(p) == 0 or (b - a) / 2 < tol:
            return p
        
        iterations += 1

        # Move a or b depending on signage
        if f(a) * f(p) > 0:
            a = p
        else:
            b = p
    
    raise RuntimeError(f"Method failed to converge within max_iter={max_iter} iterations.")
        
