from typing import Callable

def secant(f: Callable[[float], float], p0: float, 
           p1: float, tol: float=1e-8, max_iter: int=100) -> list:
    '''
    Find a root of f(x) = 0 using secant method.
    Order of convergence is faster than linear but slower than quadratic.
    Error measured by absolute difference between successive approximations.

    Parameters:
        f: Callable[[float], float]
            Continuous function f(x) whose root is being found.
        p0: float
            First initial approximation.
        p1: float
            Second initial approximation.
        tol: float
            Absolute tolerance for convergence.
        max_iter: int
            Maximum number of iterations.

    Returns:
        list:
            List of consecutive approximations to the root of f(x).
    '''

    Pn = [p0, p1]
    iterations = 0

    while iterations < max_iter:
         p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
         Pn.append(p)

         if abs(p - p1) < tol:
              return Pn
         
         iterations += 1

         p0 = p1
         p1 = p
    
    raise RuntimeError(f"Method failed to converge within max_iter={max_iter} iterations.")
         

