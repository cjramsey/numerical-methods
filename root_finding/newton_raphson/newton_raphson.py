from typing import Callable

def newton_raphson(f: Callable[[float], float], f_prime: Callable[[float], float], 
                   p0: float, tol: float=1e-8, max_iter: int=100) -> float:
    '''
    Find a root of f(x) = 0 using the Newton-Raphson method.
    Order of convergence is quadratic.
    Error measured by absolute difference between successive approximations.

    Parameters:
        f: Callable[[float], float]
            Continuous function f(x) whose root is being found.
        f_prime: Callable[[float], float]
            Derivative of f(x), f'(x).
        p0: float
            First initial approximation.
        tol: float
            Absolute tolerance for convergence.
        max_iter: int
            Maximum number of iterations.

    Returns:
        float:
           Approximation to the root of f(x).
    '''

    iterations = 1

    while iterations < max_iter:
        p = p0 - f(p0) / f_prime(p0)

        if abs(p - p0) < tol:
            return p
        
        iterations += 1

        p0 = p
    
    raise RuntimeError(f"Method failed to converge within max_iter={max_iter} iterations.")