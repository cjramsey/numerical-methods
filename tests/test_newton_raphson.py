import pytest

from root_finding.newton_raphson.newton_raphson import newton_raphson

def f(x):
    return x**2 - 2

def f_prime(x):
    return 2*x

def test_convergence():
    root = newton_raphson(f, f_prime, 1, tol=1e-10)
    assert root == pytest.approx(2**(1/2), 1e-10)
    
def test_max_iterations_failure():
    with pytest.raises(RuntimeError):
        newton_raphson(f, f_prime, 1, tol=1e-14, max_iter=4)