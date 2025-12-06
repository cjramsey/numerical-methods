import pytest

from root_finding.secant.secant import secant

def f(x):
    return x**2 - 2

def test_convergence():
    # Secant function is currently returning array of approximations
    # Will give caller optionality to return scalar or array in future
    root = secant(f, 1, 2, tol=1e-10)
    assert root[-1] == pytest.approx(2**(1/2), 1e-10)

def test_max_iterations_failure():
    with pytest.raises(RuntimeError):
        secant(f, 1, 2, tol=1e-14, max_iter=5)