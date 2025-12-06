import pytest

from root_finding.bisection.bisection import bisection
from root_finding.exceptions import PreconditionError

def f(x):
    return x**2 - 2

def test_convergence():
    root = bisection(f, 1, 2, tol=1e-10)
    assert root == pytest.approx(2**(1/2), abs=1e-10)

def test_precondition_violation():
    with pytest.raises(PreconditionError):
        bisection(f, 2, 3)
    
def test_max_iterations_failure():
    with pytest.raises(RuntimeError):
        bisection(f, 1, 2, tol=1e-14, max_iter=10)