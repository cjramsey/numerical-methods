import pytest

from root_finding.false_position.false_position import false_position
from root_finding.exceptions import PreconditionError

def f(x):
    return x**2 - 2

def test_convergence():
    # False position function is currently returning array of approximations
    # Will give caller optionality to return scalar or array in future
    root = false_position(f, 1, 2, tol=1e-10)
    assert root[-1] == pytest.approx(2**(1/2), 1e-10)

def test_precondition_violation():
    with pytest.raises(PreconditionError):
        false_position(f, 2, 3)
    
def test_max_iterations_failure():
    with pytest.raises(RuntimeError):
        false_position(f, 1, 2, tol=1e-14, max_iter=10)