import numpy as np
import pytest

from linear_systems.gauss_elimination.gauss_elimination import gauss_elimination
from linear_systems.exceptions import NoUniqueSolutionError

def test_3x3_int_solution():
    A = np.array(([1, 2, 3], [2, 3, 4], [3, 4, 6]))
    b = np.array([1, 1, 1])
    x = np.array([-1, 1, 0])
    assert gauss_elimination(A, b) == pytest.approx(x)

def test_infinite_solution():
    A = np.array(([1, 2], [1, 2]))
    b = np.array([1, 2])
    with pytest.raises(NoUniqueSolutionError):
        gauss_elimination(A, b)