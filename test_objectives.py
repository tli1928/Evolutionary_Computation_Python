"""
Ryan Huang, Anthony Gemma, Eric Wu, Teng li
DS3500 / Resource Allocation with Evolutionary Computing
HW4
3/28
"""


import pytest
from numpy import genfromtxt
from objectives import *

@pytest.fixture
def solutions():
    # Test files
    test1 = genfromtxt('test1.csv', delimiter=',')
    test2 = genfromtxt('test2.csv', delimiter=',')
    test3 = genfromtxt('test3.csv', delimiter=',')
    return [test1, test2, test3]

def test_overallocation(solutions):
    assert overallocation(solutions[0]) == 37, "Incorrect overallocated TAs"
    assert overallocation(solutions[1]) == 41, "Incorrect overallocated TAs"
    assert overallocation(solutions[2]) == 23, "Incorrect overallocated TAs"


def test_conflicts(solutions):
    assert conflicts(solutions[0]) == 8, "Incorrect time conflicts"
    assert conflicts(solutions[1]) == 5, "Incorrect time conflicts"
    assert conflicts(solutions[2]) == 2, "Incorrect time conflicts"


def test_undersupport(solutions):
    assert undersupport(solutions[0]) == 1, "Incorrect undersupported TAs"
    assert undersupport(solutions[1]) == 0, "Incorrect undersupported TAs"
    assert undersupport(solutions[2]) == 7, "Incorrect undersupported TAs"


def test_unwilling(solutions):
    assert unwilling(solutions[0]) == 53, "Incorrect unwilling TAs"
    assert unwilling(solutions[1]) == 58, "Incorrect unwilling TAs"
    assert unwilling(solutions[2]) == 43, "Incorrect unwilling TAs"


def test_unpreferred(solutions):
    assert unpreferred(solutions[0]) == 15, "Incorrect unpreferred TAs"
    assert unpreferred(solutions[1]) == 19, "Incorrect unpreferred TAs"
    assert unpreferred(solutions[2]) == 10, "Incorrect unpreferred TAs"

