import pytest


def test_addition():
    assert 1 + 1 == 2


def test_subtraction():
    assert 5 - 2 == 3


# Function to calculate the square of a number
def square_numbers(num):
    return num * num


# Parametrize decorator to test the square function with different inputs
@pytest.mark.parametrize("input_value, expected_output", [(2, 4), (-3, 9), (0, 0)])
def test_square(input_value, expected_output):
    assert square_numbers(input_value) == expected_output
