"""Example test module."""

import pytest


@pytest.mark.parametrize("input, expected", [(1, 2), (3, 4)])
def test_example(inputs, expected):
    """A sample test :param inputs: :param expected: :return:"""
    assert inputs + 1 == expected
