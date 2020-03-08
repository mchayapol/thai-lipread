# -*- coding: utf-8 -*-

import pytest
from vec2viz.skeleton import fib

__author__ = "Chayapol Moemeng"
__copyright__ = "Chayapol Moemeng"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
