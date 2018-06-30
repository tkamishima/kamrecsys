#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_string_equal)
import numpy as np

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


def test_generate_score_bins():
    from kamrecsys.metrics import generate_score_bins

    assert_allclose(
        generate_score_bins([1., 5., 0.5]),
        [-np.inf, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, np.inf])
    assert_allclose(
        generate_score_bins(np.array([0, 4, 1])),
        [-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])


def test_statistics_mean():
    from kamrecsys.metrics import statistics_mean

    def fun1(x):
        x_sum = np.sum(np.asarray(x))
        return x_sum, x_sum**2

    def fun2(x, y):
        x_sum = np.sum(np.asarray(x))
        y_sum = np.sum(np.asarray(y))
        return x_sum + y_sum, x_sum - y_sum

    def fun3(x, y):
        return np.mean(x) + np.mean(y)

    # list of dicts
    dict1 = {'0': [1, 2], '1': [10, 20, 30]}
    dict2 = {'1': [100, 200], '0': [1, 2, 3, 4]}

    assert_allclose(statistics_mean(fun1, dict1), [31.5, 1804.5])
    assert_allclose(statistics_mean(fun2, dict1, dict2), [186.5, -123.5])
    assert_allclose(statistics_mean(fun3, dict1, dict2), 87.0)
    assert_allclose(statistics_mean(np.mean, dict1), 10.75)

    # list of lists
    list1 = [[1, 2], [10, 20, 30]]
    list2 = [[1, 2, 3, 4], [100, 200]]

    assert_allclose(statistics_mean(fun1, dict1), [31.5, 1804.5])
    assert_allclose(statistics_mean(fun2, dict1, dict2), [186.5, -123.5])
    assert_allclose(statistics_mean(fun3, dict1, dict2), 87.0)
    assert_allclose(statistics_mean(np.mean, dict1), 10.75)


# =============================================================================
# Main Routine
# =============================================================================


if __name__ == '__main__':
    run_module_suite()
