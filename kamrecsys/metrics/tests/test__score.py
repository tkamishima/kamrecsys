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
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest

import numpy as np

# =============================================================================
# Variables
# =============================================================================

y_true = [5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
y_pred = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
          4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
          4.01993828853, 4.56297459028]

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestMeanAbsoluteError(TestCase):

    def test_class(self):
        from kamrecsys.metrics import mean_absolute_error

        mean, stdev = mean_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(mean, 0.9534215971390001, delta=1e-5)
        self.assertAlmostEqual(stdev, 0.6602899115612394, delta=1e-5)


class TestMeanSquaredError(TestCase):

    def test_class(self):
        from kamrecsys.metrics import mean_squared_error

        rmse, mean, stdev = mean_squared_error(y_true, y_pred)

        self.assertAlmostEqual(rmse, 1.1597394143516166, delta=1e-5)
        self.assertAlmostEqual(mean, 1.3449955092006309, delta=1e-5)
        self.assertAlmostEqual(stdev, 1.4418716080648177, delta=1e-5)


class TestScoreHistogram(unittest.TestCase):

    def test_class(self):
        from kamrecsys.metrics import score_histogram

        hist, scores = score_histogram(y_pred)
        assert_array_equal(hist, [0, 0, 2, 7, 1])
        assert_array_equal(scores, [1, 2, 3, 4, 5])

        hist, scores = score_histogram(y_pred, scores=[3, 5])
        assert_array_equal(hist, [3, 7])
        assert_array_equal(scores, [3, 5])

        hist, scores = score_histogram(
            np.linspace(0.0, 1.0, 21), scores=[0.2, 0.4])
        assert_array_equal(hist, [6, 15])
        assert_array_equal(scores, [0.2, 0.4])

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
