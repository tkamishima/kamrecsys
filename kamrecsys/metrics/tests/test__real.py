#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest
import numpy as np
from sklearn.utils import (
    assert_all_finite,
    safe_asarray)

y_true = [5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
y_pred = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
          4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
          4.01993828853, 4.56297459028]

class TestBaseMetrics(unittest.TestCase):

    def test_class(self):
        from .. import BaseRealMetrics

        stats = BaseRealMetrics([], [])
        self.assertDictEqual(stats.metrics, {})
        self.assertEqual(stats.name, 'real_metrics')
        with self.assertRaises(ValueError):
            BaseRealMetrics([np.nan], [1.0])
        with self.assertRaises(ValueError):
            BaseRealMetrics([100], [np.inf])
        with self.assertRaises(ValueError):
            BaseRealMetrics([[100]], [1.0])

class TestMeanAbsoluteError(unittest.TestCase):

    def test_class(self):
        from .. import MeanAbsoluteError

        metrics = MeanAbsoluteError(y_true, y_pred)
        self.assertEqual(metrics.name, 'mean_absolute_error')
        self.assertAlmostEqual(metrics.metrics['mean'],
                               0.9534215971390001,
                               delta=1e-5)
        self.assertAlmostEqual(metrics.metrics['stdev'],
                               0.6602899115612394,
                               delta=1e-5)

if __name__ == '__main__':
    unittest.main()
