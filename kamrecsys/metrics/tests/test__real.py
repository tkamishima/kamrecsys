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
        self.assertDictEqual(vars(stats),
            {'metrics': {}, 'name': 'real_metrics'})
        with self.assertRaises(ValueError):
            BaseRealMetrics([np.nan], [1.0])
        with self.assertRaises(ValueError):
            BaseRealMetrics([100], [np.inf])
        with self.assertRaises(ValueError):
            BaseRealMetrics([[100]], [1.0])

"""
class TestDescriptiveStatistics(unittest.TestCase):

    def test_class(self):
        from .._base import DescriptiveStatistics

        metrics = DescriptiveStatistics(test_data, name="dummry")
        self.assertEqual(metrics.name, "dummry")

        metrics = DescriptiveStatistics(test_data)
        self.assertEqual(metrics.name, "descriptive_statistics")
        self.assertDictEqual(metrics.metrics,
            {'nos_samples': 10,
             'mean': np.mean(test_data),
             'stdev': np.std(test_data)})
"""

if __name__ == '__main__':
    unittest.main()
