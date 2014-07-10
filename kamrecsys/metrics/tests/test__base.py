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

test_data = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
             4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
             4.01993828853, 4.56297459028]

class TestBaseMetrics(unittest.TestCase):

    def test_class(self):
        from .. import BaseMetrics

        stats = BaseMetrics()
        self.assertDictEqual(vars(stats),
            {'metrics': {}, 'name': 'metrics'})

        stats.metrics['a'] = 1.0
        stats.metrics['b'] = 2.0
        self.assertEqual(stats.name, "metrics")
        self.assertEqual(stats.metrics, {'a': 1.0, 'b': 2.0})
        stats.metrics['B'] = 3.0
        stats.metrics['A'] = 2.0
        self.assertListEqual(stats.subnames(),
                             ['A', 'B', 'a', 'b'])
        self.assertListEqual(stats.fullnames(),
                             ['metrics_A', 'metrics_B',
                              'metrics_a', 'metrics_b'])
        self.assertListEqual(stats.values(),
                             [2.0, 3.0, 1.0, 2.0])


class TestDescriptiveStatistics(unittest.TestCase):

    def test_class(self):
        from .. import DescriptiveStatistics

        metrics = DescriptiveStatistics(test_data, name="dummry")
        self.assertEqual(metrics.name, "dummry")

        metrics = DescriptiveStatistics(test_data)
        self.assertEqual(metrics.name, "descriptive_statistics")
        self.assertDictEqual(metrics.metrics,
            {'nos_samples': 10,
             'mean': np.mean(test_data),
             'stdev': np.std(test_data)})
        with self.assertRaises(ValueError):
            metrics = DescriptiveStatistics([np.nan])

if __name__ == '__main__':
    unittest.main()
