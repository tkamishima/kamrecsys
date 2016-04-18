#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest

import numpy as np

# =============================================================================
# Module variables
# =============================================================================

test_data = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
             4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
             4.01993828853, 4.56297459028]

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseMetrics(unittest.TestCase):

    def test_class(self):
        from kamrecsys.metrics import BaseMetrics

        stats = BaseMetrics()
        self.assertDictEqual(
            vars(stats),
            {'metrics': {}, 'params': {}, 'name': 'metrics'})

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
        from kamrecsys.metrics import DescriptiveStatistics

        metrics = DescriptiveStatistics(test_data, name="dummy")
        self.assertEqual(metrics.name, "dummy")

        metrics = DescriptiveStatistics(test_data)
        self.assertEqual(metrics.name, "descriptive_statistics")
        self.assertDictEqual(
            metrics.metrics,
            {
                'nos_samples': 10,
                'mean': np.mean(test_data),
                'stdev': np.std(test_data)})
        with self.assertRaises(ValueError):
            DescriptiveStatistics([np.nan])


class TestHistogram(unittest.TestCase):

    def test_class(self):
        from kamrecsys.metrics import Histogram

        m = Histogram(test_data, name="dummy")
        self.assertEqual(m.name, "dummy")
        assert_allclose(m.metrics['count'], [0, 0, 2, 7, 1])
        assert_allclose(m.metrics['density'], [0.0, 0.0, 0.2, 0.7, 0.1])

        m = Histogram(test_data, bins=[-np.inf, 4.0, np.inf])
        self.assertEqual(m.name, "histogram")
        assert_allclose(m.metrics['count'], [3, 7])
        assert_allclose(m.metrics['density'], [0.3, 0.7])

        x = np.linspace(0.0, 1.0, 21)
        m = Histogram(x, bins=[-np.inf, 0.3, np.inf])
        assert_allclose(m.metrics['count'], [6, 15])
        assert_allclose(m.metrics['density'], [0.28571429, 0.71428571])

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
