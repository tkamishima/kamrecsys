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


class TestBaseMetrics(unittest.TestCase):

    def test_class(self):
        from .._base import BaseMetrics

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


if __name__ == '__main__':
    unittest.main()
