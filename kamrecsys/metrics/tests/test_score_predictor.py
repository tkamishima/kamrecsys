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


class TestScorePredictorReport(TestCase):

    def test_class(self):
        from kamrecsys.metrics import score_predictor_report

        stats = score_predictor_report(y_true, y_pred, disp=False)
        self.assertEqual(stats['n_samples'], 10)
        self.assertAlmostEqual(
            stats['mean absolute error'], 0.9534215971390001, delta=1e-5)
        self.assertAlmostEqual(
            stats['root mean squared error'], 1.1597394143516166, delta=1e-5)
        self.assertAlmostEqual(
            stats['true']['mean'], 4.1, delta=1e-5)
        self.assertAlmostEqual(
            stats['true']['stdev'], 1.04403065089, delta=1e-5)
        self.assertAlmostEqual(
            stats['predicted']['mean'], 3.99361964567, delta=1e-5)
        self.assertAlmostEqual(
            stats['predicted']['stdev'], 0.383771468193, delta=1e-5)


class TestScorePredictorStatistics(TestCase):

    def test_class(self):
        from kamrecsys.metrics import score_predictor_statistics

        stats = score_predictor_statistics(
            y_true, y_pred, score_domain=(1, 5, 1))

        self.assertEqual(stats['n_samples'], 10)
        assert_allclose(stats['score levels'], (1, 2, 3, 4, 5))

        sub_stats = stats['mean absolute error']
        self.assertAlmostEqual(
            sub_stats['mean'], 0.9534215971390001, delta=1e-5)
        self.assertAlmostEqual(
            sub_stats['stdev'], 0.6602899115612394, delta=1e-5)

        sub_stats = stats['mean squared error']
        self.assertAlmostEqual(
            sub_stats['rmse'], 1.1597394143516166, delta=1e-5)
        self.assertAlmostEqual(
            sub_stats['mean'], 1.3449955092006309, delta=1e-5)
        self.assertAlmostEqual(
            sub_stats['stdev'], 1.4418716080648177, delta=1e-5)

        sub_stats = stats['true']
        self.assertAlmostEqual(
            sub_stats['mean'], 4.1, delta=1e-5)
        self.assertAlmostEqual(
            sub_stats['stdev'], 1.04403065089, delta=1e-5)
        assert_array_equal(sub_stats['histogram'], (0, 1, 2, 2, 5))
        assert_allclose(
            sub_stats['histogram density'],
            [0.0, 0.1, 0.2, 0.2, 0.5],
            rtol=1e-5)

        sub_stats = stats['predicted']
        self.assertAlmostEqual(
            sub_stats['mean'], 3.99361964567, delta=1e-5)
        self.assertAlmostEqual(
            sub_stats['stdev'], 0.383771468193, delta=1e-5)
        assert_array_equal(sub_stats['histogram'], (0, 0, 2, 7, 1))
        assert_allclose(
            sub_stats['histogram density'],
            [0.0, 0.0, 0.2, 0.7, 0.1],
            rtol=1e-5)

        # check predicted scores
        stats = score_predictor_statistics(y_true, y_pred)
        assert_allclose(stats['score levels'], [1, 2, 3, 4, 5], rtol=1e-5)
        assert_array_equal(stats['true']['histogram'], [0, 1, 2, 2, 5])
        assert_array_equal(stats['predicted']['histogram'], [0, 0, 2, 7, 1])

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
