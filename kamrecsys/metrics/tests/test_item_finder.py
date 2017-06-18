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
    assert_almost_equal,
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

# =============================================================================
# Variables
# =============================================================================

y_true = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0]
y_pred = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
          4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
          4.01993828853, 4.56297459028]

# =============================================================================
# Functions
# =============================================================================


def test_item_finder_report():
    from kamrecsys.metrics import item_finder_report

    with assert_raises(ValueError):
        item_finder_report([0], [1])

    stats = item_finder_report(y_true, y_pred, disp=False)
    assert_allclose(stats['area_under_the_curve'], 0.4285714285714286,
                    rtol=1e-5)

    assert_equal(stats['n_samples'], 10)
    assert_allclose(stats['true']['mean'], 0.7, rtol=1e-5)
    assert_allclose(stats['true']['stdev'], 0.45825756949558405, rtol=1e-5)
    assert_allclose(stats['predicted']['mean'], 3.99361964567, rtol=1e-5)
    assert_allclose(stats['predicted']['stdev'], 0.383771468193, rtol=1e-5)

# =============================================================================
# Test Classes
# =============================================================================

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
