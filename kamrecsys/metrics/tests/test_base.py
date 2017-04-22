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


class TestGenerateScoreBins(TestCase):

    def test_generate_score_bins(self):
        from kamrecsys.metrics import generate_score_bins

        assert_allclose(
            generate_score_bins([1., 5., 0.5]),
            [-np.inf, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, np.inf])
        assert_allclose(
            generate_score_bins(np.array([0, 4, 1])),
            [-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
