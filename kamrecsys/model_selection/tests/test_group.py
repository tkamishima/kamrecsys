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

from sklearn.model_selection import LeaveOneGroupOut
from kamrecsys.model_selection import interlace_group

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestInterlaceGroup(TestCase):

    def test_function(self):

        group = interlace_group(7, 3)
        assert_array_equal(group, [0, 1, 2, 0, 1, 2, 0])

        X = np.arange(7, dtype=int).reshape(-1, 1)

        cv = LeaveOneGroupOut()
        cv_iter = cv.split(X, groups=group)

        train_X, test_X = next(cv_iter)
        assert_array_equal(train_X, [1, 2, 4, 5])
        assert_array_equal(test_X, [0, 3, 6])

        train_X, test_X = next(cv_iter)
        assert_array_equal(train_X, [0, 2, 3, 5, 6])
        assert_array_equal(test_X, [1, 4])

        train_X, test_X = next(cv_iter)
        assert_array_equal(train_X, [0, 1, 3, 4, 6])
        assert_array_equal(test_X, [2, 5])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
