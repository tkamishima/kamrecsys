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


def test_GroupWiseKfold():

    from kamrecsys.model_selection import KFoldWithinGroups

    groups = np.array([1, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 1, 3])

    # error handling
    with assert_raises(ValueError):
        KFoldWithinGroups(n_splits=1)

    with assert_raises(ValueError):
        cv = KFoldWithinGroups(n_splits=3)
        cv.split(np.arange(10), groups=np.zeros(9)).next()

    with assert_raises(ValueError):
        cv = KFoldWithinGroups(n_splits=4)
        cv.split(np.arange(13), groups=groups).next()

    # function
    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(3)
    for i, g in enumerate(cv.split(np.arange(13), groups=groups)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 2, 2, 2])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(5)
    for i, g in enumerate(cv.split(np.arange(13), groups=None)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(3, shuffle=True, random_state=1234)
    for i, g in enumerate(cv.split(np.arange(13), groups=groups)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [1, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 2, 1])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(5, shuffle=True, random_state=1234)
    for i, g in enumerate(cv.split(np.arange(13), groups=None)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 2, 1, 4, 3, 3, 4, 2, 2, 1, 0, 1, 0])


def test_InterlacedKFold():

    from kamrecsys.model_selection import InterlacedKFold

    with assert_raises(ValueError):
        InterlacedKFold(n_splits=1)

    with assert_raises(ValueError):
        cv = InterlacedKFold(n_splits=2)
        cv.split(np.zeros(1))

    with assert_raises(ValueError):
        cv = InterlacedKFold(n_splits=2)
        cv.split(np.zeros(3), np.zeros(2))

    test_fold = np.empty(7, dtype=int)
    cv = InterlacedKFold(n_splits=3)
    iter = cv.split(np.zeros(7))
    train_i, test_i = iter.next()
    assert_array_equal(train_i, [1, 2, 4, 5])
    assert_array_equal(test_i, [0, 3, 6])
    test_fold[test_i] = 0
    train_i, test_i = iter.next()
    test_fold[test_i] = 1
    train_i, test_i = iter.next()
    test_fold[test_i] = 2
    assert_array_equal(test_fold, [0, 1, 2, 0, 1, 2, 0])
    with assert_raises(StopIteration):
        iter.next()

    cv = InterlacedKFold(n_splits=3)
    assert_equal(cv.get_n_splits(), 3)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
