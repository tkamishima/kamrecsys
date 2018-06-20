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


def test_pergroup_kfold():

    from kamrecsys.model_selection import generate_pergroup_kfold

    group = np.array([1, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 1, 3])

    # error handling
    with assert_raises(ValueError):
        generate_pergroup_kfold(10, None, 1)

    with assert_raises(ValueError):
        generate_pergroup_kfold(10, np.zeros(9), 3)

    with assert_raises(ValueError):
        generate_pergroup_kfold(13, group, 4)

    # function
    # test_fold = generate_pergroup_kfold(13, group, 3)
    # assert_array_equal(
    #     test_fold, [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 2, 2, 2])
    #
    # test_fold = generate_pergroup_kfold(13, None, 5)
    # assert_array_equal(
    #     test_fold, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])

    test_fold = generate_pergroup_kfold(
        13, group, 3, shuffle=True, random_state=1234)
    assert_array_equal(
        test_fold, [0, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 0])

    test_fold = generate_pergroup_kfold(
        13, None, 5, shuffle=True, random_state=1234)
    assert_array_equal(
        test_fold, [0, 2, 1, 4, 3, 3, 4, 2, 2, 1, 0, 1, 0])


def test_generate_interlace_kfold():

    from kamrecsys.model_selection import generate_interlace_kfold

    # error handling
    with assert_raises(ValueError):
        generate_interlace_kfold(1, 1)

    with assert_raises(ValueError):
        generate_interlace_kfold(1, 2)

    # check function
    group = generate_interlace_kfold(7, 3)
    assert_array_equal(group, [0, 1, 2, 0, 1, 2, 0])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
