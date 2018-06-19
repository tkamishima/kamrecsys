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
