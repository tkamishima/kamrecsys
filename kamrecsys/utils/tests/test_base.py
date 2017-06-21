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
import numpy as np

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def test_is_binary_score():
    from kamrecsys.utils import is_binary_score

    assert_(is_binary_score([0, 1, 1, 0, 1]))
    assert_(is_binary_score(np.identity(3), allow_uniorm=True))
    assert_(is_binary_score([0, 0, 0]))
    assert_(is_binary_score([1], allow_uniorm=True))

    assert_(is_binary_score([0, 1, 1, 0, 1], allow_uniorm=False))
    assert_(is_binary_score(np.identity(3), allow_uniorm=False))
    assert_(not is_binary_score([0, 0, 0], allow_uniorm=False))
    assert_(not is_binary_score([1], allow_uniorm=False))

# =============================================================================
# Test Classes
# =============================================================================

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
