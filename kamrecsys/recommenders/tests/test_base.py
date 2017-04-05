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

from kamrecsys.recommenders import BaseEventRecommender

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class EventRecommender(BaseEventRecommender):

    def __init__(self):
        super(EventRecommender, self).__init__(random_state=1234)

    def raw_predict(self):
        pass

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseEventRecommender(TestCase):

    def test_class(self):
        pass


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
