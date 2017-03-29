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
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest

import numpy as np

from kamrecsys.recommenders import BaseEventItemFinder
from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

class EventItemFinder(BaseEventItemFinder):

    def __init__(self):
        super(EventItemFinder, self).__init__(random_state=1234)

    def raw_predict(self):
        pass

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseEventItemFinder(unittest.TestCase):

    def setUp(self):
        self.rec = EventItemFinder()
        self.data = load_movielens_mini()

    def test__get_event_array(self):
        data = load_movielens_mini()
        data.filter_event(
            np.logical_and(data.event[:, 0] < 5, data.event[:, 1] < 5))

        event, n_objects = self.rec._get_event_array(data, sparse_type='array')
        assert_array_equal(
            event[:5, :5],
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0]])

        event2, n_objects = self.rec._get_event_array(data, sparse_type='csr')
        assert_array_equal(event, event2.todense())

        event2, n_objects = self.rec._get_event_array(data, sparse_type='csc')
        assert_array_equal(event, event2.todense())

        event2, n_objects = self.rec._get_event_array(data, sparse_type='lil')
        assert_array_equal(event, event2.todense())


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    unittest.main()
