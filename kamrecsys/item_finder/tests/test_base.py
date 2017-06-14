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

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.item_finder import BaseImplicitItemFinder

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class ImplicitItemFinder(BaseImplicitItemFinder):

    def __init__(self):
        super(ImplicitItemFinder, self).__init__(random_state=1234)

    def raw_predict(self):
        pass


class TestBaseImplicitItemFinder(TestCase):

    def test__get_event_array(self):
        rec = ImplicitItemFinder()
        data = load_movielens_mini()
        data.filter_event(
            np.logical_and(data.event[:, 0] < 5, data.event[:, 1] < 5))
        
        rec.fit(data)
        ev, n_objects = rec.get_event_array(sparse_type='array')
        assert_array_equal(
            ev[:5, :5],
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0]])

        ev2, n_objects = rec.get_event_array('csr')
        assert_array_equal(ev, ev2.todense())

        ev2, n_objects = rec.get_event_array('csc')
        assert_array_equal(ev, ev2.todense())

        ev2, n_objects = rec.get_event_array('lil')
        assert_array_equal(ev, ev2.todense())

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
