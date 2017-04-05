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
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)

import numpy as np
import sys

from kamrecsys.score_predictor import BaseEventScorePredictor
from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


class EventScorePredictor(BaseEventScorePredictor):

    def __init__(self):
        super(EventScorePredictor, self).__init__(random_state=1234)

    def raw_predict(self):
        pass

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseScorePredictor(TestCase):

    def setUp(self):
        self.rec = EventScorePredictor()
        self.data = load_movielens_mini()

    def test__event_and_score(self):
        data = load_movielens_mini()

        event, sc, n_objects = self.rec._get_event_and_score(data, (0, 1), 0)

        assert_array_equal(
            event,
            [[2, 1], [7, 6], [2, 0], [7, 3], [0, 5],
             [4, 9], [4, 7], [6, 5], [4, 6], [5, 6],
             [0, 9], [7, 0], [0, 8], [0, 1], [1, 0],
             [0, 7], [0, 0], [1, 9], [0, 4], [3, 6],
             [4, 3], [3, 0], [4, 8], [0, 3], [3, 8],
             [0, 6], [6, 6], [0, 2], [3, 7], [7, 8]])
        assert_allclose(
            sc,
            [3., 4., 4., 4., 5., 4., 5., 5., 5., 3.,
             3., 4., 5., 3., 4., 1., 5., 2., 3., 2.,
             5., 4., 5., 3., 4., 4., 4., 4., 4., 4.])
        assert_array_equal(n_objects, [8, 10])

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
