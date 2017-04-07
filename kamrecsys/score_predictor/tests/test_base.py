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

from kamrecsys.score_predictor import BaseScorePredictor
from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Variables
# =============================================================================

true_sc = [
    3., 4., 4., 4., 5., 4., 5., 5., 5., 3.,
    3., 4., 5., 3., 4., 1., 5., 2., 3., 2.,
    5., 4., 5., 3., 4., 4., 4., 4., 4., 4.]

# =============================================================================
# Functions
# =============================================================================


class ScorePredictor(BaseScorePredictor):

    def __init__(self):
        super(ScorePredictor, self).__init__(random_state=1234)

    def raw_predict(self):
        pass

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseScorePredictor(TestCase):

    def test_class(self):
        data = load_movielens_mini()
        rec = ScorePredictor()

        # fit()
        rec.fit(data, event_index=(0, 1), score_index=0)

        self.assertEqual(rec.n_stypes, 1)
        assert_allclose(rec.score_domain, [1., 5., 1.])
        assert_allclose(rec.score, true_sc)
        self.assertEqual(rec.n_score_levels, 5)

        # get_score()
        assert_allclose(rec.get_score(), true_sc)

        # remove_data
        rec.remove_data()
        self.assertEqual(rec.n_stypes, 0)
        self.assertIsNone(rec.score_domain)
        self.assertIsNone(rec.score)
        self.assertIsNone(rec.n_score_levels)

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
