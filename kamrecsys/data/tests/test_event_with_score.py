#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)
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

import os
import numpy as np

from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_test_data():
    from kamrecsys.data import EventWithScoreData
    from kamrecsys.datasets import SAMPLE_PATH

    infile = os.path.join(SAMPLE_PATH, 'pci.event')
    dtype = np.dtype([('event', 'U18', 2), ('score', np.float)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, event_otypes=np.array([0, 1]))
    data.set_event(x['event'], x['score'], score_domain=(1.0, 5.0, 0.5))
    return data, x


# =============================================================================
# Test Classes
# =============================================================================


class TestEventWithScoreData(unittest.TestCase):

    def test_set_event(self):
        data, x = load_test_data()

        # test info related to scores
        assert_allclose(data.score[:5], [3., 4., 3.5, 5., 3.])
        assert_allclose(data.score_domain, [1.0, 5.0, 0.5])
        self.assertEqual(data.n_stypes, 1)
        self.assertEqual(data.n_score_levels, 9)

    def test_digitize_score(self):
        data, x = load_test_data()

        digitized_scores = data.digitize_score()
        assert_array_equal(digitized_scores[:5], [4, 6, 5, 8, 4])
        assert_array_equal(digitized_scores[-5:], [4, 3, 4, 5, 6])

        digitized_scores = data.digitize_score(np.linspace(1.0, 5.0, 9))
        assert_array_equal(digitized_scores, np.arange(9))

    def test_filter_event(self):
        data = load_movielens_mini()

        data.filter_event(data.score > 3)
        assert_allclose(
            data.score,
            [4., 4., 4., 5., 4., 5., 5., 5., 4., 5.,
             4., 5., 5., 4., 5., 4., 4., 4., 4., 4., 4.])

        assert_allclose(
            data.to_eid(0, data.event[:, 0]),
            [10, 5, 10, 1, 7, 7, 9, 7, 10, 1,
             2, 1, 7, 6, 7, 6, 1, 9, 1, 6, 10])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
