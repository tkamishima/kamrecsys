#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)

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
    dtype = np.dtype([('event', 'S18', 2), ('score', np.float)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, n_stypes=1,
                              event_otypes=np.array([0, 1]))
    data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0, 0.5))
    return data, x


# =============================================================================
# Test Classes
# =============================================================================


class TestEventUtilMixin(unittest.TestCase):

    def test_to_eid_event(self):
        data, x = load_test_data()

        # test to_eid_event
        check = data.to_eid_event(data.event)
        assert_array_equal(x['event'], check)

        # test to_eid_event / per line conversion
        check = np.empty_like(data.event, dtype=x['event'].dtype)
        for i, j in enumerate(data.event):
            check[i, :] = data.to_eid_event(j)
        assert_array_equal(x['event'], check)

    def test_to_iid_event(self):
        from kamrecsys.data import EventWithScoreData
        data, x = load_test_data()

        # test EventData.to_iid_event
        assert_array_equal(data.event, data.to_iid_event(x['event']))

        # test EventData.to_iid_event / per line conversion
        check = np.empty_like(x['event'], dtype=np.int)
        for i, j in enumerate(x['event']):
            check[i, :] = data.to_iid_event(j)
        assert_array_equal(data.event, check)


class TestEventWithScoreData(unittest.TestCase):

    def test_set_events(self):
        data, x = load_test_data()

        # test info related to scores
        assert_allclose(data.score[:5], [3., 4., 3.5, 5., 3.])
        assert_allclose(data.score_domain, [1.0, 5.0, 0.5])
        self.assertEqual(data.n_scores, 1)
        self.assertEqual(data.n_score_levels, 9)

    def test_digitize_score(self):
        data, x = load_test_data()

        digitized_scores = data.digitize_score()
        assert_array_equal(digitized_scores[:5], [4, 6, 5, 8, 4])
        assert_array_equal(digitized_scores[-5:], [4, 3, 4, 5, 6])

        digitized_scores = data.digitize_score(np.linspace(1.0, 5.0, 9))
        assert_array_equal(digitized_scores, np.arange(9))

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
