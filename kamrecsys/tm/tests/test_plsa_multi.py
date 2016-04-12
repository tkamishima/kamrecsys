#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

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
from sklearn.utils import check_random_state

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestEventScorePredictor(unittest.TestCase):

    def test__init_params(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.tm.plsa_multi import EventScorePredictor

        data = load_movielens_mini()

        rcmdr = EventScorePredictor(k=7)
        rcmdr._rng = check_random_state(1234)
        rcmdr.n_score_levels_ = data.n_score_levels
        ev, sc, _ = rcmdr._get_event_and_score(data, (0, 1), 0)
        rcmdr.n_events_ = ev.shape[0]
        sc = data.digitize_score(sc)
        rcmdr._init_params(ev, sc)

        assert_allclose(
            rcmdr._q[0, :],
            [2.02037583e-04, 9.24806626e-04, 9.93708976e-01, 3.02458235e-04,
             3.07529874e-04, 1.53842507e-03, 3.01576650e-03],
            rtol=1e-5)

        assert_allclose(
            rcmdr._q[12, :],
            [5.80971583e-05, 1.09346454e-03, 1.43951449e-03, 1.21643700e-03,
             9.96005511e-01, 1.57245892e-04, 2.97297431e-05],
            rtol=1e-5)

    def test_class(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.tm.plsa_multi import EventScorePredictor

        data = load_movielens_mini()

        rcmdr = EventScorePredictor(k=2, random_state=1234)
        self.assertDictEqual(
            vars(rcmdr), {
                'k': 2, 'tol': 1e-05, 'maxiter': 100, 'alpha': 1.0,
                'random_state': 1234, '_rng': None,
                'iid': None, 'eid': None, 'n_objects': None, 'n_otypes': 0,
                'n_score_levels_': 0,
                'i_loss_': np.inf, 'f_loss_': np.inf, 'n_iter_': 0,
                'pz_': None, 'pygz_': None, 'prgz_': None, 'pxgz_': None,
                'n_events_': 0, 'n_users_': 0, 'n_items_': 0, '_q': None,
                'score_levels_': None, 'use_expectation': True})

        # import logging
        # logging.getLogger('kamrecsys').addHandler(logging.StreamHandler())
        rcmdr.fit(data)
        self.assertAlmostEqual(rcmdr.i_loss_,
                               1.97477687885788, delta=1e-5)
        self.assertAlmostEqual(rcmdr.f_loss_,
                               0.532957494417634, delta=1e-5)
        assert_allclose(rcmdr.score_levels_, [1, 2, 3, 4, 5], rtol=1e-5)

        # known user and item
        self.assertAlmostEqual(rcmdr.predict((1, 7)),
                               3.15758362953, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((1, 9)),
                               3.15755141779, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 7)),
                               3.15736862434, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 9)),
                               3.15725064792, delta=1e-5)

        # known user and unknown item
        self.assertAlmostEqual(rcmdr.predict((1, 11)),
                               3.15710024514, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 12)),
                               3.15560057575, delta=1e-5)

        # unknown user and known item
        self.assertAlmostEqual(rcmdr.predict((3, 7)),
                               3.15677852794, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((11, 9)),
                               3.15642545907, delta=1e-5)

        # unknown user and item
        self.assertAlmostEqual(rcmdr.predict((3, 11)),
                               3.15150656272, delta=1e-5)

        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 7], [3, 11],
            [5, 7], [5, 9], [5, 12]])
        assert_allclose(
            rcmdr.predict(x),
            [3.15758362953, 3.15755141779, 3.15710024514,
             3.15677852794, 3.15677852794, 3.15150656272,
             3.15736862434, 3.15725064792, 3.15560057575],
            rtol=1e-5)

        rcmdr.use_expectation = False
        assert_allclose(
            rcmdr.predict(x),
            [4., 4., 4., 4., 4., 4., 4., 4., 4.],
            rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
