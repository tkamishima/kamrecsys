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

    def test_class(self):
        import numpy as np
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
                'score_levels_': None, 'score_dist_': None})

        # import logging
        # logging.getLogger('kamrecsys').addHandler(logging.StreamHandler())
        rcmdr.fit(data)
        self.assertAlmostEqual(rcmdr.i_loss_,
                               1.97477687885788, delta=1e-5)
        self.assertAlmostEqual(rcmdr.f_loss_,
                               0.532957494417634, delta=1e-5)
        assert_allclose(rcmdr.score_dist_,
                        [0.03333333, 0.06666667, 0.2, 0.43333333, 0.26666667],
                        rtol=1e-5)
        assert_allclose(rcmdr.score_levels_, [1, 2, 3, 4, 5], rtol=1e-5)

        # known user and item
        self.assertAlmostEqual(rcmdr.predict((1, 7)),
                               3.15758362953, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((1, 9)),
                               3.15755141779, delta=1e-5)
        assert_allclose(
            rcmdr.predict([[5, 7], [5, 9]]),
            [3.15736862434, 3.15725064792],
            rtol=1e-5)

        # known user and unknown item
        # self.assertAlmostEqual(rcmdr.predict((1, 11)),
        #                       3.6480799850368273, delta=1e-5)
        # self.assertAlmostEqual(rcmdr.predict((5, 12)),
        #                        3.4710520150321895, delta=1e-5)

        # unknown user and known item
        # self.assertAlmostEqual(rcmdr.predict((3, 7)),
        #                        3.6336318795279228, delta=1e-5)
        # self.assertAlmostEqual(rcmdr.predict((11, 9)),
        #                        4.2482001235634943, delta=1e-5)

        # unknown user and item
        self.assertAlmostEqual(rcmdr.predict((3, 11)),
                               3.83333333333, delta=1e-5)
        # x = np.array([
        #     [1, 7], [1, 9], [1, 11],
        #     [3, 7], [3, 9], [3, 11],
        #     [5, 7], [5, 9], [5, 11]])
        # assert_allclose(
        #     rcmdr.predict(x),
        #     [3.98736414, 4.98921188, 3.64807999, 3.63363188, 4.24820012,
        #      3.72369841, 3.41419681, 3.9818882, 3.47105202],
        #     rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
