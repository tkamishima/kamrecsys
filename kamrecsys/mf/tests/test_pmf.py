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
        from kamrecsys.mf.pmf import EventScorePredictor

        data = load_movielens_mini()

        recommender = EventScorePredictor(C=0.1, k=2, tol=1e-03,
                                          random_state=1234)
        self.assertDictEqual(
            vars(recommender),
            {'C': 0.1, 'n_otypes': 0, 'bu_': None, 'bi_': None, 'k': 2,
             'p_': None, 'q_': None, '_coef': None, 'f_loss_': np.inf,
             'iid': None, 'i_loss_': np.inf, 'eid': None, 'tol': 1e-03,
             'n_objects': None, '_dt': None, 'mu_': None, 'opt_outputs_': None,
             'maxiter': 200, 'random_state': 1234, '_rng': None})

        recommender.fit(data, disp=False)
        self.assertAlmostEqual(recommender.i_loss_,
                               0.74652578358324106, delta=1e-5)
        self.assertAlmostEqual(recommender.f_loss_,
                               0.025638738121075231, delta=1e-5)

        self.assertAlmostEqual(recommender.predict((1, 7)),
                               3.9873641434545979, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((1, 9)),
                               4.9892118821609106, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((1, 11)),
                               3.6480799850368273, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((3, 7)),
                               3.6336318795279228, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((3, 9)),
                               4.2482001235634943, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((3, 11)),
                               3.7236984083417841, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((5, 7)),
                               3.4141968145802597, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((5, 9)),
                               3.9818882049478654, delta=1e-5)
        self.assertAlmostEqual(recommender.predict((5, 11)),
                               3.4710520150321895, delta=1e-5)
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            recommender.predict(x),
            [3.98736414, 4.98921188, 3.64807999, 3.63363188, 4.24820012,
             3.72369841, 3.41419681, 3.9818882, 3.47105202],
            rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
