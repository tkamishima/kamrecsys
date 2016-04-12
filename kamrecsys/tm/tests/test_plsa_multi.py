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

        rcmdr = EventScorePredictor(tol=1e-8, k=2, random_state=1234)
        self.assertDictEqual(
            vars(rcmdr), {
                'k': 2, 'tol': 1e-8, 'maxiter': 100, 'alpha': 1.0,
                'random_state': 1234, '_rng': None,
                'iid': None, 'eid': None, 'n_objects': None, 'n_otypes': 0,
                'n_score_levels_': 0,
                'i_loss_': np.inf, 'f_loss_': np.inf, 'n_iter_': 0,
                'pZ_': None, 'pYgZ_': None, 'pRgZ_': None, 'pXgZ_': None,
                'n_events_': 0, 'n_users_': 0, 'n_items_': 0, '_q': None,
                'score_levels_': None, 'use_expectation': True})

        # import logging
        # logging.getLogger('kamrecsys').addHandler(logging.StreamHandler())
        rcmdr.fit(data)

        self.assertAlmostEqual(rcmdr.i_loss_, 5.41836900049, delta=1e-5)
        self.assertAlmostEqual(rcmdr.f_loss_, 5.17361298499, delta=1e-5)
        assert_allclose(rcmdr.score_levels_, [1, 2, 3, 4, 5], rtol=1e-5)

        # known user and item
        self.assertAlmostEqual(rcmdr.predict((1, 7)),
                               3.64580117249, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((1, 9)),
                               3.6587422493, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 7)),
                               3.60707987724, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 9)),
                               3.62184516985, delta=1e-5)

        # known user and unknown item
        self.assertAlmostEqual(rcmdr.predict((1, 11)),
                               3.66032199689, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((5, 12)),
                               3.62387542269, delta=1e-5)

        # unknown user and known item
        self.assertAlmostEqual(rcmdr.predict((3, 7)),
                               3.60821491793, delta=1e-5)
        self.assertAlmostEqual(rcmdr.predict((11, 9)),
                               3.62304301551, delta=1e-5)

        # unknown user and item
        self.assertAlmostEqual(rcmdr.predict((3, 11)),
                               3.62507437787, delta=1e-5)

        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rcmdr.predict(x),
            [3.64580117249, 3.6587422493, 3.66032199689,
             3.60821491793, 3.62304301551, 3.62507437787,
             3.60707987724, 3.62184516985, 3.62387542269],
            rtol=1e-5)

        rcmdr.use_expectation = False
        assert_allclose(
            rcmdr.predict(x),
            [4., 5., 5., 4., 4., 4., 4., 4., 4.],
            rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
