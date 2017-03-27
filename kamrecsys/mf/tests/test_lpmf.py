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

import os
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


class TestEventItemFinder(unittest.TestCase):

    def test_loss(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.lpmf import EventItemFinder

        # setup
        data = load_movielens_mini()
        rec = EventItemFinder(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        ev, n_objects = rec._get_event_array(data, sparse_type='csr')
        rec._init_coef(ev, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        # self.assertAlmostEqual(rec.loss(rec._coef, tev, tsc, n_objects),
        #                        15.811193562306155)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        # self.assertAlmostEqual(rec.loss(rec._coef, tev, tsc, n_objects),
        #                        2.4876784107910042)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        # self.assertAlmostEqual(rec.loss(rec._coef, tev, tsc, n_objects),
        #                        186.54355756518166)

    def test_grad_loss(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.lpmf import EventItemFinder

        # setup
        data = load_movielens_mini()
        rec = EventItemFinder(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        ev, n_objects = rec._get_event_array(data, sparse_type='csr')
        rec._init_coef(ev, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        # grad = rec.grad_loss(rec._coef, tev, tsc, n_objects)
        # assert_allclose(grad[:5],
        #                 [-1.86666667, -0.76666667, -0.13333333,
        #                  -0.23333333, -0.13333333])
        # assert_allclose(grad[-5:],
        #                 [0., 0., 0., 0., 0.])

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        # grad = rec.grad_loss(rec._coef, tev, tsc, n_objects)
        # assert_allclose(grad[:8],
        #                 [4.66666667e-01, 2.33424242e-01, 3.34242424e-02,
        #                  1.00090909e-01, 3.34242424e-02, 9.09090909e-05,
        #                  9.09090909e-05, 9.09090909e-05])
        # assert_allclose(grad[-8:],
        #                 [0.26675758, 0.26675758, 0.16675758, 0.16675758,
        #                  0.06675758, 0.06675758, 0.20009091, 0.20009091])

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        # grad = rec.grad_loss(rec._coef, tev, tsc, n_objects)
        # assert_allclose(grad[:8],
        #                 [-3.036162e+01, 1.805286e+01, 7.528713e+00,
        #                  6.039861e+00, 1.494799e+00, -1.060639e+01,
        #                  4.545455e-05, -2.267424e+01],
        #                 rtol=1e-5)
        # assert_allclose(grad[-8:],
        #                 [22.285565, 22.285565, 27.127236, 27.127236,
        #                  -19.282996, -19.282996, 1.63885, 1.63885],
        #                 rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        # grad = rec.grad_loss(rec._coef, tev, tsc, n_objects)
        # assert_allclose(grad[:8],
        #                 [2.294263e+01, 2.365823e+01, 1.206014e+01,
        #                  1.584586e+01, 7.741774e+00, -6.048586e+00,
        #                  2.727273e-05, -1.748194e+01],
        #                 rtol=1e-5)
        # assert_allclose(grad[-8:],
        #                 [-17.932169, -17.355552, 3.694892, 5.874205,
        #                  -9.412818, -8.337879, 5.433757, 8.25851],
        #                 rtol=1e-5)

    def test_class(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.lpmf import EventItemFinder

        # setup
        data = load_movielens_mini()
        rec = EventItemFinder(C=0.1, k=2, tol=1e-03, random_state=1234)

        self.assertDictEqual(
            vars(rec),
            {'C': 0.1, 'n_otypes': 0, 'bu_': None, 'bi_': None, 'k': 2,
             'p_': None, 'q_': None, '_coef': None, 'f_loss_': np.inf,
             'iid': None, 'i_loss_': np.inf, 'eid': None, 'tol': 1e-03,
             'n_objects': None, '_dt': None, 'mu_': None, 'opt_outputs_': None,
             'maxiter': 200, 'random_state': 1234, '_rng': None})

        # recommender.fit(data, disp=False)
        # self.assertAlmostEqual(recommender.i_loss_,
        #                        0.74652578358324106, delta=1e-5)
        # self.assertAlmostEqual(recommender.f_loss_,
        #                        0.025638738121075231, delta=1e-5)
        #
        # self.assertAlmostEqual(recommender.predict((1, 7)),
        #                        3.9873641434545979, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((1, 9)),
        #                        4.9892118821609106, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((1, 11)),
        #                        3.6480799850368273, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((3, 7)),
        #                        3.6336318795279228, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((3, 9)),
        #                        4.2482001235634943, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((3, 11)),
        #                        3.7236984083417841, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((5, 7)),
        #                        3.4141968145802597, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((5, 9)),
        #                        3.9818882049478654, delta=1e-5)
        # self.assertAlmostEqual(recommender.predict((5, 11)),
        #                        3.4710520150321895, delta=1e-5)
        # x = np.array([
        #     [1, 7], [1, 9], [1, 11],
        #     [3, 7], [3, 9], [3, 11],
        #     [5, 7], [5, 9], [5, 11]])
        # assert_allclose(
        #     recommender.predict(x),
        #     [3.98736414, 4.98921188, 3.64807999, 3.63363188, 4.24820012,
        #      3.72369841, 3.41419681, 3.9818882, 3.47105202],
        #     rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
