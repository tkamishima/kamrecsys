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

    def test_logistic(self):
        from kamrecsys.mf.lpmf import EventItemFinder

        rec = EventItemFinder()
        self.assertAlmostEqual(rec.sigmoid(0.), 0.5)
        self.assertAlmostEqual(rec.sigmoid(1.), 1 / (1 + 1 / np.e))
        self.assertAlmostEqual(rec.sigmoid(-1.), 1 / (1 + np.e))
        self.assertAlmostEqual(rec.sigmoid(1000.), 1. - 1e-15)
        self.assertAlmostEqual(rec.sigmoid(-1000.), 1e-15)

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

        # initial parameters
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.48692872837196544)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.25)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.71486054877995686)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.67952160620227464)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.94514307182100432)

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

        # initial parameters
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.04537848)
        assert_allclose(
            grad[1:5],
            [-0.00113131, 0.01021755, 0.0088779,  0.00491166],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.00177269, 0.00622547, 0.00210903, 0.00738912],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [3.21924356e-04, -4.07076819e-03, 4.45760105e-03, 6.24894939e-05],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.00371728, 0.0053892, 0.0031896, 0.00046543],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.03125)
        assert_allclose(
            grad[1:5],
            [-0.015625, 0.009375, 0.009375, 0.003125],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.00625, 0.003125, 0., 0.003125],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0., 0., 0., 0.],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0., 0., 0., 0.],
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.00411054)
        assert_allclose(
            grad[1:5],
            [0.00181262, 0.00247743, 0.00247743, 0.00231122],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.00197993, 0.00222924, 0.00214614, 0.00222924],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.00181262, 0.00181262, 0.00247743, 0.00247743],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.00214614, 0.00214614, 0.00222924, 0.00222924],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.01190121)
        assert_allclose(
            grad[1:5],
            [-0.0001093, 0.00261595, 0.00237643, 0.0021332],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.00338797, 0.00397934, 0.00395807, 0.00415455],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.00088723, 0.00170888, 0.00139592, 0.00425231],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.00070631, 0.00250352, 0.00071364, 0.00251819],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 5.73670102e-06)
        assert_allclose(
            grad[1:5],
            [0.00145455, 0.00127381, 0.00109177, 0.00090986],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.00345474, 0.00372762, 0.00400025, 0.00427298],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.00545455, 0.00181818, 0.00560238, 0.00181858],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [3.74627573e-03, 1.45479750e-03, 3.69175425e-03, 1.63661587e-03],
            rtol=1e-5)

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

        rec.fit(data, disp=False)
        self.assertAlmostEqual(rec.i_loss_,
                               0.486928728372, delta=1e-5)
        self.assertAlmostEqual(rec.f_loss_,
                               0.12337278276141388, delta=1e-5)

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
