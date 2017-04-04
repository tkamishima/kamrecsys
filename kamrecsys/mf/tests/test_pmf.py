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

    def test_loss(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.pmf import EventScorePredictor

        # setup
        data = load_movielens_mini()
        rec = EventScorePredictor(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        ev, sc, n_objects = rec._get_event_and_score(data, (0, 1), 0)
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               0.74652578358324118, delta=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               15.699999999999999, delta=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               2.4648484848484848, delta=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               1.1806969696969696, delta=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               62.877151622787892, delta=1e-5)

    def test_grad_loss(self):
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.pmf import EventScorePredictor

        # setup
        data = load_movielens_mini()
        rec = EventScorePredictor(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        ev, sc, n_objects = rec._get_event_and_score(data, (0, 1), 0)
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], -0.0212968638573, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0363138387, 0.0167620468, -0.0260414192, -0.0018029422],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0757083971, -0.0252334057, -0.0393345837, 0.0252304997],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.004779893, -0.0979339524, 0.0125761178, -0.0028554421],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0109046825, 0.0536041084, 0.0211451743, -0.0097362783],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], -3.83333333333, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [-1.2, -0.2, -0.2333333333, -0.4666666667],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.7333333333, -0.3333333333, -0.6, -0.3],
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
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], 1.16666666667, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.4684848485, 0.1351515152, 0.1018181818, 0.2018181818],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.2684848485, 0.1684848485, 0.0684848485, 0.2018181818],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.4684848485, 0.4684848485, 0.1351515152, 0.1351515152],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0684848485, 0.0684848485, 0.2018181818, 0.2018181818],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], 0.02, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0166666667, 0.0435151515, -0.0096363636, 0.0572121212],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.0929090909, 0.0730909091, -0.0300606061, 0.1201212121],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.0024242424, -0.0148484848, 0.0095757576, 0.0451515152],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0163030303, -0.0315151515, 0.0586969697, 0.1184848485],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], 7.82805333333, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [2.6081212121, 0.5674860606, 0.4974642424, 1.1014690909],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [1.6601212121, 0.8680739394, 1.0315733333, 0.9286727273],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [5.6286878788, 1.2421515152, 1.2160690667, 0.3078901818],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [3.3865059879, 1.0290278788, 2.8935863758, 0.9260363636],
            rtol=1e-5)

    def test_class(self):
        import numpy as np
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.pmf import EventScorePredictor

        data = load_movielens_mini()

        rec = EventScorePredictor(C=0.1, k=2, tol=1e-03, random_state=1234)

        self.assertDictEqual(
            vars(rec),
            {'C': 0.1, 'n_otypes': 0, 'bu_': None, 'bi_': None, 'k': 2,
             'p_': None, 'q_': None, '_coef': None, 'mu_': None, '_dt': None,
             'fit_results_': {'initial_loss': np.inf, 'final_loss': np.inf},
             'iid': None, 'eid': None, 'tol': 1e-03, 'n_objects': None,
             'maxiter': 200, 'random_state': 1234, '_rng': None})

        rec.fit(data, disp=False)
        self.assertAlmostEqual(rec.fit_results_['initial_loss'],
                               0.74652578358324106, delta=1e-5)
        self.assertAlmostEqual(rec.fit_results_['final_loss'],
                               0.025638738121075231, delta=1e-5)

        # single prediction
        self.assertAlmostEqual(rec.predict((1, 7)),
                               3.9873641434545979, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 9)),
                               4.9892118821609106, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 11)),
                               3.6480799850368273, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 7)),
                               3.6336318795279228, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 9)),
                               4.2482001235634943, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 11)),
                               3.7236984083417841, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 7)),
                               3.4141968145802597, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 9)),
                               3.9818882049478654, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 11)),
                               3.4710520150321895, delta=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [3.9873641434545979, 4.9892118821609106, 3.6480799850368273,
             3.6336318795279228, 4.2482001235634943, 3.7236984083417841,
             3.4141968145802597, 3.9818882049478654, 3.4710520150321895],
            rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
