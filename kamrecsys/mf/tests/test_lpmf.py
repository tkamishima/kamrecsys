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
                               1.3445493746024519, delta=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               0.69314718055994518, delta=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               3.2298971666709479, delta=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               2.5557470027227374, delta=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        self.assertAlmostEqual(rec.loss(rec._coef, ev, n_objects),
                               7.6233440104662717, delta=1e-5)

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
        self.assertAlmostEqual(grad[0], 0.369344884225, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0164132567, 0.0766020199, 0.0494304519, 0.0523213185],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0079795284, 0.0300003295, 0.0221719417, 0.044446273],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.0038983432, -0.0106347977, 0.0474443387, -0.0228580645],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.028416647,  0.031852457, 0.0234708758, -0.0061293161],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.125, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0625, 0.0375, 0.0375, 0.0125],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.025, 0.0125, 0., 0.0125],
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
        self.assertAlmostEqual(grad[0], 0.618307149076, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0009815755, 0.1009815755, 0.1009815755, 0.0759815755],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0261488967, 0.0636488967, 0.0511488967, 0.0636488967],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0009815755, 0.0009815755, 0.1009815755, 0.1009815755],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0511488967, 0.0511488967, 0.0636488967, 0.0636488967],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.603873544619, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0036235458, 0.0968933318, 0.0973800461, 0.0728391677],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.0261537573, 0.0639998264, 0.051830763, 0.0646479187],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0001843817, -0.001805364, 0.0202513936, 0.0985296955],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0246426542, 0.0503762175, 0.030960323, 0.0630115551],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        self.assertAlmostEqual(grad[0], 0.624990535043, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0014530243, 0.1012713178, 0.101089603, 0.0759078805],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0284538602, 0.0662266844, 0.0539994948, 0.0667722935],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0054511944, 0.0018176899, 0.2220968956, 0.0468177236],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.1677438113, 0.0514540403, 0.2141894977, 0.0641359299],
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
             'p_': None, 'q_': None, '_coef': None, 'mu_': None, '_dt': None,
             'fit_results_': {'initial_loss': np.inf, 'final_loss': np.inf},
             'iid': None, 'eid': None, 'tol': 1e-03, 'n_objects': None,
             'maxiter': 200, 'random_state': 1234, '_rng': None})

        rec.fit(data, disp=False)
        self.assertAlmostEqual(rec.fit_results_['initial_loss'],
                               1.3445493746, delta=1e-5)
        self.assertAlmostEqual(rec.fit_results_['final_loss'],
                               0.30760976439390564, delta=1e-5)

        # single prediction
        self.assertAlmostEqual(rec.predict((1, 7)),
                               0.984542941978, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 9)),
                               0.934243410501, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 11)),
                               0.957504275371, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 7)),
                               0.590183096344, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 9)),
                               0.202161811915, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 11)),
                               0.22899801898, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 7)),
                               0.000727177442114, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 9)),
                               0.0399527433316, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 11)),
                               0.08288774155, delta=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [0.984542941978, 0.934243410501, 0.957504275371,
             0.590183096344, 0.202161811915, 0.22899801898,
             0.000727177442114, 0.0399527433316, 0.08288774155],
            rtol=1e-5)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
