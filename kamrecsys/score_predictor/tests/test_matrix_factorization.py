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
    TestCase,
    run_module_suite,
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_string_equal)
import numpy as np

from sklearn.utils import check_random_state

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.score_predictor import PMF

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec._rng = check_random_state(rec.random_state)
        ev = data.event
        sc = data.score
        n_objects = data.n_objects
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 0.7291206184050988,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 15.7,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 2.4166666666666665,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 1.14175,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 62.720038744000014,
            rtol=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec._rng = check_random_state(rec.random_state)
        ev = data.event
        sc = data.score
        n_objects = data.n_objects
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -0.0212968638573, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0363059823, 0.0167339885, -0.0260526426, -0.0018141656],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0757162534, -0.02525473, -0.0393169069, 0.0252035637],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0047896573, -0.0979586199, 0.012605792, -0.0028619178],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0108829851, 0.0536257719, 0.0211630636, -0.0097388071],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -3.83333333333, rtol=1e-5)
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
        assert_allclose(grad[0], 1.16666666667, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.4685185185, 0.1351851852, 0.1018518519, 0.2018518519],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.2685185185, 0.1685185185, 0.0685185185, 0.2018518519],
        rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.4685185185, 0.4685185185, 0.1351851852, 0.1351851852],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0685185185, 0.0685185185, 0.2018518519, 0.2018518519],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 0.02, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0166666667, 0.0435185185, -0.0096296296, 0.0572222222],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.092962963, 0.0731481481, -0.03, 0.1201851852],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.0024074074, -0.0148148148, 0.0095925926, 0.0451851852],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0162962963, -0.0314814815, 0.0587037037, 0.1185185185],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 7.82805333333, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [2.6081481481, 0.5675096296, 0.4974844444, 1.1014859259],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [1.6601851852, 0.868142963, 1.0316474074, 0.9287518519],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [5.6287888889, 1.2421851852, 1.2161727704, 0.3079238519],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [3.3865753481, 1.0290548148, 2.8936547259, 0.9260666667],
            rtol=1e-5)

    def test_class(self):

        data = load_movielens_mini()

        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec.fit(data)

        assert_allclose(
            rec.fit_results_['initial_loss'], 0.7291206184050988, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.013777730279425669, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 3.997101590073, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 4.24151516373, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 3.498237262002, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [3.997101590073, 4.24151516373, 3.498237262002],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 3.997101590073, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 4.977093755711, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 3.62779086784, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 3.683330861026, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 4.24151516373, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 3.70802937382, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 3.521554946725, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 4.000964107588, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 3.498237262002, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [3.997101590073, 4.977093755711, 3.62779086784,
             3.683330861026, 4.24151516373, 3.70802937382,
             3.521554946725, 4.000964107588, 3.498237262002],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
