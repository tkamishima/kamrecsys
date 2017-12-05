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
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)

import numpy as np
from scipy import sparse as sparse
from sklearn.utils import check_random_state

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.item_finder import LogisticPMF, ImplicitLogisticPMF

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestLogisticPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        data.binarize_score(3)
        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

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
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               0.79866250328432664, delta=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               0.69314718055994495, delta=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               1.6048971666709417, delta=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               1.2222097391222033, delta=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        self.assertAlmostEqual(rec.loss(rec._coef, ev, sc, n_objects),
                               3.8015778204130948, delta=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        data.binarize_score()
        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)
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
        self.assertAlmostEqual(grad[0], -0.0898075009641, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0498375445, 0.0196824754, -0.0076252639, -0.0152406454],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.047243776, -0.016061485, -0.0600631291, 0.0394999422],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0732194455, -0.0916040813, 0.0230876243, -0.0058589254],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0100183948, 0.0840505693, 0.0506182641, -0.0145589506],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], -0.2, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0., 0., 0., -0.0333333333],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0333333333, -0.0166666667, -0.0666666667, 0.0166666667],
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
        self.assertAlmostEqual(grad[0], 0.293307149076, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1662538982, 0.0347053251, 0.0347053251, 0.034259135],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0671462783, 0.0344822301, 0.0009258017, 0.0678155634],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.1662538982, 0.1662538982, 0.0347053251, 0.0347053251],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0009258017, 0.0009258017, 0.0678155634, 0.0678155634],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], 0.277458370362, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1570038779, 0.0316662108, 0.0314347479, 0.0311668477],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.0662855871, 0.0346589824, 0.0013409587, 0.068573974],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0323098665, 0.1588220597, 0.0072059694, 0.0333025744],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0006022479, -0.0001135868, 0.0329233506, 0.0669376103],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        self.assertAlmostEqual(grad[0], 0.29999005844, delta=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1681171557, 0.0346052024, 0.0344230632, 0.0342412342],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0701199036, 0.037059973, 0.003999325, 0.0709389099],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.3647789426, 0.0818168701, 0.0732647379, 0.0318180313],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0037432625, 0.0014538705, 0.2063560658, 0.0683025463],
            rtol=1e-5)

    def test_class(self):

        data = load_movielens_mini()
        data.binarize_score()

        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec.fit(data)

        self.assertAlmostEqual(rec.fit_results_['initial_loss'],
                               0.79866250328432664, delta=1e-5)
        self.assertAlmostEqual(rec.fit_results_['final_loss'],
                               0.13320756749404694, delta=1e-5)

        # single prediction
        self.assertAlmostEqual(rec.predict((1, 7)),
                               0.9999001529839445, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 9)),
                               0.9812735864286053, delta=1e-5)
        self.assertAlmostEqual(rec.predict((1, 11)),
                               0.8573434778608382, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 7)),
                               0.8960621219290574, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 9)),
                               0.9757212407381008, delta=1e-5)
        self.assertAlmostEqual(rec.predict((3, 11)),
                               0.9363199952844578, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 7)),
                               0.9988443342984381, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 9)),
                               0.9573433592687439, delta=1e-5)
        self.assertAlmostEqual(rec.predict((5, 11)),
                               0.8913562407390253, delta=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [0.9999001529839445, 0.9812735864286053, 0.8573434778608382,
             0.8960621219290574, 0.9757212407381008, 0.9363199952844578,
             0.9988443342984381, 0.9573433592687439, 0.8913562407390253],
            rtol=1e-5)


class TestImplicitLogisticPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        n_objects = data.n_objects
        ev = sparse.coo_matrix(
            (np.ones(data.n_events, dtype=int),
             (data.event[:, 0], data.event[:, 1])), shape=n_objects)
        ev = ev.tocsr()
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

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, tol=1e-03, random_state=1234)

        rec._rng = check_random_state(rec.random_state)
        n_objects = data.n_objects
        ev = sparse.coo_matrix(
            (np.ones(data.n_events, dtype=int),
             (data.event[:, 0], data.event[:, 1])), shape=n_objects)
        ev = ev.tocsr()
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

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, tol=1e-03, random_state=1234)

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
    run_module_suite()
