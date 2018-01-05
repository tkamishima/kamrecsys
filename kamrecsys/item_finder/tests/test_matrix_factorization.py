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
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 0.7665586528005817,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 0.69314718055994495,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 1.5567153484891236,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 1.1832627694252338,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 3.6444649416252157,
            rtol=1e-5)

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
        assert_allclose(grad[0], -0.0898075009641, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0498308105, 0.0196757414, -0.0076319979, -0.0152389619],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0472451789, -0.0160642908, -0.060056816, 0.0394887188],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0732353188, -0.0916441815, 0.0231358636, -0.0058694524],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0099831228, 0.084085786, 0.0506473455, -0.0145630614],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -0.2, rtol=1e-5)
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
        assert_allclose(grad[0], 0.293307149076, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1662875682, 0.0347389951, 0.0347389951, 0.0342928051],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0671799483, 0.0345159001, 0.0009594717, 0.0678492334],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.1662875682, 0.1662875682, 0.0347389951, 0.0347389951],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0009594717, 0.0009594717, 0.0678492334, 0.0678492334],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 0.277458370362, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1570038779, 0.0316695778, 0.0314414819, 0.0311769487],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.0663394591, 0.0347162214, 0.0014015648, 0.068637947],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0323267015, 0.1588557297, 0.0072228044, 0.0333362444],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-5.9551391499e-04, -7.9916718879e-05, 3.2930084632e-02,
             6.6971280375e-02], rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 0.29999005844, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.1681440917, 0.0346287714, 0.0344432652, 0.0342580692],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0701838766, 0.0371289966, 0.0040733991, 0.0710180345],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.3648799527, 0.0818505402, 0.0733684416, 0.0318517013],
            rtol=1e-5)
        assert_allclose(
            grad[-4:], [0.0038126227, 0.0014808065, 0.206424416, 0.0683328493],
            rtol=1e-5)

    def test_class(self):

        data = load_movielens_mini()
        data.binarize_score()

        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec.fit(data)

        assert_allclose(
            rec.fit_results_['initial_loss'], 0.7665586528005817, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.07714505638740253, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 0.999880367504, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 0.976596331684, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 0.886485084449, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [0.999880367504, 0.976596331684, 0.886485084449],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 0.999880367504, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 0.985475882857, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 0.851425801703, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 0.890682628776, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 0.976596331684, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 0.932930852359, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 0.998741832467, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 0.962962765249, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 0.886485084449, rtol=1e-5)


        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [0.999880367504, 0.985475882857, 0.851425801703, 0.890682628776,
             0.976596331684, 0.932930852359, 0.998741832467, 0.962962765249,
             0.886485084449],
            rtol=1e-5)


class TestImplicitLogisticPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

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
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 1.3111240714555008,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 0.69314718055994518,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 3.1817153484891296,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 2.516800033025768,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 7.466231131678393,
            rtol=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

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
        assert_allclose(grad[0], 0.369344884225, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0163795867, 0.0766087539, 0.0494371859, 0.0523347865],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.0079542759, 0.0300129557, 0.0221887768, 0.0444588993],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.00388247, -0.0106748979, 0.047492578, -0.0228685915],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0284519191, 0.0318876738, 0.0234999572, -0.0061334269],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 0.125, atol=1e-5)
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
        assert_allclose(grad[0], 0.618307149076, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0010152455, 0.1010152455, 0.1010152455, 0.0760152455],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0261825668, 0.0636825668, 0.0511825668, 0.0636825668],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0010152455, 0.0010152455, 0.1010152455, 0.1010152455],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0511825668, 0.0511825668, 0.0636825668, 0.0636825668],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 0.603873544619, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.0036235458, 0.0968966988, 0.0973867802, 0.0728492687],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [0.0262076293, 0.0640570655, 0.051891369, 0.0647118918],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0002012168, -0.0017716939, 0.0202682287, 0.0985633655],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0246493882, 0.0504098876, 0.030967057, 0.0630452251],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 0.624990535043, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0014799603, 0.1012948868, 0.101109805, 0.0759247155],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.0285178333, 0.066295708, 0.0540735689, 0.0668514181],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0055522045, 0.00185136, 0.2222005993, 0.0468513936],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.1678131716, 0.0514809763, 0.2142578479, 0.0641662329],
            rtol=1e-5)

    def test_class(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec.fit(data)
        assert_allclose(
            rec.fit_results_['initial_loss'], 1.3111240714555008, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.21148605953543897, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 0.98444253128, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 0.225589860753, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 0.090758250123, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [0.98444253128, 0.225589860753, 0.090758250123],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 0.98444253128, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 0.924884069088, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 0.952482724921, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 0.618744630907, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 0.225589860753, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 0.2295914768, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 0.00080319875342, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 0.0396585424477, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 0.0907582501228, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [0.98444253128, 0.924884069088, 0.952482724921,
             0.618744630907, 0.225589860753, 0.2295914768,
             0.00080319875342, 0.0396585424477, 0.0907582501228],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
