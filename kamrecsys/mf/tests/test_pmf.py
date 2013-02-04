#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

##### Test Classes #####

class TestEventScorePredictor(unittest.TestCase):
    def runTest(self):
        import numpy as np
        from kamrecsys.datasets import load_movielens_mini
        from kamrecsys.mf.pmf import EventScorePredictor

        np.random.seed(1234)
        data = load_movielens_mini()

        with self.assertRaises(ValueError):
            EventScorePredictor(C=0.1, k=0)

        recommender = EventScorePredictor(C=0.1, k=2)
        self.assertDictEqual(
            vars(recommender),
                {'C': 0.1, 'n_otypes': 0, 'bu_': None, 'bi_': None, 'k': 2,
                 'p_': None, 'q_': None, '_coef': None, 'f_loss_': np.inf,
                 'iid': None, 'i_loss_': np.inf, 'eid': None,
                 'n_objects': None, '_dt': None, 'mu_': None})

        recommender.fit(data, disp=False, gtol=1e-03)
        self.assertAlmostEqual(recommender.predict((1, 7)), 4.00074631485)
        self.assertAlmostEqual(recommender.predict((1, 9)), 4.98286035672)
        self.assertAlmostEqual(recommender.predict((1, 11)), 3.44741578214)
        self.assertAlmostEqual(recommender.predict((3, 7)), 3.89716397809)
        self.assertAlmostEqual(recommender.predict((3, 9)), 4.20400627475)
        self.assertAlmostEqual(recommender.predict((3, 11)), 3.66306486366)
        self.assertAlmostEqual(recommender.predict((5, 7)), 3.7468479513)
        self.assertAlmostEqual(recommender.predict((5, 9)), 3.96853184458)
        self.assertAlmostEqual(recommender.predict((5, 11)), 3.60148694779)
        x = np.array([[1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_array_almost_equal(
            recommender.predict(x),
            [4.00074631, 4.98286036, 3.44741578, 3.89716398, 4.20400627,
             3.66306486, 3.74684795, 3.96853184, 3.60148695])

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
