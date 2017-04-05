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

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.recommender import BaseRecommender, BaseEventRecommender

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class Recommender(BaseRecommender):

    def predict(self, eev):
        pass

class EventRecommender(BaseEventRecommender):

    def __init__(self):
        super(EventRecommender, self).__init__()

    def raw_predict(self, ev):

        return(np.zeros(ev.shape[0]))

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseRecommender(TestCase):

    def test_class(self):

        # setup
        data = load_movielens_mini()

        # __init__()
        rec = Recommender(random_state=1234)

        self.assertEqual(rec.n_otypes, 0)
        self.assertIsNone(rec.n_objects)
        self.assertIsNone(rec.eid)
        self.assertIsNone(rec.iid)
        self.assertEqual(rec.random_state, 1234)
        self.assertIsNone(rec._rng)
        self.assertDictEqual(rec.fit_results_, {})

        # _set_object_info()
        rec._set_object_info(data)

        self.assertEqual(rec.n_otypes, 2)
        assert_array_equal(rec.n_objects, [8, 10])
        assert_array_equal(rec.eid, data.eid)
        assert_array_equal(rec.iid, data.iid)

        # to_eid()
        self.assertEqual(rec.to_eid(0, 0), 1)
        self.assertEqual(rec.to_eid(1, 0), 1)

        # to_iid()
        self.assertEqual(rec.to_iid(0, 1), 0)
        self.assertEqual(rec.to_iid(1, 1), 0)


class TestBaseEventRecommender(TestCase):

    def test_class(self):

        # setup
        data = load_movielens_mini()

        rec = EventRecommender()
        rec._set_object_info(data)

        # _set_object_info
        rec._set_event_info(data)

        assert_array_equal(rec.event_otypes, [0, 1])
        self.assertEqual(rec.s_event, 2)

        # predict
        self.assertEqual(rec.predict([0, 0]).ndim, 0)
        self.assertEqual(rec.predict([[0, 0]]).ndim, 0)
        self.assertEqual(rec.predict([[0, 0], [0, 1]]).ndim, 1)
        assert_array_equal(rec.predict([[0, 0], [0, 1]]).shape, (2,))
        with self.assertRaises(TypeError):
            rec.predict([[0]])

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
