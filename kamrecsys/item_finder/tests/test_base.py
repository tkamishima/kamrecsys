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

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.item_finder import (
    BaseImplicitItemFinder, BaseExplicitItemFinder)

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class ExplicitItemFinder(BaseExplicitItemFinder):

    def __init__(self):
        super(ExplicitItemFinder, self).__init__(random_state=1234)

    def raw_predict(self, ev):
        pass


class ImplicitItemFinder(BaseImplicitItemFinder):

    def __init__(self):
        super(ImplicitItemFinder, self).__init__(random_state=1234)

    def raw_predict(self):
        pass


class TestBaseExplicitItemFinder(TestCase):

    def test_fit(self):
        rec = ExplicitItemFinder()
        data = load_movielens_mini()
        true_sc = np.where(data.score <= 3, 0, 1)
        data.binarize_score()

        rec.fit(data, event_index=(0, 1))

        # check whether score info is correctly set
        assert_allclose(rec.score_domain, [0, 1, 1])
        assert_allclose(rec.score, true_sc)
        assert_equal(rec.n_score_levels, 2)

        # error checker
        data.score_domain = (0, 5, 1)
        with assert_raises(ValueError):
            rec.fit(data)
        data.score_domain = (0, 1, 1)

        data.n_score_levels = 3
        with assert_raises(ValueError):
            rec.fit(data)
        data.n_score_levels = 2

        data.score = np.zeros(30)
        with assert_raises(ValueError):
            rec.fit(data)
        data.score = np.ones(30)
        with assert_raises(ValueError):
            rec.fit(data)
        data.score = np.repeat(np.arange(3), 10)
        with assert_raises(ValueError):
            rec.fit(data)

    def test_class(self):
        rec = ExplicitItemFinder()
        data = load_movielens_mini()
        true_sc = np.where(data.score <= 3, 0, 1)
        data.binarize_score()
        rec.fit(data, event_index=(0, 1))

        # get_score()
        assert_allclose(rec.get_score(), true_sc)

        # remove_data
        rec.remove_data()
        self.assertIsNone(rec.score)
        assert_allclose(rec.score_domain, [0, 1, 1])
        self.assertEqual(rec.n_score_levels, 2)


class TestBaseImplicitItemFinder(TestCase):

    def test__get_event_array(self):
        rec = ImplicitItemFinder()
        data = load_movielens_mini()
        data.filter_event(
            np.logical_and(data.event[:, 0] < 5, data.event[:, 1] < 5))

        rec.fit(data)
        ev, n_objects = rec.get_event_array(sparse_type='array')
        assert_array_equal(
            ev[:5, :5],
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
             [1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])

        ev2, n_objects = rec.get_event_array('csr')
        assert_array_equal(ev, ev2.todense())

        ev2, n_objects = rec.get_event_array('csc')
        assert_array_equal(ev, ev2.todense())

        ev2, n_objects = rec.get_event_array('lil')
        assert_array_equal(ev, ev2.todense())


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
