#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest

##### Test Classes #####

class TestSushi3Class(unittest.TestCase):
    def test_load_sushi3_score(self):
        from .. import load_sushi3b_score

        data = load_sushi3b_score()

        self.assertListEqual(data.__dict__.keys(),
                             ['event_otypes', 'n_otypes', 'n_events',
                              'feature', 'event', 'iid',
                              'event_feature', 'score', 'eid', 'n_objects',
                              'n_stypes',
                              's_event', 'score_domain'])
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 50000)
        self.assertEqual(data.s_event, 2)
        assert_array_equal(data.n_objects, [5000, 100])

        # events
        self.assertEqual(data.n_stypes, 1)
        assert_array_equal(data.score_domain, [0., 4.])
        assert_array_equal(
            data.event[:5],
            [[0, 1], [0, 3], [0, 4], [0, 12], [0, 44]])
        assert_array_equal(
            data.event[3220:3225],
            [[322, 4], [322, 7], [322, 10], [322, 11], [322, 20]])
        assert_array_equal(
            data.event[-5:],
            [[4999, 19], [4999, 23], [4999, 25], [4999, 42], [4999, 47]])
        assert_array_equal(data.eid[0][:5], [0, 1, 2, 3, 4])
        self.assertEqual(data.eid[0][322], 322)
        assert_array_equal(data.eid[0][-5:], [4995, 4996, 4997, 4998, 4999])
        assert_array_equal(data.eid[1][:5], [0, 1, 2, 3, 4])
        assert_array_equal(data.eid[1][-5:], [95, 96, 97, 98, 99])
        assert_array_equal(data.score[:5], [4., 0., 2., 3., 3.])
        assert_array_equal(data.score[3220:3230],
                           [1., 0., 0., 1., 0., 2., 2., 1., 0., 4.])
        assert_array_equal(data.score[-5:], [0., 2., 4., 2., 0.])

        # users
        self.assertEqual(data.feature[0][322]['original_uid'], 0)
        self.assertEqual(data.feature[0][322]['gender'], 0)
        self.assertEqual(data.feature[0][322]['age'], 2)
        self.assertEqual(data.feature[0][322]['answer_time'], 785)
        self.assertEqual(data.feature[0][322]['child_prefecture'], 26)
        self.assertEqual(data.feature[0][322]['child_region'], 6)
        self.assertEqual(data.feature[0][322]['child_ew'], 1)
        self.assertEqual(data.feature[0][322]['current_prefecture'], 8)
        self.assertEqual(data.feature[0][322]['current_region'], 3)
        self.assertEqual(data.feature[0][322]['current_ew'], 0)
        self.assertEqual(data.feature[0][322]['is_moved'], 1)

        # items
# todo: sushi name in unicode
        #self.assertEqual(data.feature[1][8]['name'], u'とろ')
        self.assertEqual(data.feature[1][8]['is_maki'], 1)
        self.assertEqual(data.feature[1][8]['is_seafood'], 0)
        self.assertEqual(data.feature[1][8]['genre'], 1)
        self.assertAlmostEqual(data.feature[1][8]['heaviness'],
                               0.551854655563967)
        self.assertAlmostEqual(data.feature[1][8]['frequency'],
                               2.05753217259652)
        self.assertAlmostEqual(data.feature[1][8]['price'],
                               4.48545454545455)
        self.assertAlmostEqual(data.feature[1][8]['supply'],
                               0.8)

##### Main routine #####

if __name__ == '__main__':
    unittest.main()
