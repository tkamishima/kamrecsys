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

class TestFlixsterClass(unittest.TestCase):
    def test_load_flixster_rating(self):
        from .. import load_flixster_rating

        data = load_flixster_rating()

        self.assertListEqual(data.__dict__.keys(),
                             ['event_otypes', 'n_otypes', 'n_events',
                              'feature', 'event', 'iid',
                              'event_feature', 'score', 'eid', 'n_objects',
                              'n_stypes',
                              's_event', 'score_domain'])
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 8196077)
        self.assertEqual(data.s_event, 2)
        assert_array_equal(data.n_objects, [147612, 48794])

        # events
        self.assertEqual(data.n_stypes, 1)
        assert_array_equal(data.score_domain, [0.5, 5.0])
        assert_array_equal(
            data.event[:5],
            [[124545, 57], [124545, 665], [124545, 969],
            [124545, 1650], [124545, 2230]])
        assert_array_equal(
            data.event[-5:],
            [[14217, 28183], [14217, 36255], [14217, 37636],
            [14217, 40326], [14217, 48445]])
        assert_array_equal(data.eid[0][:5],
                           [6, 7, 8, 9, 11])
        assert_array_equal(data.eid[0][-5:],
                           [1049477, 1049489, 1049491, 1049494, 1049508])
        assert_array_equal(data.eid[1][:5],
                           [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:],
                           [66712, 66714, 66718, 66725, 66726])
        assert_array_equal(data.score[:5], [1.5, 1.0, 2.0, 1.0, 5.0])
        assert_array_equal(data.score[-5:], [5.0, 4.0, 3.0, 4.0, 5.0])

##### Main routine #####

if __name__ == '__main__':
    unittest.main()