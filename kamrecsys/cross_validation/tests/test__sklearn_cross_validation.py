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


class TestKFold(unittest.TestCase):

    def test_class(self):
        from .. import KFold

        kf = KFold(6, interlace=True)
        self.assertEqual(len(kf), 3)
        self.assertEqual(kf.__repr__(),
                         "kamrecsys."
                         "cross_validation._sklearn_cross_validation."
                         "KFold(n=6, n_folds=3, interlace=True, "
                         "shuffle=False, random_state=None)")
        iter = kf.__iter__()
        train_index, test_index = iter.next()
        assert_array_equal(train_index, [1, 2, 4, 5])
        assert_array_equal(test_index, [0, 3])
        train_index, test_index = iter.next()
        assert_array_equal(train_index, [0, 2, 3, 5])
        assert_array_equal(test_index, [1, 4])
        train_index, test_index = iter.next()
        assert_array_equal(train_index, [0, 1, 3, 4])
        assert_array_equal(test_index, [2, 5])

        kf = KFold(5, n_folds=2, interlace=True)
        self.assertEqual(len(kf), 2)
        iter = kf.__iter__()
        train_index, test_index = iter.next()
        assert_array_equal(train_index, [1, 3])
        assert_array_equal(test_index, [0, 2, 4])
        train_index, test_index = iter.next()
        assert_array_equal(train_index, [0, 2, 4])
        assert_array_equal(test_index, [1, 3])


if __name__ == '__main__':
    unittest.main()
