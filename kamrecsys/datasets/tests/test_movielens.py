#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange
import six

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

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestLoadMovielens100k(unittest.TestCase):
    def test_load_movielens100k(self):
        from kamrecsys.datasets import load_movielens100k

        data = load_movielens100k()
        self.assertListEqual(
            sorted(data.__dict__.keys()),
            sorted(['event_otypes', 'n_otypes', 'n_events', 'n_score_levels',
                    'n_scores', 'feature', 'event', 'iid', 'event_feature',
                    'score', 'eid', 'n_objects', 'n_stypes', 's_event',
                    'score_domain']))
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 100000)
        self.assertEqual(data.s_event, 2)
        assert_array_equal(data.n_objects, [943, 1682])
        self.assertEqual(data.n_stypes, 1)
        assert_array_equal(data.score_domain, [1., 5., 1.])
        assert_array_equal(
            data.event[:5],
            [[195, 241], [185, 301], [21, 376], [243, 50], [165, 345]])
        assert_array_equal(
            data.event[-5:],
            [[879, 475], [715, 203], [275, 1089], [12, 224], [11, 202]])
        assert_array_equal(data.eid[0][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[0][-5:], [939, 940, 941, 942, 943])
        assert_array_equal(data.eid[1][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:], [1678, 1679, 1680, 1681, 1682])
        self.assertListEqual(data.event_feature.dtype.descr,
                             [('timestamp', '<i8')])
        assert_array_equal(
            data.event_feature[:5].astype(np.int),
            [881250949, 891717742, 878887116, 880606923, 886397596])
        assert_array_equal(
            data.event_feature[-5:].astype(np.int),
            [880175444, 879795543, 874795795, 882399156, 879959583])
        assert_array_equal(data.score[:5], [3., 3., 1., 2., 1.])
        assert_array_equal(data.score[-5:], [3., 5., 1., 2., 3.])
        assert_array_equal(
            [data.iid[0][1], data.iid[0][2], data.iid[0][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[1][1], data.iid[1][2], data.iid[1][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[0][943], data.iid[0][942], data.iid[0][900]],
            [942, 941, 899])
        assert_array_equal(
            [data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]],
            [1681, 1680, 999])
        if six.PY3:
            self.assertEqual(
                str(data.feature[0][:3]),
                "[(24, 0, 19, b'85711') "
                "(53, 1, 1, b'94043') "
                "(23, 0, 20, b'32067')]")
            self.assertEqual(
                str(data.feature[0][-3:]),
                "[(20, 0, 18, b'97229') "
                "(48, 1, 12, b'78209') "
                "(22, 0, 18, b'77841')]")
        else:
            self.assertEqual(
                str(data.feature[0][:3]),
                "[(24, 0, 19, '85711') "
                "(53, 1, 1, '94043') "
                "(23, 0, 20, '32067')]")
            self.assertEqual(
                str(data.feature[0][-3:]),
                "[(20, 0, 18, '97229') "
                "(48, 1, 12, '78209') "
                "(22, 0, 18, '77841')]")
        self.assertEqual(len(data.feature[0]), 943)
        if six.PY3:
            self.assertEqual(
                str(data.feature[1][:1]),
                "[ (\"b'Toy Story (1995)'\", 1, 1, 1995, "
                "[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
                "b'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)')]")
            self.assertEqual(
                str(data.feature[1][-1:]),
                "[ (\"b'Scream of Stone (Schrei aus Stein) (1991)'\", 8, 3, 1996, "
                "[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
                "b'http://us.imdb.com/M/title-exact?Schrei%20aus%20Stein%20(1991)')]")
        else :
            self.assertEqual(
                str(data.feature[1][:1]),
                "[ (u'Toy Story (1995)', 1, 1, 1995, "
                "[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
                "'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)')]")
            self.assertEqual(
                str(data.feature[1][-1:]),
                "[ (u'Scream of Stone (Schrei aus Stein) (1991)', 8, 3, 1996, "
                "[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
                "'http://us.imdb.com/M/title-exact?Schrei%20aus%20Stein%20(1991)')]")
        self.assertEqual(len(data.feature[1]), 1682)

    def test_MOVIELENS100K_INFO(self):
        from kamrecsys.datasets import MOVIELENS100K_INFO

        assert_array_equal(
            MOVIELENS100K_INFO['user_occupation'],
            ['None', 'Other', 'Administrator', 'Artist', 'Doctor', 'Educator',
             'Engineer', 'Entertainment', 'Executive', 'Healthcare',
             'Homemaker', 'Lawyer', 'Librarian', 'Marketing', 'Programmer',
             'Retired', 'Salesman', 'Scientist', 'Student', 'Technician',
             'Writer'])
        assert_array_equal(
            MOVIELENS100K_INFO['item_genre'],
            ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
             'War', 'Western'])


class TestLoadMovielens1m(unittest.TestCase):
    def test_load_movielens1m(self):
        from kamrecsys.datasets import load_movielens1m

        data = load_movielens1m()
        self.assertListEqual(
            sorted(data.__dict__.keys()),
            sorted([
                'event_otypes', 'n_otypes', 'n_events',
                'n_score_levels', 'n_scores',
                'feature', 'event', 'iid',
                'event_feature', 'score', 'eid', 'n_objects',
                'n_stypes',
                's_event', 'score_domain']))
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 1000209)
        self.assertEqual(data.s_event, 2)
        assert_array_equal(data.n_objects, [6040, 3706])
        self.assertEqual(data.n_stypes, 1)
        assert_array_equal(data.score_domain, [1., 5., 1.])
        assert_array_equal(
            data.to_eid_event(data.event)[:5],
            [[1, 1193], [1, 661], [1, 914], [1, 3408],
             [1, 2355]])
        assert_array_equal(
            data.to_eid_event(data.event)[-5:],
            [[6040, 1091], [6040, 1094], [6040, 562], [6040, 1096],
             [6040, 1097]])
        assert_array_equal(data.eid[0][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[0][-5:], [6036, 6037, 6038, 6039, 6040])
        assert_array_equal(data.eid[1][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:], [3948, 3949, 3950, 3951, 3952])
        self.assertEqual(str(data.event_feature.dtype),
                         "[('timestamp', '<i8')]")
        self.assertEqual(
            str(data.event_feature[:5]),
            "[(978300760,) (978302109,) (978301968,)"
            " (978300275,) (978824291,)]")
        self.assertEqual(
            str(data.event_feature[-5:]),
            "[(956716541,) (956704887,) (956704746,)"
            " (956715648,) (956715569,)]")
        assert_array_equal(data.score[:5], [5., 3., 3., 4., 5.])
        assert_array_equal(data.score[-5:], [1., 5., 5., 4., 4.])
        assert_array_equal(
            [data.iid[0][1], data.iid[0][2], data.iid[0][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[1][1], data.iid[1][2], data.iid[1][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[0][943], data.iid[0][942], data.iid[0][900]],
            [942, 941, 899])
        assert_array_equal(
            [data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]],
            [1545, 1544, 936])
        if six.PY3:
            self.assertEqual(
                str(data.feature[0][:3]),
                "[(1, 0, 10, b'48067') (0, 6, 16, b'70072') "
                "(0, 2, 15, b'55117')]")
            self.assertEqual(
                str(data.feature[0][-3:]),
                "[(1, 6, 1, b'14706') (1, 4, 0, b'01060') "
                "(0, 2, 6, b'11106')]")
        else:
            self.assertEqual(
                str(data.feature[0][:3]),
                "[(1, 0, 10, '48067') (0, 6, 16, '70072')"
                " (0, 2, 15, '55117')]")
            self.assertEqual(
                str(data.feature[0][-3:]),
                "[(1, 6, 1, '14706') (1, 4, 0, '01060') (0, 2, 6, '11106')]")
        self.assertEqual(len(data.feature[0]), 6040)
        if six.PY3:
            self.assertEqual(
                str(data.feature[1][:1]),
                "[ (b'Toy Story (1995)', 1995, "
                "[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]")
            self.assertEqual(
                str(data.feature[1][-1:]),
                "[ (b'Contender, The (2000)', 2000, "
                "[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]")
        else:
            self.assertEqual(
                str(data.feature[1][:1]),
                "[ (u'Toy Story (1995)', 1995, "
                "[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]")
            self.assertEqual(
                str(data.feature[1][-1:]),
                "[ (u'Contender, The (2000)', 2000, "
                "[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]")
        self.assertEqual(len(data.feature[1]), 3706)

    def test_MOVIELENS1M_INFO(self):
        from kamrecsys.datasets import MOVIELENS1M_INFO

        assert_array_equal(
            MOVIELENS1M_INFO['user_age'],
            ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+'])
        assert_array_equal(
            MOVIELENS1M_INFO['user_occupation'],
            ['Other or Not Specified', 'Academic/Educator', 'Artist',
             'Clerical/Admin', 'College/Grad Student', 'Customer Service',
             'Doctor/Health Care', 'Executive/Managerial', 'Farmer',
             'Homemaker', 'K-12 Student', 'Lawyer', 'Programmer', 'Retired',
             'Sales/Marketing', 'Scientist', 'Self-Employed',
             'Technician/Engineer', 'Tradesman/Craftsman', 'Unemployed',
             'Writer'])
        assert_array_equal(
            MOVIELENS1M_INFO['item_genre'],
            ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
             'War', 'Western'])

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
