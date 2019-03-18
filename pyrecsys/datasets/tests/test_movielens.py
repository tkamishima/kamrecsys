#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest

##### Test Classes #####

class TestLoadMovielens100k(unittest.TestCase):
    def test_load_movielens100k(self):
        from pyrecsys.datasets import load_movielens100k

        data = load_movielens100k()
        self.assertListEqual(data.__dict__.keys(),
            ['event_otypes', 'n_otypes', 'n_events', 'feature', 'event', 'iid',
             'event_feature', 'score', 'eid', 'n_objects', 'n_stypes',
             's_event', 'score_domain'])
        self.assertEqual(str(data.event_otypes), '[0 1]')
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 100000)
        self.assertEqual(data.s_event, 2)
        self.assertEqual(str(data.n_objects), '[ 943 1682]')
        self.assertEqual(data.n_stypes, 1)
        self.assertEqual(str(data.score_domain), '[ 1.  5.]')
        self.assertEqual(
            str(data.event[:5]),
            "[[195 241]\n [185 301]\n [ 21 376]\n [243  50]\n [165 345]]")
        self.assertEqual(
            str(data.event[-5:]),
            "[[ 879  475]\n [ 715  203]\n [ 275 1089]\n"
            " [  12  224]\n [  11  202]]")
        self.assertEqual(str(data.eid[0][:5]), '[1 2 3 4 5]')
        self.assertEqual(str(data.eid[0][-5:]), '[939 940 941 942 943]')
        self.assertEqual(str(data.eid[1][:5]), '[1 2 3 4 5]')
        self.assertEqual(str(data.eid[1][-5:]), '[1678 1679 1680 1681 1682]')
        self.assertListEqual(data.event_feature.dtype.descr,
            [('timestamp', '<i8')])
        self.assertEqual(
            str(data.event_feature[:5]),
            '[(881250949,) (891717742,) (878887116,) (880606923,) (886397596,)]')
        self.assertEqual(
            str(data.event_feature[-5:]),
            '[(880175444,) (879795543,) (874795795,) (882399156,) (879959583,)]')
        self.assertEqual(str(data.score[:5]), '[ 3.  3.  1.  2.  1.]')
        self.assertEqual(str(data.score[-5:]), '[ 3.  5.  1.  2.  3.]')
        self.assertTupleEqual(
            (data.iid[0][1], data.iid[0][2], data.iid[0][3]),
            (0, 1, 2))
        self.assertTupleEqual(
            (data.iid[1][1], data.iid[1][2], data.iid[1][3]),
            (0, 1, 2))
        self.assertTupleEqual(
            (data.iid[0][943], data.iid[0][942], data.iid[0][900]),
            (942, 941, 899))
        self.assertTupleEqual(
            (data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]),
            (1681, 1680, 999))
        self.assertEqual(
            str(data.feature[0][:5]),
            "[(24, 0, 19, '85711') (53, 1, 1, '94043') (23, 0, 20, '32067')\n"
            " (24, 0, 19, '43537') (33, 1, 1, '15213')]")
        self.assertEqual(
            str(data.feature[0][-5:]),
            "[(26, 1, 18, '33319') (32, 0, 2, '02215') (20, 0, 18, '97229')\n"
            " (48, 1, 12, '78209') (22, 0, 18, '77841')]")
        self.assertEqual(len(data.feature[0]), 943)
        self.assertEqual(
            str(data.feature[1][:5]),
            "[ (u'Toy Story (1995)', 1, 1, 1995, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)')\n"
            " (u'GoldenEye (1995)', 1, 1, 1995, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?GoldenEye%20(1995)')\n"
            " (u'Four Rooms (1995)', 1, 1, 1995, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)')\n"
            " (u'Get Shorty (1995)', 1, 1, 1995, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995)')\n"
            " (u'Copycat (1995)', 1, 1, 1995, [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?Copycat%20(1995)')]")
        self.assertEqual(
            str(data.feature[1][-5:]),
            "[ (u\"Mat\' i syn (1997)\", 6, 2, 1998, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Mat%27+i+syn+(1997)')\n"
            " (u'B. Monkey (1998)', 6, 2, 1998, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?B%2E+Monkey+(1998)')\n"
            " (u'Sliding Doors (1998)', 1, 1, 1998, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'http://us.imdb.com/Title?Sliding+Doors+(1998)')\n"
            " (u'You So Crazy (1994)', 1, 1, 1994, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?You%20So%20Crazy%20(1994)')\n"
            " (u'Scream of Stone (Schrei aus Stein) (1991)', 8, 3, 1996, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Schrei%20aus%20Stein%20(1991)')]")
        self.assertEqual(len(data.feature[1]), 1682)

    def test_MOVIELENS100K_INFO(self):
        from pyrecsys.datasets import MOVIELENS100K_INFO

        self.assertEqual(
            str(MOVIELENS100K_INFO['user_occupation']),
            "['None' 'Other' 'Administrator' 'Artist' 'Doctor' 'Educator' 'Engineer'\n"
            " 'Entertainment' 'Executive' 'Healthcare' 'Homemaker' 'Lawyer' 'Librarian'\n"
            " 'Marketing' 'Programmer' 'Retired' 'Salesman' 'Scientist' 'Student'\n"
            " 'Technician' 'Writer']")
        self.assertEqual(
            str(MOVIELENS100K_INFO['item_genre']),
            "['Action' 'Adventure' 'Animation' \"Children's\" 'Comedy' 'Crime'\n"
            " 'Documentary' 'Drama' 'Fantasy' 'Film-Noir' 'Horror' 'Musical' 'Mystery'\n"
            " 'Romance' 'Sci-Fi' 'Thriller' 'War' 'Western']")


class TestLoadMovielens1m(unittest.TestCase):
    def test_load_movielens1m(self):
        from pyrecsys.datasets import load_movielens1m

        data = load_movielens1m()
        self.assertListEqual(
            data.__dict__.keys(),
            ['event_otypes', 'n_otypes', 'n_events', 'feature', 'event', 'iid',
             'event_feature', 'score', 'eid', 'n_objects', 'n_stypes',
             's_event', 'score_domain'])
        self.assertEqual(str(data.event_otypes), '[0 1]')
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 1000209)
        self.assertEqual(data.s_event, 2)
        self.assertEqual(str(data.n_objects), '[6040 3706]')
        self.assertEqual(data.n_stypes, 1)
        self.assertEqual(str(data.score_domain), '[ 1.  5.]')
        self.assertEqual(
            str(data.to_eid_event(data.event)[:5]),
            "[[   1 1193]\n [   1  661]\n [   1  914]\n [   1 3408]\n"
            " [   1 2355]]")
        self.assertEqual(
            str(data.to_eid_event(data.event)[-5:]),
            "[[6040 1091]\n [6040 1094]\n [6040  562]\n [6040 1096]\n"
            " [6040 1097]]")
        self.assertEqual(str(data.eid[0][:5]), '[1 2 3 4 5]')
        self.assertEqual(str(data.eid[0][-5:]), '[6036 6037 6038 6039 6040]')
        self.assertEqual(str(data.eid[1][:5]), '[1 2 3 4 5]')
        self.assertEqual(str(data.eid[1][-5:]), '[3948 3949 3950 3951 3952]')
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
        self.assertEqual(str(data.score[:5]), '[ 5.  3.  3.  4.  5.]')
        self.assertEqual(str(data.score[-5:]), '[ 1.  5.  5.  4.  4.]')
        self.assertTupleEqual(
            (data.iid[0][1], data.iid[0][2], data.iid[0][3]),
            (0, 1, 2))
        self.assertTupleEqual(
            (data.iid[1][1], data.iid[1][2], data.iid[1][3]),
            (0, 1, 2))
        self.assertTupleEqual(
            (data.iid[0][943], data.iid[0][942], data.iid[0][900]),
            (942, 941, 899))
        self.assertTupleEqual(
            (data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]),
            (1545, 1544, 936))
        self.assertEqual(
            str(data.feature[0][:5]),
            "[(1, 0, 10, '48067') (0, 6, 16, '70072') (0, 2, 15, '55117')\n"
            " (0, 4, 7, '02460') (0, 2, 20, '55455')]")
        self.assertEqual(
            str(data.feature[0][-5:]),
            "[(1, 2, 15, '32603') (1, 4, 1, '76006') (1, 6, 1, '14706')\n"
            " (1, 4, 0, '01060') (0, 2, 6, '11106')]")
        self.assertEqual(len(data.feature[0]), 6040)
        self.assertEqual(
            str(data.feature[1][:5]),
            "[ (u'Toy Story (1995)', 1995, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Jumanji (1995)', 1995, [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Grumpier Old Men (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])\n"
            " (u'Waiting to Exhale (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Father of the Bride Part II (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]")
        self.assertEqual(
            str(data.feature[1][-5:]),
            "[ (u'Meet the Parents (2000)', 2000, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Requiem for a Dream (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Tigerland (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Two Family House (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
            " (u'Contender, The (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]")
        self.assertEqual(len(data.feature[1]), 3706)

    def test_MOVIELENS1M_INFO(self):
        from pyrecsys.datasets import MOVIELENS1M_INFO
        self.assertEqual(
            str(MOVIELENS1M_INFO['user_age']),
            "['Under 18' '18-24' '25-34' '35-44' '45-49' '50-55' '56+']")
        self.assertEqual(
            str(MOVIELENS1M_INFO['user_occupation']),
            "['Other or Not Specified' 'Academic/Educator' 'Artist' 'Clerical/Admin'\n"
            " 'College/Grad Student' 'Customer Service' 'Doctor/Health Care'\n"
            " 'Executive/Managerial' 'Farmer' 'Homemaker' 'K-12 Student' 'Lawyer'\n"
            " 'Programmer' 'Retired' 'Sales/Marketing' 'Scientist' 'Self-Employed'\n"
            " 'Technician/Engineer' 'Tradesman/Craftsman' 'Unemployed' 'Writer']")
        self.assertEqual(
            str(MOVIELENS1M_INFO['item_genre']),
            "['Action' 'Adventure' 'Animation' \"Children's\" 'Comedy' 'Crime'\n"
            " 'Documentary' 'Drama' 'Fantasy' 'Film-Noir' 'Horror' 'Musical' 'Mystery'\n"
            " 'Romance' 'Sci-Fi' 'Thriller' 'War' 'Western']")

##### Main routine #####
if __name__ == '__main__':
    unittest.main()



