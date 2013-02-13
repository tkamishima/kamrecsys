#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from unittest import TestCase

##### Test Classes #####

class TestLoadPCISample(TestCase):

    def test_load_pci_sample(self):
        from kamrecsys.datasets import load_pci_sample
        data = load_pci_sample()
        self.assertEqual(str(data.event_otypes), '[0 1]')
        self.assertEqual(data.n_otypes, 2)
        self.assertEqual(data.n_events, 35)
        self.assertEqual(str(data.feature), '[None None]')
        self.assertEqual(
            str(data.event),
            "[[2 1]\n [2 2]\n [2 5]\n [2 3]\n [2 4]\n [5 1]\n [5 2]\n"
            " [5 0]\n [5 3]\n [5 5]\n [5 4]\n [0 2]\n [0 0]\n [0 5]\n"
            " [0 3]\n [0 4]\n [3 1]\n [3 2]\n [3 0]\n [3 3]\n [3 4]\n"
            " [3 5]\n [6 2]\n [6 3]\n [6 5]\n [1 1]\n [1 2]\n [1 0]\n"
            " [1 3]\n [1 5]\n [1 4]\n [4 1]\n [4 2]\n [4 3]\n [4 4]]")
        self.assertDictEqual(data.iid[0],
                {'Jack Matthews': 2, 'Mick LaSalle': 5, 'Claudia Puig': 0,
                 'Lisa Rose': 3, 'Toby': 6, 'Gene Seymour': 1,
                 'Michael Phillips': 4})
        self.assertDictEqual(data.iid[1],
                {'Lady in the Water': 1, 'Just My Luck': 0,
                 'Superman Returns': 3, 'You, Me and Dupree': 5,
                 'Snakes on a Planet': 2, 'The Night Listener': 4})
        self.assertIsNone(data.event_feature)
        self.assertEqual(
            str(data.score),
            "[ 3.   4.   3.5  5.   3.   3.   4.   2.   3.   2.   3.   3.5"
            "  3.   2.5  4.\n  4.5  2.5  3.5  3.   3.5  3.   2.5  4.5  4."
            "   1.   3.   3.5  1.5  5.   3.5\n  3.   2.5  3.   3.5  4. ]")
        self.assertEqual(
            str(data.eid[0]),
            "['Claudia Puig' 'Gene Seymour' 'Jack Matthews' 'Lisa Rose'\n"
            " 'Michael Phillips' 'Mick LaSalle' 'Toby']")
        self.assertEqual(
            str(data.eid[1]),
            "['Just My Luck' 'Lady in the Water' 'Snakes on a Planet'"
            " 'Superman Returns'\n 'The Night Listener' 'You, "
            "Me and Dupree']")
        self.assertEqual(str(data.n_objects), '[7 6]')
        self.assertEqual(data.n_stypes, 1)
        self.assertEqual(data.s_event, 2)
        self.assertEqual(str(data.score_domain), '[ 1.  5.]')

##### Do Test #####
if __name__ == '__main__':
    unittest.main()
