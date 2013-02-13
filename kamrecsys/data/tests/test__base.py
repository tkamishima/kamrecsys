#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

##### Test Classes #####

class TestBaseData(TestCase):

    def test_class(self):
        from kamrecsys.datasets import load_pci_sample

        data = load_pci_sample()

        self.assertEqual(data.to_iid(0, 'Mick LaSalle'), 5)
        with self.assertRaises(ValueError):
            x = data.to_iid(0, 'Dr. X')
        self.assertEqual(data.to_eid(1, 4), 'The Night Listener')
        with self.assertRaises(ValueError):
            x = data.to_eid(1, 100)

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
