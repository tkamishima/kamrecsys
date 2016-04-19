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
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseData(unittest.TestCase):

    def test_class(self):
        from kamrecsys.datasets import load_pci_sample

        data = load_pci_sample()

        self.assertEqual(data.to_iid(0, 'Mick LaSalle'), 5)
        with self.assertRaises(ValueError):
            data.to_iid(0, 'Dr. X')
        self.assertEqual(data.to_eid(1, 4), 'The Night Listener')
        with self.assertRaises(ValueError):
            data.to_eid(1, 100)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    unittest.main()
