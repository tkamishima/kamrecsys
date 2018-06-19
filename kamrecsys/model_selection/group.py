#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group Generator for cross validations 

Generated groups will be used with functions using groups, such as
:class:`sklearn.model_selection.LeaveOneGroupOut` .
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging

import numpy as np

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def generate_interlace_kfold(n, n_splits=3):
    """
    Generate k-folds by a interlace grouping.
    
    The i-th data is assigned to the (i mod n_splits)-th group.
    This is used with :class:`sklearn.model_selection.PredefinedSplit` .
    In a case of a standard k-fold cross validation, subsequent data are tend
    to be grouped into the same fold.  However, this is inconvenient, if
    subsequent data are highly correlated.
    
    Parameters
    ----------
    n : int
        the number of data. It must be n `n > n_splits` .
    n_splits : int, default=3
        Number of folds. It must be `n_splits >= 2` .

    Returns
    -------
    test_fold : array, shape=(n,)
        a sequence of indicator-numbers representing the group assignment
    """
    n_splits = int(n_splits)
    if n_splits < 2:
        ValueError('n_splits must be larger or equal than 2.')

    n = int(n)
    if n < n_splits:
        ValueError('n must be larger than n_splits.')

    return np.arange(n, dtype=int) % n_splits


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script

if __name__ == '__main__':
    _test()
