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
from sklearn.model_selection import KFold

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


def generate_pergroup_kfold(
        n_samples, groups=None, n_splits=3, shuffle=False, random_state=None):
    """
    Generate per Groups K-fold split

    Data are first divided into groups specified by `groups` . Then, each group
    is further divided into K-folds.  The elements having the same fold number
    are assigned to the same fold.
    This is used with :class:`sklearn.model_selection.PredefinedSplit` .

    Parameters
    ----------
    n_samples : int
        Total number of elements.
    groups : array, dtype=int, shape=(n,)
        the specification of group. If `None` , an entire data is treated as
        one group.
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    test_fold : array, shape=(n,)
        a sequence of indicator-numbers representing the group assignment
    """

    # error handling
    if n_splits < 2:
        raise ValueError('n_splits must be larger or equal than 2.')

    if groups is None:
        groups = np.zeros(n_samples, dtype=int)
    else:
        groups = np.asanyarray(groups, dtype=int)
        if n_samples != groups.shape[0]:
            raise ValueError(
                'Inconsistent size of groups and total number of elements.')

    # generate test_fold
    cv = KFold(n_splits, shuffle, random_state)
    test_fold = np.empty(n_samples, dtype=int)
    test_fold[:] = 5
    for g in np.unique(groups):
        fold = 0
        g_index = np.arange(n_samples, dtype=int)[groups == g]

        if g_index.shape[0] < n_splits:
            raise ValueError(
                'the size of each group must be larger than n_splits')

        for train_i, test_i in cv.split(g_index):
            test_fold[g_index[test_i]] = fold
            fold += 1

    return test_fold


def generate_interlace_kfold(n_data, n_splits=3):
    """
    Generate interlace group.
    
    The i-th data is assigned to the (i mod n_splits)-th group.
    This is used with :class:`sklearn.model_selection.PredefinedSplit` .
    In a case of a standard k-fold cross validation, subsequent data are tend
    to be grouped into the same fold.  However, this is inconvenient, if
    subsequent data are highly correlated.
    
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. It must be `n_splits >= 2` .
    n_data : int
        the number of data. It must be n_data `n_data > n_splits` .

    Returns
    -------
    group : array, shape=(n_data,)
        a sequence of indicator-numbers indicating the group assignment
    """
    n_splits = int(n_splits)
    if n_splits < 2:
        ValueError('n_splits must be larger or equal than 2.')

    n_data = int(n_data)
    if n_data < n_splits:
        ValueError('n_data must be larger than n_splits.')

    return np.arange(n_data, dtype=int) % n_splits


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
