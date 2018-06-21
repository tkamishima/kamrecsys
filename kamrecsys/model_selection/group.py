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
from sklearn.utils import indexable, check_random_state
from sklearn.model_selection import (
    BaseCrossValidator, PredefinedSplit, KFold)

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


# =============================================================================
# Classes
# =============================================================================


class KFoldWithinGroups(BaseCrossValidator):
    """
        Generate per Groups K-fold split



class KFoldWithinGroups(BaseCrossValidator):
    """
    Generate K-fold splits within each group

    Data are first divided into groups specified by `groups` . Then, each group
    is further divided into K-folds.  The elements having the same fold number
    are assigned to the same fold.

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
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):

        super(KFoldWithinGroups, self).__init__()

        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        # check arguments
        X, y, groups = indexable(X, y, groups)

        # get the number of samples
        n_samples = X.shape[0]
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}."
                 ).format(self.n_splits, n_samples))

        for train, test in super(KFoldWithinGroups, self).split(X, y, groups):
            yield train, test

    def _iter_test_masks(self, X, y=None, groups=None):
        # yields mask array for test splits
        n_samples = X.shape[0]

        # if groups is not specified, an entire data is specified as one group
        if groups is None:
            groups = np.zeros(n_samples, dtype=int)

        # constants
        indices = np.arange(n_samples)
        test_fold = np.empty(n_samples, dtype=bool)
        rng = check_random_state(self.random_state)
        group_indices = np.unique(groups)
        iters = np.empty(group_indices.shape[0], dtype=object)

        # generate iterators
        cv = KFold(self.n_splits, self.shuffle, rng)
        for i, g in enumerate(group_indices):
            group_member = indices[groups == g]
            iters[i] = cv.split(group_member)

        # generate training and test splits
        for fold in xrange(self.n_splits):
            test_fold[:] = False
            for i, g in enumerate(group_indices):
                group_train_i, group_test_i = iters[i].next()
                test_fold[indices[groups == g][group_test_i]] = True
            yield test_fold

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class InterlacedKFold(BaseCrossValidator):
    """
    k-folds by a interlaced grouping.

    The i-th data is assigned to the (i mod n_splits)-th group.

    Subsequent data are are grouped into the same fold in a case of a standard
    k-fold cross validation, but this is inconvenient if subsequent data are
    highly correlated.  This class is useful in such a situation.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. It must be `n_splits >= 2` .
    """

    def __init__(self, n_splits=3):

        super(InterlacedKFold, self).__init__()

        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = X.shape[0]
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}."
                 ).format(self.n_splits, n_samples))

        # generate test fold
        test_fold = np.arange(n_samples, dtype=int) % self.n_splits
        cv = PredefinedSplit(test_fold)

        return(cv.split())

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


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
