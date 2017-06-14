#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Class for Item Finders
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
from abc import ABCMeta

import numpy as np
from scipy import sparse as sparse
from six import with_metaclass

from ..base import BaseEventRecommender
from ..data import EventData

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


class BaseImplicitItemFinder(with_metaclass(ABCMeta, BaseEventRecommender)):
    """
    Recommenders to find good items from event data
    """

    def __init__(self, random_state=None):
        super(BaseImplicitItemFinder, self).__init__(random_state=random_state)

    def fit(self, data, event_index=(0, 1), random_state=None):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model
        random_state: RandomState or an int seed (None by default)
            A random number generator instance
        """
        super(BaseImplicitItemFinder, self).fit(data, event_index, random_state)

    def get_event_array(self, sparse_type='csr'):
        """
        Set statistics of input dataset, and generate a matrix representing
        implicit feedbacks.

        Parameters
        ----------
        sparse_type: str
            type of sparse matrix: 'csr', 'csc', 'lil', or 'array'
            default='csr'

        Returns
        -------
        ev : array, shape=(n_users, n_items), dtype=int
            return rating matrix that takes 1 if it is consumed, 0 otherwise.
            if event data are not available, return None
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events
        """

        # validity of arguments
        if sparse_type not in ['csr', 'csc', 'lil', 'array']:
            raise TypeError("illegal type of sparse matrices")

        # get number of objects
        n_objects = self.n_objects[self.event_otypes[self.event_index]]

        # get event data
        users = self.event[:, self.event_index[0]]
        items = self.event[:, self.event_index[1]]
        scores = np.ones_like(users, dtype=int)

        # generate array
        ev = sparse.coo_matrix((scores, (users, items)), shape=n_objects)
        if sparse_type == 'csc':
            ev = ev.tocsc()
        elif sparse_type == 'csr':
            ev = ev.tocsr()
        elif sparse_type == 'lil':
            ev = ev.tolil()
        else:
            ev = ev.toarray()

        return ev, n_objects

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
