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

from ..recommender import BaseEventRecommender
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


class BaseEventItemFinder(with_metaclass(ABCMeta, BaseEventRecommender)):
    """
    Recommenders to find good items from event data
    """

    def __init__(self, random_state=None):
        super(BaseEventItemFinder, self).__init__(
            random_state=random_state)

    def fit(self, random_state=None):
        """
        fitting model
        """
        super(BaseEventItemFinder, self).fit(random_state=random_state)

    def _get_event_array(self, data, event_index=(0, 1), sparse_type='csr'):
        """
        Set statistics of input dataset, and generate a matrix representing
        implicit feedbacks.

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventData`
            data to fit
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model
        sparse_type: str
            type of sparse matrix: 'csr', 'csc', 'lil', or 'array'
            default='csr'

        Returns
        -------
        event_array: array, shape=(n_users, n_items), dtype=int
            return rating matrix that takes 1 if it is consumed, 0 otherwise.
            if event data are not available, return None
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events
        """

        # validity of arguments
        if sparse_type not in ['csr', 'csc', 'lil', 'array']:
            raise TypeError("illegal type of sparse matrices")

        if not isinstance(data, EventData):
            raise TypeError("input data must data.EventData class")

        # import meta information of objects and events to this recommender
        self._set_object_info(data)
        self._set_event_info(data)
        event_index = np.asarray(event_index)

        # get number of objects
        n_objects = self.n_objects[self.event_otypes[event_index]]

        # get event data
        users = data.event[:, event_index[0]]
        items = data.event[:, event_index[1]]
        scores = np.ones_like(users, dtype=int)

        # generate array
        event = sparse.coo_matrix((scores, (users, items)), shape=n_objects)
        if sparse_type == 'csc':
            event = event.tocsc()
        elif sparse_type == 'csr':
            event = event.tocsr()
        elif sparse_type == 'lil':
            event = event.tolil()
        else:
            event = event.toarray()

        return event, n_objects

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
