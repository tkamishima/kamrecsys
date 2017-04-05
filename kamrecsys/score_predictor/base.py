#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for Score Predictors
"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# =============================================================================
# Imports
# =============================================================================
import logging
from abc import ABCMeta

import numpy as np
from six import with_metaclass

from ..recommender import BaseEventRecommender
from ..data import EventWithScoreData

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


class BaseEventScorePredictor(with_metaclass(ABCMeta, BaseEventRecommender)):
    """
    Recommenders to predict preference scores from event data
    """

    def __init__(self, random_state=None):
        super(BaseEventScorePredictor, self).__init__(
            random_state=random_state)

    def fit(self, random_state=None):
        """
        fitting model
        """
        super(BaseEventScorePredictor, self).fit(random_state=random_state)

    def _get_event_and_score(self, data, event_index, score_index):
        """
        Parameters
        ----------
        data : :class:`kamrecsys.data.EventWithScoreData`
            data to fit
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used in
            a recommendation model
        score_index : int
            Ignored if score of data is a single criterion type. In a multi-
            criteria case, specify the position of the target score in a score
            vector. (default=0)

        Returns
        -------
        event : array_like, shape=(n_events, event_index.shape[0])
            an extracted set of events
        score : array_like, shape=(n_events,)
            scores for each event
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.EventWithScoreData`
            class
        """
        if not isinstance(data, EventWithScoreData):
            raise TypeError("input data must data.EventWithScoreData class")

        # import meta information of objects and events to this recommender
        self._set_object_info(data)
        self._set_event_info(data)
        event_index = np.asarray(event_index)

        # get event data
        event = np.atleast_2d(data.event)[:, event_index]

        # get score information
        if data.n_stypes == 1:
            score = data.score
        else:
            score = data.score[:, score_index]

        # get number of objects
        n_objects = self.n_objects[self.event_otypes[event_index]]

        return event, score, n_objects

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
