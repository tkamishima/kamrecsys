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

from ..base import BaseEventRecommender
from ..data import EventWithScoreData, ScoreUtilMixin

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


class BaseScorePredictor(
    with_metaclass(ABCMeta, BaseEventRecommender, ScoreUtilMixin)):
    """
    Recommenders to predict preference scores from event data

    Attributes
    ----------
    score_domain : tuple or 1d-array of tuple
        i-th tuple is a triple of the minimum, the maximum, and strides of the
        i-th score
    score : array_like, shape=(n_events) or (n_events, n_stypes)
        rating scores of each events. this array takes a vector shape if
        `n_rtypes` is 1; otherwise takes
    n_stypes : int
        number of score types
    n_score_levels : int or array, dtype=int, shape=(,n_stypes)
        the number of score levels
    score_index : int
        Ignored if score of data is a single criterion type. In a multi-
        criteria case, specify the position of the target score in a score
        vector.
    """

    def __init__(self, random_state=None):
        super(BaseScorePredictor, self).__init__(
            random_state=random_state)

        # set empty score information
        self._empty_score_info()
        self.score_index = 0

    def get_score(self):
        """
        return score information

        Returns
        -------
        sc : array_like, shape=(n_events,)
            scores for each event
        """

        # get score information
        if self.n_stypes == 1:
            sc = self.score
        else:
            sc = self.score[:, self.score_index]

        return sc

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        super(BaseScorePredictor, self).remove_data()
        self._empty_score_info()

    def fit(self, data, event_index=(0, 1), score_index=0, random_state=None):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model
        score_index : int
            Ignored if score of data is a single criterion type. In a multi-
            criteria case, specify the position of the target score in a score
            vector. (default=0)
        random_state: RandomState or an int seed (None by default)
            A random number generator instance
        """
        super(BaseScorePredictor, self).fit(
            data, event_index, random_state)

        # set object information in data
        self._set_score_info(data)
        self.score_index = score_index

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
