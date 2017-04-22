#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data model: rating events
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
from abc import ABCMeta
from six import with_metaclass

from . import EventData
from metrics import generate_score_bins


# =============================================================================
# Public symbols
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

class ScoreUtilMixin(with_metaclass(ABCMeta, object)):
    """
    Methods that are commonly used in data containers and recommenders for
    handling scores.
    """

    def _set_score_info(self, data):
        """

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.EventWithScoreData`
            class
        """
        if not isinstance(data, EventWithScoreData):
            raise TypeError("input data must data.EventWithScoreData class")

        self.score_domain = data.score_domain
        self.score = data.score
        self.n_score_levels = data.n_score_levels

    def generate_score_bins(self):
        """
        Generate histogram bins for scores from score_domain
                
        Returns
        -------
        score_bins : array, shape=(n_score_levels,), dtype=float
            bins of histogram. boundaries of bins are placed at the center of
            adjacent scores.
        """

        return generate_score_bins(self.score_domain)

    def get_score_levels(self):
        """
        get a set of possible score levels

        Returns
        -------
        score_levels : array, shape=(n_score_levels,)
            a set of possible score levels
        """
        return np.linspace(
            self.score_domain[0], self.score_domain[1], self.n_score_levels)


class EventWithScoreData(EventData, ScoreUtilMixin):
    """ Container of rating events, rating scores, and features.

    Rating scores are assigned at each rating event.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.

    Attributes
    ----------
    score_domain : tuple or 1d-array of tuple
        i-th tuple is a triple of the minimum, the maximum, and strides of the
        i-th score
    score : array_like, shape=(n_events,)
        rating scores of each events.
    n_score_levels : int
        the number of score levels

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2, event_otypes=None):
        super(EventWithScoreData, self).__init__(n_otypes=n_otypes,
                                                 event_otypes=event_otypes)
        self.score_domain = None
        self.score = None
        self.n_score_levels = None

    def set_event(self, event, score, score_domain=None, event_feature=None):
        """
        Set event data from structured array.

        Parameters
        ----------
        event : array_like, shape=(n_events, s_event)
            each row corresponds to an event represented by a vector of object
            with external ids
        score : array_like, shape=(n_events,)
            a set of rating scores
        score_domain : optional, tuple or 1d-array of tuple
            min and max of scores, and the interval between scores
        event_feature : optional, array_like, shape=(n_events, variable)
            feature of events
        """

        super(EventWithScoreData, self).set_event(event, event_feature)

        self.score = np.asarray(score)
        self.score_domain = np.asanyarray(score_domain)
        self.n_score_levels = (
            int((score_domain[1] - score_domain[0]) / score_domain[2]) + 1)

    def digitize_score(self, score=None):
        """
        Returns discretized scores that starts with 0

        Parameters
        ----------
        score : array, optional
            if specified, the scores in this array is digitized; otherwise
            `self.score` is converted.

        Returns
        -------
        digitized_scores : array, dtype=int, shape=(n_events,)
        """

        bins = self.generate_score_bins()

        digitized_scores = score
        if digitized_scores is None:
            digitized_scores = np.digitize(self.score, bins) - 1
        else:
            digitized_scores = np.digitize(score, bins) - 1

        return digitized_scores

    def filter_event(self, filter_cond):
        """
        replace event data with those consisting of events whose corresponding
        `filter_cond` is `True`.   

        Parameters
        ----------
        filter_cond : array, dtype=bool, shape=(n_events,)
            Boolean array that specifies whether each event should be included
            in a new event array.
        """

        # check whether event info is available
        if self.event is None:
            return

        # filter out event related information
        super(EventWithScoreData, self).filter_event(filter_cond)

        # filter out event data
        if self.score is not None:
            self.score = self.score[filter_cond]


# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system ---------------------------------------------------------
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

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
