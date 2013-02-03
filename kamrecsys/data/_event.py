#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data model: rating events
"""

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np
from abc import ABCMeta

from ._base import BaseData

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['EventUtilMixin', 'EventData', 'EventWithScoreData']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class EventUtilMixin(object):
    """
    Methods that are commonly used in data containers and recommenders for
    handling events..
    """

    __multiclass__ = ABCMeta

    def to_eid_event(self, data):
        """
        convert an event vector or array represented by internal ids to those
        by external ids.

        Parameters
        ----------
        data : array_like
            array whose elements are represented by external ids
        
        Returns
        -------
        new_data : array_like
            array whose elements are represented by external ids
        """
        if data.ndim == 1 and data.shape[0] == self.s_event:
            new_data = np.array([self.eid[self.event_otypes[e]][data[e]]
                                 for e in xrange(self.s_event)],
                                                               dtype=self.eid[
                                                                     0].dtype)
        elif data.ndim == 2 and data.shape[1] == self.s_event:
            new_data = np.empty_like(data, dtype=self.eid[0].dtype)
            for e in xrange(self.s_event):
                new_data[:, e] = self.eid[self.event_otypes[e]][data[:, e]]
        else:
            raise TypeError("Shape of input is illegal")

        return new_data

    def to_iid_event(self, ev, missing_values=None):
        """
        convert an event vector or array represented by external ids to those
        by internal ids.

        Parameters
        ----------
        ev : array_like
            array whose elements are represented by external ids
        missing_values : optional, int or array_like, shape=(s_event,)
            if unknown external ids are detected, these will be converted to -1.
            as default, numbers of possible objects are used.

        Returns
        -------
        new_ev : array_like
            array whose elements are represented by external ids

        Raises
        ------
        TypeError
            Shape of an input array is illegal
        """
        if missing_values is None:
            missing_values = self.n_objects[self.event_otypes]
        if ev.ndim == 1 and ev.shape[0] == self.s_event:
            new_ev = np.array([self.iid[self.event_otypes[e]].\
                               get(ev[e], missing_values[e])
                               for e in xrange(self.s_event)], dtype=np.int)
        elif ev.ndim == 2 and ev.shape[1] == self.s_event:
            new_ev = np.empty_like(ev, dtype=np.int)
            for e in xrange(self.s_event):
                iid = self.iid[self.event_otypes[e]]
                new_ev[:, e] = [iid.get(i, missing_values[e])
                                for i in ev[:, e]]
        else:
            raise TypeError('The shape of an input is illegal')

        return new_ev


class EventData(BaseData, EventUtilMixin):
    """ Container of rating events and their associated features.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.

    Attributes
    ----------
    s_event : int
        the size of event, which is the number of objects to reprent a rating
        event
    event : array_like, shape=(, s_event), dtype=int
        each row is a vector of internal ids that indicates the tareget of
        rating event
    event_feature : array_like, shape=(n_events, variable), dtype=variable
        i-the row contains the feature assigned to the i-th event

    Raises
    ------
    ValueError
        if n_otypes < 1 or event_otypes is illegal.

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2, event_otypes=None):
        super(EventData, self).__init__(n_otypes=n_otypes)
        if event_otypes is None:
            self.s_event = n_otypes
            self.event_otypes = np.arange(self.s_event, dtype=int)
        else:
            if event_otypes.ndim != 1 or\
               np.min(event_otypes) < 0 or\
               np.max(event_otypes) >= n_otypes:
                raise ValueError("Illegal event_otypes specification")
            self.s_event = event_otypes.shape[0]
            self.event_otypes = np.asarray(event_otypes)
        self.n_events = 0
        self.event = None
        self.event_feature = None

    def set_events(self, event, event_feature=None):
        """Set event data from structured array.

        Parameters
        ----------
        event : array_like, shape=(n_events, s_event) 
            each row corresponds to an event represented by a vector of object
            with external ids
        event_feature : optional, array_like, shape=(n_events, variable)
            feature of events
        """
        for otype in xrange(self.n_otypes):
            self.n_objects[otype], self.eid[otype], self.iid[otype] =\
            self._gen_id(event[:, self.event_otypes == otype])

        self.event = np.empty_like(event, dtype=np.int)
        for e in xrange(self.s_event):
            iid = self.iid[self.event_otypes[e]]
            self.event[:, e] = [iid[i] for i in event[:, e]]

        self.n_events = self.event.shape[0]
        if event_feature is not None:
            self.event_feature = np.asarray(event_feature).copy()
        else:
            self.event_feature = None


class EventWithScoreData(EventData):
    """ Container of rating events, rating scores, and features.

    Rating scores are assigned at each rating event.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional 
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    n_stypes : optional, int
        see attribute n_stypes (default=1)

    Attributes
    ----------
    score_domain : tuple or 1d-ndarray of tuple
        i-th tuple is a pair of the minimum and the maximum score of i-th score
    score : array_like, shape=(n_events) or (n_events, n_stypes)
        rating scores of each events. this array takes a vector shape if
        `n_rtypes` is 1; otherwise takes

    Raises
    ------
    ValueError
        if n_otypes < 1 or n_stypes < 1 or event_otypes is illegal.

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2, n_stypes=1, event_otypes=None):
        if n_stypes < 1:
            raise ValueError("n_styeps must be >= 1")
        super(EventWithScoreData, self).__init__(n_otypes=n_otypes,
                                                 event_otypes=event_otypes)
        self.n_stypes = n_stypes
        self.score_domain = None
        self.score = None

    def set_events(self, event, score, score_domain=None, event_feature=None):
        """Set event data from structured array.

        Parameters
        ----------
        event : array_like, shape=(n_events, s_event) 
            each row corresponds to an event represented by a vector of object
            with external ids
        score : array_like, shape=(n_events) or (n_stypes, 
            i-th ele
        score_domain : optional, tuple or 1d-ndarray of tuple
            min and max of scores. as 
        event_feature : optional, array_like, shape=(n_events, variable)
            feature of events
        """

        super(EventWithScoreData, self).set_events(event, event_feature)

        self.score = np.asanyarray(score)
        self.score_domain = np.asanyarray(score_domain)

#==============================================================================
# Functions 
#==============================================================================

#==============================================================================
# Module initialization 
#==============================================================================

# init logging system ---------------------------------------------------------

logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

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
