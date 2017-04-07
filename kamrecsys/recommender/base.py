#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Recommender Classes
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
from abc import ABCMeta, abstractmethod

import numpy as np
from six import with_metaclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array

from ..data import BaseData, EventData, EventUtilMixin

# =============================================================================
# Module metadata variables
# =============================================================================

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


class BaseRecommender(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Abstract class for all recommenders

    Attributes
    ----------
    n_otypes : int
        the number of object types, succeed from training data sets
    n_objects : array_like, shape=(n_otypes), dtype=int
        the number of different objects in each type, succeed from training
        data sets
    eid : array_like, shape=(n_otypes,), dtype=(array_like)
        conversion table to external ids, succeed from training data sets
    iid : dictionary
        conversion table to internal ids, succeed from training data sets
    fit_results_ : dict
        Side information about results of fitting
    random_state : RandomState or an int seed (None by default)
        A random number generator instance

    Raises
    ------
    ValueError
        if n_otypes < 1
    """

    def __init__(self, random_state=None):
        self.n_otypes = 0
        self.n_objects = None
        self.eid = None
        self.iid = None
        self.random_state = random_state
        self._rng = None
        self.fit_results_ = {}

    def _set_object_info(self, data):
        """
        import object meta information of input data to recommenders

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.BaseData` class
        """
        if not isinstance(data, BaseData):
            raise TypeError("input data must data.BaseData class")

        self.n_otypes = data.n_otypes
        self.n_objects = data.n_objects
        self.eid = data.eid
        self.iid = data.iid

    def fit(self, data, random_state=None):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        random_state: RandomState or an int seed (None by default)
            A random number generator instance
        """

        # set random state
        if random_state is None:
            random_state = self.random_state
        self._rng = check_random_state(random_state)

        # set object information in data
        self._set_object_info(data)

    @abstractmethod
    def predict(self, eev, **kwargs):
        """
        abstract method: predict score of given event represented by external
        ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """
        pass

    def to_eid(self, otype, iid):
        """
        convert an internal id to the corresponding external id
        (Copied from data.BaseData)

        Parameters
        ----------
        otype : int
            object type
        iid : int
            an internal id

        Returns
        -------
        eid : int
            the corresponding external id
        
        Raises
        ------
        ValueError
            an internal id is out of range
        """
        try:
            return self.eid[otype][iid]
        except IndexError:
            raise ValueError("Illegal internal id")

    def to_iid(self, otype, eid):
        """
        convert an external id to the corresponding internal id.
        (Copied from data.BaseData)

        Parameters
        ----------
        otype : int
            object type
        eid : int
            an external id
        
        Returns
        -------
        iid : int
            the corresponding internal id
        
        Raises
        ------
        ValueError
            an external id is out of range
        """
        try:
            return self.iid[otype][eid]
        except KeyError:
            raise ValueError("Illegal external id")


class BaseEventRecommender(
        with_metaclass(ABCMeta, BaseRecommender, EventUtilMixin)):
    """
    Recommenders using a data.EventData class or its subclasses
    
    Attributes
    ----------
    s_event : int
        the size of event, which is the number of object types to represent a
        rating event
    event_otypes : array_like, shape=(s_event,)
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    event : array_like, shape=(n_events, s_event), dtype=int
        each row is a vector of internal ids that indicates the target of
        rating event
    event_feature : array_like, shape=(n_events, variable), dtype=variable
        i-the row contains the feature assigned to the i-th event
    event_index : array_like, shape=(s_event,)
            a set of indexes to specify the elements in events that are used in
            a recommendation model
    """

    def __init__(self, random_state=None):
        super(BaseEventRecommender, self).__init__(random_state=random_state)

        self._empty_event_info()
        self.event_index = None

    def _empty_event_info(self):
        """
        Set empty Event Information
        """
        self.n_events = 0
        self.event = None
        self.event_feature = None

    def _set_event_info(self, data):
        """
        import event meta information of input data to recommenders

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventData`
            input data

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.EventData` class
        """
        if not isinstance(data, EventData):
            raise TypeError("input data must data.EventData class")

        self.s_event = data.s_event
        self.event_otypes = data.event_otypes
        self.n_events = data.n_events
        self.event = data.event
        self.event_feature = data.event_feature

    def get_event(self):
        """
        Returns numbers of objects and an event array
    
        Returns
        -------
        ev : array_like, shape=(n_events, event_index.shape[0])
            an extracted set of events
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events
        """

        # get event data
        ev = np.atleast_2d(self.event)[:, self.event_index]

        # get number of objects
        n_objects = self.n_objects[self.event_otypes[self.event_index]]

        return ev, n_objects

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        self._empty_event_info()
        self.event_index = None

    def fit(self, data, event_index=None, random_state=None):
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
        super(BaseEventRecommender, self).fit(data, random_state)

        # set object information in data
        self._set_event_info(data)
        if event_index is None:
            self.event_index = np.arange(self.s_event, dtype=int)
        else:
            self.event_index = np.asanyarray(event_index, dtype=int)

    @abstractmethod
    def raw_predict(self, ev, **kwargs):
        """
        abstract method: predict score of given one event represented by
        internal ids

        Parameters
        ----------
        ev : array_like, shape=(n_events, s_event)
            events represented by internal id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

    def predict(self, eev):
        """
        predict score of given event represented by external ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

        eev = check_array(np.atleast_2d(eev), dtype=int)
        if eev.shape[1] != self.s_event:
            raise TypeError("unmatched sized of events")

        return np.squeeze(self.raw_predict(self.to_iid_event(eev)))


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
