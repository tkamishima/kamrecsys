#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Container: abstract classes
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
from six import with_metaclass
import numpy as np

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


class BaseData(with_metaclass(ABCMeta, object)):
    """
    Abstract class for data container

    Instances of this class contain only information about objects.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)

    Attributes
    ----------
    n_objects : array_like, shape=(n_otypes), dtype=int
        the number of different objects in each type.  the first and the
        second types of objects are referred by the keywords, ``user`` or
        ``item``.
    eid : array_like, shape=(n_otypes,), dtype=(array_like)
        id[i] is a vector of external ids. the j-th element of the array is the
        external id that corresponds to the object with internal id j.
    iid : dictionary
        id[i] is a dictionary for internal ids whose object type is i. the
        value for the key 'j' contains the internal id of the object whose
        external id is j.
    feature : array_like, shape=(n_otypes), dtype=array_like
        i-the element contains the array of features for i-th object types,
        whose shape is (n_object[i], variable). j-th row of the i-th array
        contains a feature for the object whose internal id is j.

    Raises
    ------
    ValueError
        if n_otypes < 1

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2):
        if n_otypes < 1:
            raise ValueError("n_otypes must be >= 1")
        self.n_otypes = n_otypes
        self.n_objects = np.zeros(self.n_otypes, dtype=int)
        self.eid = np.empty(self.n_otypes, dtype=np.object)
        self.iid = np.empty(self.n_otypes, dtype=np.object)
        self.feature = np.empty(self.n_otypes, dtype=np.object)

    def set_features(self, otype, eid, feature):
        """
        Set object feature
        
        Parameters
        ----------
        otype : int
            target object type
        eid : array_like, shape=(n_objects,)
            external ids of the corresponding object features
        feature : array_like
            array of object feature
        """
        iid = self.iid[otype]
        index = np.repeat(len(eid), len(iid))
        for i, j in enumerate(eid):
            if j in iid:
                index[iid[j]] = i
        self.feature[otype] = feature[index].copy()

    def to_eid(self, otype, iid):
        """
        convert an internal id to the corresponding external id

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
        convert an external id to the corresponding internal id

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

    @staticmethod
    def _gen_id(event):
        """
        Generate a conversion map between internal and external ids
        
        Parameter
        ---------
        event : array_like
            array contains all objects of the specific type

        Returns
        -------
        n_objects : int
            the number of unique objects
        eid : array, shape=(variable,)
            map from internal id to external id
        iid : dict
            map from external id to internal id
        """
        eid = np.sort(np.unique(event))
        iid = {}
        for i in xrange(len(eid)):
            iid[eid[i]] = i
        return len(eid), eid, iid

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
