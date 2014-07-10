#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics for General Purpose
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
from abc import ABCMeta, abstractmethod
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BaseMetrics']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Classes
#==============================================================================


class BaseMetrics(object):
    """
    Base class for metrics

    Parameters
    ----------
    name : str, default="metrics", optional
        name of metrics

    Attributes
    ----------
    name : str
        name of metric
    metrics : dict
        dictionary of metrics
    """

    __metaclass__ = ABCMeta

    def __init__(self, name='metrics'):
        self.name = name
        self.metrics = {}

    def subnames(self):
        """
        list of metric sub names

        Returns
        -------
        subnames : list, dtype=str
            list of metrics sub-names
        """
        return sorted(self.metrics)

    def fullnames(self):
        """
        list of metric sub names

        Returns
        -------
        fullnames : list, dtype=str
            list of metrics full-names
        """
        return [ self.name + '_' + subname for subname in self.subnames()]

    def values(self):
        """
        list of metric values

        Returns
        -------
        values : list, dtype=float or int
            list of metric values
        """
        return [self.metrics[subname] for subname in self.subnames()]

#==============================================================================
# Module initialization
#==============================================================================

# init logging system
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

# Check if this is call as command script

if __name__ == '__main__':
    _test()
