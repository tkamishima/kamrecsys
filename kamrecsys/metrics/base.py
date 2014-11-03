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
from sklearn.utils import (
    assert_all_finite,
    safe_asarray)

#==============================================================================
# Public symbols
#==============================================================================

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
        self.params = {}

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


class DescriptiveStatistics(BaseMetrics):
    """
    descriptive statistics

    * nos_samples : the number of samples
    * mean : mean
    * stdev : standard deviation

    Parameters
    ----------
    x : array, shpae=(n_samples), dtype=float or int
        data

    """

    def __init__(self, x, name="descriptive_statistics"):
        super(DescriptiveStatistics, self).__init__(name=name)

        # check inputs
        x = safe_asarray(x)
        assert_all_finite(x)

        # the number of samples
        self.metrics['nos_samples'] = x.shape[0]

        # mean and standard deviation of true values
        self.metrics['mean'] = np.mean(x)
        self.metrics['stdev'] = np.std(x)


class Histogram(BaseMetrics):
    """
    Histogram

    * bin : boundaries of bins
    * count : the numbers of samples in each bin
    * density : densities of samples in each bin

    Parameters
    ----------
    x : array, shape=(n_samples), dtype=float or int
        samples
    bins : array, shape=(n_bins)
        boundaries of histogram bins
    """

    def __init__(self, x,
                 bins=(-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf),
                 name='histogram'):
        super(Histogram, self).__init__(name=name)

        # check inputs
        x = safe_asarray(x)
        assert_all_finite(x)

        # making histogram
        hist = np.histogram(x, bins=bins)[0]

        # set statistics
        self.metrics['bin'] = list(bins)
        self.metrics['count'] = list(hist)
        self.metrics['density'] = list(hist / np.sum(hist))


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
