#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics for Score Predictors
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

from . import BaseMetrics

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BaseRealMetrics',
           'MeanAbsoluteError']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions
#==============================================================================


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error

    Parameters
    ----------
    y_true : array, shape=(n_samples), dtype=float or int
        True values.
    y_pred : array, shape=(n_samples), dtype=float or int
        Estimated values

    Returns
    -------
    name : str
        returns "mean_squared_error"
    metrics : dict

        * mean : mean of errors
        * stdev : standard deviation of errors
        * rmse : root mean squared error
    """

    # check shape of inputs
    if y_true.shape != y_pred.shape:
        raise ValueError(
            'The sizes of true and predicted values must be equal')

    # empty metrics
    metrics = {}

    # mean absolute error
    errs = (y_true - y_pred) ** 2
    metrics['mean'] = np.mean(errs)
    metrics['stdev'] = np.std(errs)
    metrics['rmse'] = np.sqrt(metrics['mean'])

    return "mean_squared_error", metrics

#==============================================================================
# Classes
#==============================================================================


class BaseRealMetrics(BaseMetrics):
    """
    Base class for metrics between real vectors

    Parameters
    ----------
    y_true : array, shape=(n_samples), dtype=float or int
        True values.
    y_pred : array, shape=(n_samples), dtype=float or int
        Predicted values
    """

    __metaclass__ = ABCMeta

    def __init__(self, y_true, y_pred, name='real_metrics'):
        super(BaseRealMetrics, self).__init__(name=name)

        # check inputs
        self._y_true = safe_asarray(y_true)
        assert_all_finite(self._y_true)
        self._y_pred = safe_asarray(y_pred)
        assert_all_finite(self._y_pred)
        if self._y_true.shape != self._y_pred.shape:
            raise ValueError(
                'The sizes of true and predicted vectors must be equal')


class MeanAbsoluteError(BaseRealMetrics):
    """
    Mean Absolute Error

    * mean : mean absolute error
    * stdev : standard deviation of absolute errors 
    """

    def __init__(self, y_true, y_pred, name='mean_absolute_error'):

        super(MeanAbsoluteError, self).__init__(y_true, y_pred, name=name)

        # mean absolute error
        errs = np.abs(self._y_true - self._y_pred)
        self.metrics['mean'] = np.mean(errs)
        self.metrics['stdev'] = np.std(errs)

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
