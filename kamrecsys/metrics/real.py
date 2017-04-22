#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics for Predicted Scores
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
from sklearn.utils import (
    as_float_array, assert_all_finite, check_consistent_length)

from . import generate_score_bins

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


def mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error and its standard deviation.
    
    If you need only mean absolute error, use 
    :func:`sklearn.metrics.mean_absolute_error`
    
    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores

    Returns
    -------
    mean : float
        mean of squared errors
    stdev : float
        standard deviation of squared errors
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calculate errors
    errs = np.abs(y_true - y_pred)
    mean = np.mean(errs)
    stdev = np.std(errs)

    return mean, stdev


def mean_squared_error(y_true, y_pred):
    """
    Root mean squre error, mean square error, and its standard deviation.

    If you need only RMSE, use 
    :func:`sklearn.metrics.mean_absolute_error`

    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores

    Returns
    -------
    rmse : float
        root mean squared error
    mean : float
        mean of absolute errors
    stdev : float
        standard deviation of absolute errors
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calculate errors
    errs = (y_true - y_pred) ** 2
    mean = np.mean(errs)
    stdev = np.std(errs)
    rmse = np.sqrt(np.maximum(mean, 0.))

    return rmse, mean, stdev


def score_histogram(x, score_domain=(1, 5, 1)):
    """
    Histogram of scores 

    Parameters
    ----------
    x : array, shape=(n_samples), dtype=float or int
        A set of scores
    score_domain : array, shape=(3,) OR int, optional 
        Domain of scores, represented by a triple of the minimum, the maximum,
        and strides of the score, if array-like.  
        The range between the minimum and the maximum are divided into the
        specified number of bins, if int.
        default=(1, 5, 1).
    Returns
    -------
    hist : array_like, shape=(n_score_levels,)
        The number of data in each bin
    scores : array_like, shape=(n_score_levels + 1,)
        sequences of possible scores
    """

    # check inputs
    assert_all_finite(x)
    if isinstance(score_domain, np.integer):
        bins = score_domain
    else:
        assert_all_finite(score_domain)
        bins = generate_score_bins(score_domain)

    # making histogram
    hist, bins = np.histogram(x, bins=bins)

    # candidates of possible scores
    if isinstance(score_domain, np.integer):
        scores = (bins[1:] + bins[:-1]) / 2
    else:
        scores = np.hstack(
            [np.arange(score_domain[0], score_domain[1], score_domain[2],
                       dtype=float),
             score_domain[1]])

    # return statistics
    return hist, scores

# =============================================================================
# Classes
# =============================================================================

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
