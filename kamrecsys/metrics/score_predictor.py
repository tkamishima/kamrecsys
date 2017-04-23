#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarizers for Score Predictors
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

import sys
import logging
import json

import numpy as np
from sklearn.utils import (
    as_float_array, assert_all_finite, check_consistent_length)

import sklearn.metrics as skm

from . import mean_absolute_error, mean_squared_error, score_histogram

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


def score_predictor_report(y_true, y_pred, disp=True):
    """
    Report belief summary of prediction performance
    
    * mean absolute error
    * root mean squared error

    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores
    disp : bool, optional, default=True
        if True, print report

    Returns
    -------
    stats : dict
        belief summary of prediction performance
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calc statistics
    stats = {}
    stats['mean_absolute_error'] = skm.mean_absolute_error(y_true, y_pred)
    stats['root_mean_squared_error'] = np.sqrt(
        np.maximum(skm.mean_squared_error(y_true, y_pred), 0.))
    stats['n_samples'] = y_true.size
    stats['true'] = {
        'mean': np.mean(y_true),
        'stdev': np.std(y_true)}
    stats['predicted'] = {
        'mean': np.mean(y_pred),
        'stdev': np.std(y_pred)}

    # display statistics
    if disp:
        print(json.dumps(
            stats, sort_keys=True, indent=4, separators=(',', ': '),
            ensure_ascii=False),
            file=sys.stderr)

    return stats


def score_predictor_statistics(y_true, y_pred, score_domain=(1, 5, 1)):
    """
    Full Statistics of prediction performance
    
    * n_samples
    * mean_absolute_error: mean, stdev
    * mean_squared_error: mean, rmse, stdev 
    * predicted: mean, stdev
    * true: mean, stdev

    Parameters
    ----------
    y_true : array, shape=(n_samples,)
        Ground truth scores
    y_pred : array, shape=(n_samples,)
        Predicted scores
    score_domain : array, shape=(3,)
        Domain of scores, represented by a triple: start, end, and stride
        default=(1, 5, 1).

    Returns
    -------
    stats : dict
        Full statistics of prediction performance
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calc statistics
    stats = {}

    # dataset size
    stats['n_samples'] = y_true.size

    # mean absolute error
    mean, stdev = mean_absolute_error(y_true, y_pred)
    stats['mean_absolute_error'] = {'mean': mean, 'stdev':stdev}

    # root mean squared error
    rmse, mean, stdev = mean_squared_error(y_true, y_pred)
    stats['mean_squared_error'] = {'rmse': rmse, 'mean': mean, 'stdev': stdev}

    # descriptive statistics of ground truth scores
    stats['true'] = {'mean': np.mean(y_true), 'stdev':np.std(y_true)}

    hist, _ = score_histogram(y_true, score_domain=score_domain)
    stats['true']['histogram'] = hist
    stats['true']['histogram_density'] = (hist / hist.sum())

    # descriptive statistics of ground predicted scores
    stats['predicted'] = {'mean': np.mean(y_pred), 'stdev': np.std(y_pred)}

    hist, _ = score_histogram(y_pred, score_domain=score_domain)
    stats['predicted']['histogram'] = hist
    stats['predicted']['histogram_density'] = (hist / hist.sum())

    return stats


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









