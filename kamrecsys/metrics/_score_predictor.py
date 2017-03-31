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

from . import mean_absolute_error, mean_squared_error

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
        if Ture, print report

    Returns
    -------
    stats : dict
        belief summary of prediction performance
    """

    # check inputs
    check_consistent_length(y_true, y_pred)
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)

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

def score_predictor_statistics(y_true, y_pred):
    """
    Full Statistics of prediction performance
    
    * n_samples
    * mean_absolute_error: mean, stdev
    * mean_squared_error: mean, rmse, stdev 
    * predicted: mean, stdev
    * true: mean, stdev

    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores
    disp : bool, optional, default=True
        if Ture, print report

    Returns
    -------
    stats : dict
        belief summary of prediction performance
    """

    # check inputs
    check_consistent_length(y_true, y_pred)
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)

    # calc statistics
    stats = {}

    stats['n_samples'] = y_true.size

    mean, stdev = mean_absolute_error(y_true, y_pred)
    stats['mean_absolute_error'] = {}
    stats['mean_absolute_error']['mean'] = mean
    stats['mean_absolute_error']['stdev'] = stdev

    rmse, mean, stdev = mean_squared_error(y_true, y_pred)
    stats['mean_squared_error'] = {}
    stats['mean_squared_error']['rmse'] = rmse
    stats['mean_squared_error']['mean'] = mean
    stats['mean_squared_error']['stdev'] = stdev

    stats['true'] = {}
    stats['true']['mean'] = np.mean(y_true)
    stats['true']['stdev'] = np.std(y_true)

    stats['predicted'] = {}
    stats['predicted']['mean'] = np.mean(y_pred)
    stats['predicted']['stdev'] = np.std(y_pred)

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









