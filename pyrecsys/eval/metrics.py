#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation: simple metrics for evaluating scores, relevance, etc.
"""

#==============================================================================
# Evaluation: metrics
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import sys
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['score_mae', 'score_rmse']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# # Functions
#==============================================================================

def score_mae(sc1, sc2):
    """
    MAE between true and estimated scores

    Parameters
    ----------
    sc1, sc2 : float or array_like, shape(variable,), dtype=float
        a pair of scores or vectors of scores to compare

    Notes
    -----
    MAE (Mean Absolute Error) is a mean of the absolute values of the
    differences between pairs of scores
     """
    sc1 = np.atleast_1d(np.asarray(sc1))
    sc2 = np.atleast_1d(np.asarray(sc2))

    return np.sum(np.abs(sc1 - sc2)) / np.float(sc1.shape[0])

def score_rmse(sc1, sc2):
    """
    MAE between true and estimated scores

    Parameters
    ----------
    sc1, sc2 : float or array_like, shape(variable,), dtype=float
        a pair of scores or vectors of scores to compare

    Notes
    -----
    RMSE (Root Mean Square Error) is a sqare root of a mean of the
    squared sum of the differences between pairs of scores
    """
    sc1 = np.atleast_1d(np.asarray(sc1))
    sc2 = np.atleast_1d(np.asarray(sc2))

    return np.sqrt(np.sum((sc1 - sc2) ** 2) / np.float(sc1.shape[0]))

#==============================================================================
# Module initialization 
#==============================================================================

# init logging system ---------------------------------------------------------

logger = logging.getLogger('pyrecsys')
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
