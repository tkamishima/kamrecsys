#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common utilities for computing metrics
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


def generate_score_bins(score_domain):
    """
    Generate histogram bins for scores from score_domain

    Parameters
    ----------
    score_domain : array-like, shape=(3,)
        a triple of the minimum, the maximum, and strides of the score

    Returns
    -------
    score_bins : array, shape=(n_score_levels,), dtype=float
        bins of histogram. boundaries of bins are placed at the center of
        adjacent scores.
    """
    bins = np.arange(score_domain[0], score_domain[1], score_domain[2],
                     dtype=float)
    bins = np.hstack([-np.inf, bins + score_domain[2] / 2, np.inf])

    return bins


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system
logger = logging.getLogger('kamiers')
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
