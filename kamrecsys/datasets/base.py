#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common definitions of datasets
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging
import os

import numpy as np

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# path to the directory containing sample files
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'data')

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_event(infile, n_otypes=2, event_otypes=None, event_dtype=None):
    """
    load event file

    Tab separated file.  The contnts of columns are as follows:

    * the first s_events columns are sets of object IDs representing events 
    * the rest of columns corresponds to event features

    Parameters
    ----------
    infile : file or str
        input file if specified; otherwise, read from default sample directory.
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        event data with score information

        event_dtype : np.dtype, default=None
    """

    return


def load_event_with_score(
        infile, n_otypes=2, event_otypes=None, score_domain=(1, 5, 1),
        event_dtype=None):
    """
    load event file with rating score

    Tab separated file.  The contnts of columns are as follows:
    
    * the first s_events columns are sets of object IDs representing events 
    * the subsequent n_stypes columns are scores
      (WARNING: multiple scores (n_styls > 1) are not currently supported)
    * the rest of columns corresponds to event features
    
    Parameters
    ----------
    infile : file or str
        input file if specified; otherwise, read from default sample directory.
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    score_domain : optional, tuple or 1d-array of tuple
        min and max of scores, and the interval between scores
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        event data with score information
    
        event_dtype : np.dtype, default=None

    .. waraning::
    
        Multiple scores (n_stypes > 1) are not supported.
    """
    return

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
