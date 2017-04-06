#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group Generator for cross validations 

Generated groups will be used with functions using groups, such as
:class:`sklearn.model_selection.LeaveOneGroupOut` .
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

def interlace_group(n_data, n_splits=3):
    """
    Generate interlace group.
    
    The i-th data is assigned to the (i mod n_splits)-th group.
    This is used with :class:`sklearn.model_selection.LeaveOneGroupOut` .
    In a case of a standard k-fold cross validation, subsequent data are tend
    to be grouped into the same fold.  Howeever, this is incovenient, if
    subsequent data are highly correlated.
    
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    n_data : int
        the number of data

    Returns
    -------
    group : array, shape=(n_data,)
        a sequence of indicator-numbers indicating the group assignment
    """
    pass

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
