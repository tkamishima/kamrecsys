#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load sample Flixster data sets
"""

from __future__ import (
    print_function,
    division,
    absolute_import)

#==============================================================================
# Imports
#==============================================================================

import sys
import os
import io
import logging
import numpy as np

from ..data import EventWithScoreData
from ._base import SAMPLE_PATH

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
# Classes
#==============================================================================

#==============================================================================
# Functions 
#==============================================================================

def load_flixster_rating(infile=None, event_dtype=None):
    """ load the sushi3b score data set

    An original data set is distributed at:
    `Mohsen Jamali <http://www.sfu.ca/~sja25/datasets/>`_.

    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----
    Format of events:

    * each event consists of a vector whose format is [user, item].
    * 8,196,077 events in total
    * 147,612 users rate 48,794 items (=movies)
    * dtype=np.int

    Format of scores:

    * one score is given to each event
    * domain of score is {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    * dtype=np.float
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'flixster.event')
    if event_dtype is None:
        dtype = np.dtype([('event', np.int, 2), ('score', np.float)])
    else:
        dtype = np.dtype([('event', np.int, 2), ('score', np.float),
                          ('event_feature', event_dtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, n_stypes=1)
    if event_dtype is None:
        data.set_events(x['event'], x['score'], score_domain=(0.5, 5.0))
    else:
        data.set_events(x['event'], x['score'], score_domain=(0.5, 5.0),
                        event_feature=x['event_feature'])

    return data

#==============================================================================
# Module initialization 
#==============================================================================

# init logging system ---------------------------------------------------------

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
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
