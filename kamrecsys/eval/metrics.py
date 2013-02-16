#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation: simple metrics for evaluating scores, relevance, etc.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

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

__all__ = []

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
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
