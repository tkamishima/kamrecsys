#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary of __THIS_MODULE__
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

#==============================================================================
# Imports
#==============================================================================

import logging
import os

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['SAMPLE_PATH']

#==============================================================================
# Constants
#==============================================================================

# path to the directory containing sample files
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'samples')

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
