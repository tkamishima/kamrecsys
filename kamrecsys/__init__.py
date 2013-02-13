#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KamRecSys: Algorithms for recommender systems in Python
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/03/25"
__version__ = "3.0.2"
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import logging

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['data', 'datasets', 'eval', 'mf', 'recommenders']

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
