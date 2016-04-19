#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross Validation and Hold-out Test
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

from .sklearn_cross_validation import KFold

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['KFold']