#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Metrics
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

from ._score import (
    mean_absolute_error,
    mean_squared_error)
from .base import (
    BaseMetrics,
    DescriptiveStatistics,
    Histogram)
from .real import (
    BaseRealMetrics,
    MeanAbsoluteError,
    MeanSquaredError)

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'mean_absolute_error',
    'mean_squared_error',
    'BaseMetrics',
    'DescriptiveStatistics',
    'Histogram',
    'BaseRealMetrics',
    'MeanAbsoluteError',
    'MeanSquaredError']
