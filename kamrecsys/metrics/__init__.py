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
from ._score_predictor import (
    score_predictor_report,
    score_predictor_statistics)

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'mean_absolute_error',
    'mean_squared_error',
    'score_predictor_report',
    'score_predictor_statistics']
