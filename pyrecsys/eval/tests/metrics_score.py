#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluation metrics for scores
"""

import numpy as np
from pyrecsys.eval.metrics import score_mae, score_rmse

sc1 = 1.0
sc2 = 3.0

np.testing.assert_almost_equal(score_mae(sc1, sc2), 2.0)
np.testing.assert_almost_equal(score_rmse(sc1, sc2), 2.0)

sc1 = np.arange(5)
sc2 = np.arange(5) + 2.0
np.testing.assert_almost_equal(score_mae(sc1, sc2), 2.0)
np.testing.assert_almost_equal(score_rmse(sc1, sc2), 2.0)

sc1 = np.arange(5)
sc2 = np.arange(5) * 2
np.testing.assert_almost_equal(score_mae(sc1, sc2), 2.0)
np.testing.assert_almost_equal(score_rmse(sc1, sc2), 2.4494897427831779)
