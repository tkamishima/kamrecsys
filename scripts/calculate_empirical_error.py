#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate errors of predicted scores
"""

import sys
import numpy as np

from kamrecsys.datasets import *
from kamrecsys.mf.pmf import EventScorePredictor

# To get the same result in any execution
np.random.seed(1234)

# load Movielens 100k data
data = load_movielens100k()

# generate empty model
# `C` : regularization parameter
# `k` : the number of latent factors
# You can tune optimization parameters, such as tol
recommender = EventScorePredictor(C=0.1, k=5, tol=1e-05)

# fit model
# See also: scipy.optimize.fmin_cg
recommender.fit(data)

# predict scores for event data that is used for training model
sc = recommender.raw_predict(data.event)

# Convert events represented by internal ids to those by external ids
# See Also: doc/glossary.rst
eev = recommender.to_eid_event(data.event)

for i in xrange(data.n_events):
    sys.stdout.write("%d %d " % (eev[i, 0], eev[i, 1]) +
                     str(data.score[i]) + " " +
                     str(sc[i]) + " " +
                     str(np.abs(np.float(data.score[i]) - sc[i])) + " " +
                     str((np.float(data.score[i]) - sc[i]) ** 2) + "\n")