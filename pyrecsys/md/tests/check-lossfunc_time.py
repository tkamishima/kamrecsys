#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit

setup = """
import sys
import numpy as np
from pyrecsys.datasets import load_movielens100k
from pyrecsys.md.latent_factor import EventScorePredictor

np.random.seed(1234)

data = load_movielens100k()

recommender = EventScorePredictor(C=2.0, k=2)
ev, sc, n_objects = recommender._get_event_and_score(data, (0,1), 0)
recommender._init_coef(ev, sc, n_objects)
"""

#recommender.fit(data, maxiter=10, disp=True)

stmt = """
recommender.grad_loss(recommender.coef_, ev, sc, n_objects)
"""

t = timeit.Timer(stmt, setup)
print t.timeit(100)

stmt = """
recommender.loss(recommender.coef_, ev, sc, n_objects)
"""
