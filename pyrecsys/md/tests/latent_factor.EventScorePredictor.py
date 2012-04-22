#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
>>> import numpy as np
>>>
>>> from pyrecsys.datasets import *
>>> from pyrecsys.md.latent_factor import EventScorePredictor
>>>
>>> np.random.seed(1234)
>>>
>>> data = load_movielens_mini()
>>>
>>> recommender = EventScorePredictor(C=0.1, k=2)
>>> print(vars(recommender))
{'C': 0.1, 'n_otypes': 0, 'bu_': None, 'bi_': None, 'k': 2, 'p_': None, 'q_': None, '_coef': None, 'f_loss_': inf, 'iid': None, 'i_loss_': inf, 'eid': None, 'n_objects': None, '_dt': None, 'mu_': None}
>>>
>>> recommender.fit(data, disp=True, gtol=1e-03)
Optimization terminated successfully.
         Current function value: 0.041362
         Iterations: 28
         Function evaluations: 55
         Gradient evaluations: 55
>>> for u in [1, 3, 5]:
...     for i in [7, 9, 11]:
...         print(u, i, recommender.predict((u, i)))
...
1 7 4.00074631485
1 9 4.98286035672
1 11 3.44741578214
3 7 3.89716397809
3 9 4.20400627475
3 11 3.66306486366
5 7 3.7468479513
5 9 3.96853184458
5 11 3.60148694779
>>> x = np.array([[1, 7], [1, 9], [1, 11], [3, 7], [3, 9], [3, 11], [5, 7], [5, 9], [5, 11]])
>>> print(recommender.predict(x))
[ 4.00074631  4.98286036  3.44741578  3.89716398  4.20400627  3.66306486
  3.74684795  3.96853184  3.60148695]
"""
from __future__ import print_function

import sys
import doctest

doctest.testmod()

sys.exit(0)

"""
import numpy as np

from pyrecsys.datasets import *
from pyrecsys.md.latent_factor import EventScorePredictor

np.random.seed(1234)

data = load_movielens_mini()

recommender = EventScorePredictor(C=0.1, k=2)
print(vars(recommender))

recommender.fit(data, disp=True, gtol=1e-03)

for u in [1, 3, 5]:
    for i in [7, 9, 11]:
        print(u, i, recommender.predict((u, i)))

x = np.array([[1, 7], [1, 9], [1, 11], [3, 7], [3, 9], [3, 11], [5, 7], [5, 9], [5, 11]])
print(recommender.predict(x))
"""
