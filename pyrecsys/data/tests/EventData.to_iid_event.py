#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np

import pyrecsys.data

infile = os.path.join('..', '..', 'datasets', 'samples', 'pci.event')
dtype = np.dtype([('event', 'S18', 2), ('score', np.float)])
x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
data = pyrecsys.data.EventWithScoreData(n_otypes=2, n_stypes=1,
                                        event_otypes=np.array([0, 1]))
data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0))

# test EventData.to_iid_event
if np.array_equal(data.event, data.to_iid_event(x['event'])):
    print("Ok")

# test EventData.to_iid_event / per line conversion
check = np.empty_like(x['event'], dtype=np.int)
for i, j in enumerate(x['event']):
    check[i, :] = data.to_iid_event(j)
if np.array_equal(data.event, check):
    print("Ok")
