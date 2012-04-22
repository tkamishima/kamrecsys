#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from pyrecsys.data import EventWithScoreData

infile = os.path.join('..', '..', 'datasets', 'samples', 'pci.event')
dtype = np.dtype([('event', 'S18', 2), ('score', np.float)])
x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
data = EventWithScoreData(n_otypes=2, n_stypes=1,
                          event_otypes=np.array([0, 1]))
data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0))

# test to_eid_event
check = data.to_eid_event(data.event)
if np.logical_and.reduce((x['event'] == check).ravel()):
    print("Ok")


# test to_eid_event / per line conversion
check = np.empty_like(data.event, dtype=x['event'].dtype)
for i, j in enumerate(data.event):
    check[i, :] = data.to_eid_event(j)
if np.logical_and.reduce((x['event'] == check).ravel()):
    print("Ok")
