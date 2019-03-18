#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

##### Test Classes #####

class EventUtilMixin(unittest.TestCase):

    def test_to_eid_event(self):
        import os
        import numpy as np
        from pyrecsys.data import EventWithScoreData
        from pyrecsys.datasets import SAMPLE_PATH

        infile = os.path.join(SAMPLE_PATH, 'pci.event')
        dtype = np.dtype([('event', 'S18', 2), ('score', np.float)])
        x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
        data = EventWithScoreData(n_otypes=2, n_stypes=1,
                                  event_otypes=np.array([0, 1]))
        data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0))

        # test to_eid_event
        check = data.to_eid_event(data.event)
        assert_array_equal(x['event'], check)


        # test to_eid_event / per line conversion
        check = np.empty_like(data.event, dtype=x['event'].dtype)
        for i, j in enumerate(data.event):
            check[i, :] = data.to_eid_event(j)
        assert_array_equal(x['event'], check)

    def test_to_iid_event(self):
        import os
        import numpy as np
        from pyrecsys.data import EventWithScoreData
        from pyrecsys.datasets import SAMPLE_PATH

        infile = os.path.join(SAMPLE_PATH, 'pci.event')
        dtype = np.dtype([('event', 'S18', 2), ('score', np.float)])
        x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
        data = EventWithScoreData(n_otypes=2, n_stypes=1,
                                  event_otypes=np.array([0, 1]))
        data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0))

        # test EventData.to_iid_event
        assert_array_equal(data.event, data.to_iid_event(x['event']))

        # test EventData.to_iid_event / per line conversion
        check = np.empty_like(x['event'], dtype=np.int)
        for i, j in enumerate(x['event']):
            check[i, :] = data.to_iid_event(j)
        assert_array_equal(data.event, check)

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
