#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
doctest pyrecsys.datasets.load_pci_sample
"""

def _test():
    """ test function for this module

    >>> from pyrecsys.datasets import load_pci_sample
    >>> data = load_pci_sample()
    >>> print vars(data)
    {'event_otypes': array([0, 1]), 'n_otypes': 2, 'n_events': 35, 'feature': array([None, None], dtype=object), 'event': array([[2, 1],
           [2, 2],
           [2, 5],
           [2, 3],
           [2, 4],
           [5, 1],
           [5, 2],
           [5, 0],
           [5, 3],
           [5, 5],
           [5, 4],
           [0, 2],
           [0, 0],
           [0, 5],
           [0, 3],
           [0, 4],
           [3, 1],
           [3, 2],
           [3, 0],
           [3, 3],
           [3, 4],
           [3, 5],
           [6, 2],
           [6, 3],
           [6, 5],
           [1, 1],
           [1, 2],
           [1, 0],
           [1, 3],
           [1, 5],
           [1, 4],
           [4, 1],
           [4, 2],
           [4, 3],
           [4, 4]]), 'iid': array([ {'Jack Matthews': 2, 'Mick LaSalle': 5, 'Claudia Puig': 0, 'Lisa Rose': 3, 'Toby': 6, 'Gene Seymour': 1, 'Michael Phillips': 4},
           {'Lady in the Water': 1, 'Just My Luck': 0, 'Superman Returns': 3, 'You, Me and Dupree': 5, 'Snakes on a Planet': 2, 'The Night Listener': 4}], dtype=object), 'event_feature': None, 'score': array([ 3. ,  4. ,  3.5,  5. ,  3. ,  3. ,  4. ,  2. ,  3. ,  2. ,  3. ,
            3.5,  3. ,  2.5,  4. ,  4.5,  2.5,  3.5,  3. ,  3.5,  3. ,  2.5,
            4.5,  4. ,  1. ,  3. ,  3.5,  1.5,  5. ,  3.5,  3. ,  2.5,  3. ,
            3.5,  4. ]), 'eid': array([ ['Claudia Puig' 'Gene Seymour' 'Jack Matthews' 'Lisa Rose'
     'Michael Phillips' 'Mick LaSalle' 'Toby'],
           ['Just My Luck' 'Lady in the Water' 'Snakes on a Planet' 'Superman Returns'
     'The Night Listener' 'You, Me and Dupree']], dtype=object), 'n_objects': array([7, 6]), 'n_stypes': 1, 's_event': 2, 'score_domain': (1.0, 5.0)}
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
