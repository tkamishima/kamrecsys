#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
>>> from pyrecsys.datasets import load_movielens1m
>>> data = load_movielens1m()
>>> print data.__dict__.keys()
['event_otypes', 'n_otypes', 'n_events', 'feature', 'event', 'iid', 'event_feature', 'score', 'eid', 'n_objects', 'n_stypes', 's_event', 'score_domain']
>>> print data.event_otypes
[0 1]
>>> print data.n_otypes
2
>>> print data.n_events
1000209
>>> print data.s_event
2
>>> print data.n_objects
[6040 3706]
>>> print data.n_stypes
1
>>> print data.score_domain
(1.0, 5.0)
>>> print data.to_eid_event(data.event)[:5]
[[   1 1193]
 [   1  661]
 [   1  914]
 [   1 3408]
 [   1 2355]]
>>> print data.to_eid_event(data.event)[-5:]
[[6040 1091]
 [6040 1094]
 [6040  562]
 [6040 1096]
 [6040 1097]]
>>> print data.eid[0][:5]
[1 2 3 4 5]
>>> print data.eid[0][-5:]
[6036 6037 6038 6039 6040]
>>> print data.eid[1][:5]
[1 2 3 4 5]
>>> print data.eid[1][-5:]
[3948 3949 3950 3951 3952]
>>> print data.event_feature.dtype
[('timestamp', '<i8')]
>>> print data.event_feature[:5]
[(978300760,) (978302109,) (978301968,) (978300275,) (978824291,)]
>>> print data.event_feature[-5:]
[(956716541,) (956704887,) (956704746,) (956715648,) (956715569,)]
>>> print data.score[:5]
[ 5.  3.  3.  4.  5.]
>>> print data.score[-5:]
[ 1.  5.  5.  4.  4.]
>>> print data.iid[0][1], data.iid[0][2], data.iid[0][3]
0 1 2
>>> print data.iid[1][1], data.iid[1][2], data.iid[1][3]
0 1 2
>>> print data.iid[0][943], data.iid[0][942], data.iid[0][900]
942 941 899
>>> print data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]
1545 1544 936
>>> print data.feature[0][:5]
[(1, 0, 10, '48067') (0, 6, 16, '70072') (0, 2, 15, '55117')
 (0, 4, 7, '02460') (0, 2, 20, '55455')]
>>> print data.feature[0][-5:]
[(1, 2, 15, '32603') (1, 4, 1, '76006') (1, 6, 1, '14706')
 (1, 4, 0, '01060') (0, 2, 6, '11106')]
>>> print len(data.feature[0])
6040
>>> print data.feature[1][:5]
[ (u'Toy Story (1995)', 1995, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Jumanji (1995)', 1995, [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Grumpier Old Men (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
 (u'Waiting to Exhale (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Father of the Bride Part II (1995)', 1995, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
>>> print data.feature[1][-5:]
[ (u'Meet the Parents (2000)', 2000, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Requiem for a Dream (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Tigerland (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Two Family House (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 (u'Contender, The (2000)', 2000, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]
>>> print len(data.feature[1])
3706
>>> from pyrecsys.datasets import movielens1m_info
>>> print movielens1m_info['user_age']
['Under 18' '18-24' '25-34' '35-44' '45-49' '50-55' '56+']
>>> print movielens1m_info['user_occupation']
['Other or Not Specified' 'Academic/Educator' 'Artist' 'Clerical/Admin'
 'College/Grad Student' 'Customer Service' 'Doctor/Health Care'
 'Executive/Managerial' 'Farmer' 'Homemaker' 'K-12 Student' 'Lawyer'
 'Programmer' 'Retired' 'Sales/Marketing' 'Scientist' 'Self-Employed'
 'Technician/Engineer' 'Tradesman/Craftsman' 'Unemployed' 'Writer']
>>> print movielens1m_info['item_genre']
['Action' 'Adventure' 'Animation' "Children's" 'Comedy' 'Crime'
 'Documentary' 'Drama' 'Fantasy' 'Film-Noir' 'Horror' 'Musical' 'Mystery'
 'Romance' 'Sci-Fi' 'Thriller' 'War' 'Western']
"""

import doctest
doctest.testmod()
