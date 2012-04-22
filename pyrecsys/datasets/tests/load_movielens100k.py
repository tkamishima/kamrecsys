#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
>>> from pyrecsys.datasets import load_movielens100k
>>> data = load_movielens100k()
>>> print(data.__dict__.keys())
['event_otypes', 'n_otypes', 'n_events', 'feature', 'event', 'iid', 'event_feature', 'score', 'eid', 'n_objects', 'n_stypes', 's_event', 'score_domain']
>>> print(data.event_otypes)
[0 1]
>>> print(data.n_otypes)
2
>>> print(data.n_events)
100000
>>> print(data.s_event)
2
>>> print(data.n_objects)
[ 943 1682]
>>> print(data.n_stypes)
1
>>> print(data.score_domain)
[ 1.  5.]
>>> print(data.event[:5])
[[195 241]
 [185 301]
 [ 21 376]
 [243  50]
 [165 345]]
>>> print(data.event[-5:])
[[ 879  475]
 [ 715  203]
 [ 275 1089]
 [  12  224]
 [  11  202]]
>>> print(data.eid[0][:5])
[1 2 3 4 5]
>>> print(data.eid[0][-5:])
[939 940 941 942 943]
>>> print(data.eid[1][:5])
[1 2 3 4 5]
>>> print(data.eid[1][-5:])
[1678 1679 1680 1681 1682]
>>> print(data.event_feature.dtype)
[('timestamp', '<i8')]
>>> print(data.event_feature[:5])
[(881250949,) (891717742,) (878887116,) (880606923,) (886397596,)]
>>> print(data.event_feature[-5:])
[(880175444,) (879795543,) (874795795,) (882399156,) (879959583,)]
>>> print(data.score[:5])
[ 3.  3.  1.  2.  1.]
>>> print(data.score[-5:])
[ 3.  5.  1.  2.  3.]
>>> print(data.iid[0][1], data.iid[0][2], data.iid[0][3])
0 1 2
>>> print(data.iid[1][1], data.iid[1][2], data.iid[1][3])
0 1 2
>>> print(data.iid[0][943], data.iid[0][942], data.iid[0][900])
942 941 899
>>> print(data.iid[1][1682], data.iid[1][1681], data.iid[1][1000])
1681 1680 999
>>> print(data.feature[0][:5])
[(24, 0, 19, '85711') (53, 1, 1, '94043') (23, 0, 20, '32067')
 (24, 0, 19, '43537') (33, 1, 1, '15213')]
>>> print(data.feature[0][-5:])
[(26, 1, 18, '33319') (32, 0, 2, '02215') (20, 0, 18, '97229')
 (48, 1, 12, '78209') (22, 0, 18, '77841')]
>>> print(len(data.feature[0]))
943
>>> print(data.feature[1][:5])
[ (u'Toy Story (1995)', (1, 1, 1995), [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)')
 (u'GoldenEye (1995)', (1, 1, 1995), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?GoldenEye%20(1995)')
 (u'Four Rooms (1995)', (1, 1, 1995), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)')
 (u'Get Shorty (1995)', (1, 1, 1995), [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995)')
 (u'Copycat (1995)', (1, 1, 1995), [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?Copycat%20(1995)')]
>>> print(data.feature[1][-5:])
[ (u"Mat' i syn (1997)", (6, 2, 1998), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Mat%27+i+syn+(1997)')
 (u'B. Monkey (1998)', (6, 2, 1998), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 'http://us.imdb.com/M/title-exact?B%2E+Monkey+(1998)')
 (u'Sliding Doors (1998)', (1, 1, 1998), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'http://us.imdb.com/Title?Sliding+Doors+(1998)')
 (u'You So Crazy (1994)', (1, 1, 1994), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?You%20So%20Crazy%20(1994)')
 (u'Scream of Stone (Schrei aus Stein) (1991)', (8, 3, 1996), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'http://us.imdb.com/M/title-exact?Schrei%20aus%20Stein%20(1991)')]
>>> print(len(data.feature[1]))
1682
>>> from pyrecsys.datasets import movielens100k_info
>>> print(movielens100k_info['user_occupation'])
['None' 'Other' 'Administrator' 'Artist' 'Doctor' 'Educator' 'Engineer'
 'Entertainment' 'Executive' 'Healthcare' 'Homemaker' 'Lawyer' 'Librarian'
 'Marketing' 'Programmer' 'Retired' 'Salesman' 'Scientist' 'Student'
 'Technician' 'Writer']
>>> print(movielens100k_info['item_genre'])
['Action' 'Adventure' 'Animation' "Children's" 'Comedy' 'Crime'
 'Documentary' 'Drama' 'Fantasy' 'Film-Noir' 'Horror' 'Musical' 'Mystery'
 'Romance' 'Sci-Fi' 'Thriller' 'War' 'Western']
"""

from __future__ import print_function

import doctest
doctest.testmod()
"""
from pyrecsys.datasets import load_movielens100k
data = load_movielens100k()
print(data.__dict__.keys())
print(data.event_otypes)
print(data.n_otypes)
print(data.n_events)
print(data.s_event)
print(data.n_objects)
print(data.n_stypes)
print(data.score_domain)
print(data.event[:5])
print(data.event[-5:])
print(data.eid[0][:5])
print(data.eid[0][-5:])
print(data.eid[1][:5])
print(data.eid[1][-5:])
print(data.event_feature.dtype)
print(data.event_feature[:5])
print(data.event_feature[-5:])
print(data.score[:5])
print(data.score[-5:])
print(data.iid[0][1], data.iid[0][2], data.iid[0][3])
print(data.iid[1][1], data.iid[1][2], data.iid[1][3])
print(data.iid[0][943], data.iid[0][942], data.iid[0][900])
print(data.iid[1][1682], data.iid[1][1681], data.iid[1][1000])
print(data.feature[0][:5])
print(data.feature[0][-5:])
print(len(data.feature[0]))
print(data.feature[1][:5])
print(data.feature[1][-5:])
print(len(data.feature[1]))
from pyrecsys.datasets import movielens100k_info
print(movielens100k_info['user_occupation'])
print(movielens100k_info['item_genre'])
"""