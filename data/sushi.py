#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Convert sushi3 data sets to KamRecSys Sample format

Instruction
-----------

1. Download original file, ``sushi3b.tgz``, from `SUSHI Preference Data Sets
   <http://www.kamishima.net/sushi/>`_.
2. Unpack this ``sushi3b.tgz``, and place the following files at
   this directory:
   sushi3b.5000.10.score sushi.idata sushi.udata
3. Run this script. As default, converted files are generated at
   ``../kamrecsys/datasets/samples/`` directory. If you want change the target
   directory, you need to specify it as the first argument of this script.
4. Remove original files, if you do not need them.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

import os
import sys
import io

# set directories

stem = 'sushi3'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', 'kamrecsys', 'datasets', 'samples')

# convert event files of score ratings
# ---------------------------------------------------------

infile = open(os.path.join(pwd, 'sushi3b.5000.10.score'), 'r')
outfile = open(os.path.join(target, stem + 'b_score.event'), 'w')

print(
"""# Sushi3b score data set
#
# Original files are distributed by the Grouplens Research Project at the site:
# http://www.kamishima.net/sushi/
# To use this data, follow the license permitted by the original distributor.
#
# This data set consists of:
#
# * 5,000 ratings (0-4) from 5000 users on 100 sushis.
# * Each user has rated exactly 10 sushis selected at random.
#
# Format
# ------
# user : int
#     user id of the user who rated the sushi
# item : int
#     item id of the sushi rated by the user
# score : int
#     rating score whose range is {1, 2, 3, 4, 5}
""", file=outfile, end="")

uid = 0
for line in infile.readlines():
    rating = line.rstrip('\r\n').split(" ")
    for iid in xrange(len(rating)):
        if int(rating[iid]) >= 0:
            print(uid, iid, rating[iid], sep="\t", file=outfile)
    uid += 1

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

exit()

infile = open(os.path.join(pwd, 'u.user'), 'r')
outfile = open(os.path.join(target, stem + '.user'), 'w')

outfile.write(
"""# User feature file for ``movielens100k.event``.
# 
# The number of users is 943.
#
# Format
# ------
# user : int
#     user id of the users which is compatible with the event file.
# age : int
#     age of the user
# gender : int
#     gender of the user, {0:male, 1:female}
# occupation : int {0,1,...,20}
#     the number indicates the occupation of the user.
#     none:0, other:1, administrator:2, artist:3, doctor:4, educator:5,
#     engineer:6, entertainment:7, executive:8, healthcare:9, homemaker:10,
#     lawyer:11, librarian:12, marketing:13, programmer:14, retired:15,
#     salesman:16,  scientist:17, student:18, technician:19, writer:20
# zip : str, length=5
#     zip code of 5 digits, which represents the residential area of the user
""")
occupation = {'none':0, 'other':1, 'administrator':2, 'artist':3, 'doctor':4,
              'educator':5, 'engineer':6, 'entertainment':7, 'executive':8,
              'healthcare':9, 'homemaker':10, 'lawyer':11, 'librarian':12,
              'marketing':13, 'programmer':14, 'retired':15, 'salesman':16,
              'scientist':17, 'student':18, 'technician':19, 'writer':20}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("|")
    outfile.write(f[0] + '\t' + f[1] + '\t')
    if f[2] == 'M':
        outfile.write('0\t')
    elif f[2] == 'F':
        outfile.write('1\t')
    else:
        raise ValueError
    outfile.write(str(occupation[f[3]]) + '\t')
    outfile.write(f[4] + '\n')

infile.close()
outfile.close()

# convert item files ----------------------------------------------------------

infile = io.open(os.path.join(pwd, 'u.item'), 'r', encoding='cp1252')
outfile = io.open(os.path.join(target, stem + '.item'), 'w', encoding='utf_8')

outfile.write(
"""# Item feature file for ``movielens100k.event``.
#
# The number of movies is 1682.
#
# Format
# ------
# item : int
#     item id of the movie which is compatible with the event file.
# name : str, length=[7, 81]
#     title of the movie with release year
# date : int * 3, (day, month, year)
#     released date
# genre : binary_int * 18
#     18 binary numbers represents a genre of the movie. 1 if the movie belongs
#     to the genre; 0 other wise. All 0 implies unknown. Each column
#     corresponds to the following genres:
#     Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
#     Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
#     Thriller, War, Western
# imdb : str, length=[0, 134]
#      URL for the movie at IMDb http://www.imdb.com
""")

month = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
         'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("|")
    outfile.write(f[0] + "\t" + f[1] + "\t")
    d = f[2].split('-')
    if len(d) == 3:
        outfile.write("%d\t%d\t%d\t" % (int(d[0]), month[d[1]], int(d[2])))
    else:
        outfile.write("0\t0\t0\t")
    outfile.write("\t".join(f[6:24]))
    outfile.write("\t" + f[4] + "\n")

infile.close()
outfile.close()
