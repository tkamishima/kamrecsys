#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Convert MovieLens 100k data sets to KamRecSys Sample format

Instruction
-----------

1. Download original file, ml-100k.zip, from `MovieLens Data Sets
   <http://www.grouplens.org/node/73>`_.
2. Unpack this ml-100k.zip, and place the following files at this directory:
   u.data, u.user, and u.item.
3. Run this script. As default, converted files are generated at
   ../kamrecsys/datasets/samples/ directory. If you want change the target
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

# help message
if ('-h' in sys.argv) or ('--help' in sys.argv):
    print(__doc__, file=sys.stderr)
    sys.exit(0)

# set directories
stem = 'movielens100k'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', "kamrecsys", 'datasets', 'samples')

# convert event files ---------------------------------------------------------

infile = open(os.path.join(pwd, 'u.data'), 'r')
outfile = open(os.path.join(target, stem + '.event'), 'w')

print(
"""# Movielens 100k data set
#
# Original files are distributed by the Grouplens Research Project at the site:
# http://www.grouplens.org/node/73
# To use this data, follow the license permitted by the original distributor.
#
# This data set consists of:
# * 100,000 ratings (1-5) from 943 users on 1682 movies. 
# * Each user has rated at least 20 movies. 
#
# Format
# ------
# user : int
#     user id of the user who rated the movie
# item : int
#     item id of the movie rated by the user
# score : int
#     rating score whose range is {1, 2, 3, 4, 5}
# timestamp : int
#     UNIX seconds since 1/1/1970 UTC
""", end='', file=outfile)

for line in infile.readlines():
    f = line.rstrip('\r\n').split("\t")
    print("\t".join(f), file=outfile)

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

infile = open(os.path.join(pwd, 'u.user'), 'r')
outfile = open(os.path.join(target, stem + '.user'), 'w')

print(
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
""", end='', file=outfile)

occupation = {'none':0, 'other':1, 'administrator':2, 'artist':3, 'doctor':4,
              'educator':5, 'engineer':6, 'entertainment':7, 'executive':8,
              'healthcare':9, 'homemaker':10, 'lawyer':11, 'librarian':12,
              'marketing':13, 'programmer':14, 'retired':15, 'salesman':16,
              'scientist':17, 'student':18, 'technician':19, 'writer':20}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("|")
    print(f[0], f[1], sep='\t', end='\t', file=outfile)
    if f[2] == 'M':
        print('0', end='\t', file=outfile)
    elif f[2] == 'F':
        print('1', end='\t', file=outfile)
    else:
        raise ValueError
    print(occupation[f[3]], end='\t', file=outfile)
    print(f[4], file=outfile)

infile.close()
outfile.close()

# convert item files ----------------------------------------------------------

infile = io.open(os.path.join(pwd, 'u.item'), 'r', encoding='cp1252')
outfile = io.open(os.path.join(target, stem + '.item'), 'w', encoding='utf-8')

print(
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
""", end='', file=outfile)

month = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
         'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("|")
    print(f[0], f[1], sep='\t', end='\t', file=outfile)
    d = f[2].split('-')
    if len(d) == 3:
        print('{0:d}\t{1:d}\t{2:d}'.format(int(d[0]), month[d[1]], int(d[2])),
              end='\t', file=outfile)
    else:
        print('0', '0', '0', sep='\t', end='\t', file=outfile)
    print('\t'.join(f[6:24]), f[4], sep='\t', file=outfile)

infile.close()
outfile.close()
