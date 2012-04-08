#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Convert MovieLens 100k data sets to PyRecSys Sample format

Instruction
-----------

1. Download original file, ml-100k.zip, from `MovieLens Data Sets
   <http://www.grouplens.org/node/73>`_.
2. Unpack this ml-100k.zip, and place the following files at this directory:
   u.data, u.user, and u.item.
3. Run this script. As default, converted files are generated at
   ../pyrecsys/datasets/samples/ directory. If you want change the target
   directory, you need to specify it as the first argument of this script.
4. Remove original files, if you do not need them.
"""

import os
import sys
import codecs

# set directories

stem = 'movielens100k'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', "pyrecsys", 'datasets', 'samples')

# convert event files ---------------------------------------------------------

infile = open(os.path.join(pwd, 'u.data'), 'r')
outfile = open(os.path.join(target, stem + '.event'), 'w')

outfile.write(
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
""")

for line in infile.readlines():
    f = line.rstrip('\r\n').split("\t")
    outfile.write("\t".join(f) + "\n")

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

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

infile = codecs.open(os.path.join(pwd, 'u.item'), 'r', 'cp1252')
outfile = codecs.open(os.path.join(target, stem + '.item'), 'w', 'utf_8')

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
