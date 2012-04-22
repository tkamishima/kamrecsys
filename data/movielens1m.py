#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert MovieLens 1M data sets to PyRecSys Sample format

Instruction
-----------

1. Download original file, ml-1m.zip, from `MovieLens Data Sets
   <http://www.grouplens.org/node/73>`_.
2. Unpack this ml-1m.zip, and place the following files at this directory:
   ratings.dat, users.dat, movies.dat
3. Run this script. As default, converted files are generated at
   ../pyrecsys/datasets/samples/ directory. If you want change the target
   directory, you need to specify it as the first argument of this script.
4. Remove original files, if you do not need them.
"""

from __future__ import unicode_literals

import os
import sys
import io
import re

# set directories

stem = 'movielens1m'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', "pyrecsys", 'datasets', 'samples')

# convert event files ---------------------------------------------------------

infile = open(os.path.join(pwd, 'ratings.dat'), 'r')
outfile = open(os.path.join(target, stem + '.event'), 'w')

outfile.write(
"""# Movielens 1M data set
#
# Original files are distributed by the Grouplens Research Project at the site:
# http://www.grouplens.org/node/73
# To use this data, follow the license permitted by the original distributor.
#
# This data set consists of:
# * 1,000,209 ratings (1-5) from 6040 users on 3706 movies. 
# * Each user has rated at least 20 movies. 
#
# Notes
# -----
# There are 3883 movies in an original file, but 3706 movies are actually
# rated.
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
#     represented in seconds since the epoch as returned by time(2)
""")

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    outfile.write("\t".join(f) + "\n")

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

infile = open(os.path.join(pwd, 'users.dat'), 'r')
outfile = open(os.path.join(target, stem + '.user'), 'w')

outfile.write(
"""# User feature file for ``movielens1m.event``.
# 
# The number of users is 6040.
#
# Format
# ------
# user : int
#     user id of the users which is compatible with the event file.
# gender : int
#     gender of the user, {0:male, 1:female}
# age : int, {0, 1,..., 6}
#     age of the user, where
#     1:"Under 18", 18:"18-24", 25:"25-34", 35:"35-44", 45:"45-49",
#     50:"50-55", 56:"56+"
# occupation : int, {0,1,...,20}
#     the number indicates the occupation of the user as follows:
#     0:"other" or not specified, 1:"academic/educator",
#     2:"artist", 3:"clerical/admin", 4:"college/grad student"
#     5:"customer service", 6:"doctor/health care", 7:"executive/managerial"
#     8:"farmer", 9:"homemaker", 10:"K-12 student", 11:"lawyer",
#     12:"programmer", 13:"retired", 14:"sales/marketing", 15:"scientist",
#     16:"self-employed", 17:"technician/engineer", 18:"tradesman/craftsman",
#     19:"unemployed", 20:"writer"
# zip : str, length=5
#     zip code of 5 digits, which represents the residential area of the user
""")

age = {'1':0, '18':1, '25':2, '35':3, '45':4, '50':5, '56':6}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    outfile.write(f[0] + '\t')
    if f[1] == 'M':
        outfile.write('0\t')
    elif f[1] == 'F':
        outfile.write('1\t')
    else:
        raise ValueError
    outfile.write(str(age[f[2]]) + '\t' + f[3] + '\t' + f[4][0:5] + '\n')

infile.close()
outfile.close()

# convert item files ----------------------------------------------------------

infile = io.open(os.path.join(pwd, 'movies.dat'), 'r', encoding='cp1252')
outfile = io.open(os.path.join(target, stem + '.item'), 'w', encoding='utf_8')

outfile.write(
"""# Item feature file for ``movielens1m.event``.
#
# The number of movies is 3883.
#
# Format
# ------
# item : int
#     item id of the movie which is compatible with the event file.
# name : str, length=[7, 81]
#     title of the movie with release year
# year : int
#     released year
# genre : binary_int * 18
#     18 binary numbers represents a genre of the movie. 1 if the movie belongs
#     to the genre; 0 other wise. All 0 implies unknown. Each column
#     corresponds to the following genres:
#     0:Action, 1:Adventure, 2:Animation, 3:Children's, 4:Comedy, 5:Crime,
#     6:Documentary, 7:Drama, 8:Fantasy, 9:Film-Noir, 10:Horror, 11:Musical,
#     12:Mystery, 13:Romance, 14:Sci-Fi, 15:Thriller, 16:War, 17:Western 
""")

genre = {"Action":0, "Adventure":1, "Animation":2, "Children's":3, "Comedy":4,
         "Crime":5, "Documentary":6, "Drama":7, "Fantasy":8, "Film-Noir":9,
         "Horror":10, "Musical":11, "Mystery":12, "Romance":13, "Sci-Fi":14,
         "Thriller":15, "War":16, "Western":17}
year_p = re.compile(r'\((\d\d\d\d)\)$')

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    if f[0] == '3845': # for buggy character in original file
        f[1] = re.sub('\&\#8230\;', '\u2026', f[1])
    outfile.write(f[0] + "\t" + f[1] + "\t")
    year = year_p.search(f[1]).group(1)
    outfile.write(year + '\t')
    d = f[2].split('|')
    o = ['0'] * 18
    for i in d:
        o[genre[i]] = '1'
    outfile.write('\t'.join(o) + '\n')

infile.close()
outfile.close()
