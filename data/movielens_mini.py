#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Shrnked movielens100k.event data set

This data set is the subset of the data in the `movielens100k` data set.
Users and items whose external ids are less or equal than 10 are collected.

Instruction
-----------

1. Generate the `movielens100k.event` file by the script `movielens100k.py`
   here.
2. Run this script.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

import os
import sys

# help message
if ('-h' in sys.argv) or ('--help' in sys.argv):
    print(__doc__, file=sys.stderr)
    sys.exit(0)

# set directories
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', "kamrecsys", 'datasets', 'samples')

# convert event files ---------------------------------------------------------

infile = open(os.path.join(target, 'movielens100k.event'), 'r')
outfile = open(os.path.join(target, 'movielens_mini.event'), 'w')

print(
"""# Movielens mini data set
#
# This data set is the subset of the data in the `movielens100k` data set.
# Users and items whose external ids are less or equal than 10 are collected.
#
# 30 events in total. 8 users rate 10 items.
""", end='', file=outfile)

for line in infile.readlines():
    if line[0] == '#':
        continue
    f = line.rstrip('\r\n').split("\t")
    if int(f[0]) <= 10 and int(f[1]) <= 10:
        print(line, end='', file=outfile)

infile.close()
outfile.close()
