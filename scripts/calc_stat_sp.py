#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate evaluation statistics for predicted scores

SYNOPSIS::

    SCRIPT [options]

Description
===========

errors between original and predicted scores

Output
------

1. The number of Samples
2. Mean True Score
3. Mean Predicted Score
4. Mean Absolute Error',
5. Mean Absolute Error (stdev)
6. Mean Squared Error
7. Mean Squared Error (stdev)
8. Root Mean Squared Error

Options
=======

-i <INPUT>, --in <INPUT>
    score predictor .result file
-o <OUTPUT>, --out <OUTPUT>
    statistics
-n, --no-timestamp or --timestamp
    specify whether .event files has 'timestamp' information,
    default=true
-j, --json
    output in a json format (default false)
-h, --help
    show this help message and exit
--version
    show program's version number and exit
"""

from __future__ import (
    print_function,
    division,
    absolute_import)

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014/06/17"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2014 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import os
import json
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions 
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Main routine
#==============================================================================


def main(opt):
    """ Main routine that exits with status code 0
    """

    ### load data

    if opt.timestamp:
        dt = np.dtype([
            ('event', np.int, 2),
            ('t_score', np.float),
            ('p_score', np.float),
            ('timestamp', np.int)
        ])
    else:
        dt = np.dtype([
            ('event', np.int, 2),
            ('t_score', np.float),
            ('p_score', np.float)
        ])
    x = np.genfromtxt(fname=opt.infile, delimiter='\t', dtype=dt)

    ### output statistics
    stats = []

    # mean scores
    stats.append(x['t_score'].shape[0])
    stats.append(np.mean(x['t_score']))
    stats.append(np.mean(x['p_score']))

    # absolute error
    errs = np.abs(x['t_score'] - x['p_score'])
    stats.append(np.mean(errs))
    stats.append(np.std(errs))

    # squared error
    errs = (x['t_score'] - x['p_score']) ** 2
    stats.append(np.mean(errs))
    stats.append(np.std(errs))
    stats.append(np.sqrt(np.mean(errs)))

    # output errors
    if opt.json:
        stats_name = [
            'nos_smaples',
            'mean_true_score',
            'mean_predicted_score',
            'mean_absolute_error',
            'mean_absolute_error_stdev',
            'mean_squared_error',
            'mean_squared_error_stdev',
            'root_mean_squared_error',
        ]
        json.dump(dict(zip(stats_name, stats)), opt.outfile)
    else:
        print(*stats, sep='\t', end='\n', file=opt.outfile)

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False)

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)
    ap.add_argument('-h', '--help', action='store_true', dest='help')
    ap.set_defaults(help=False)

    ap.add_argument("--rseed", type=int, default=None)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile', default=None,
                    type=argparse.FileType('r'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE', default=sys.stdin,
                    type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile', default=None,
                    type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(timestamp=True)
    apg.add_argument('-n', '--no-timestamp', dest='timestamp',
                     action='store_false')
    apg.add_argument('--timestamp', dest='timestamp',
                     action='store_true')
    ap.set_defaults(json=False)
    ap.add_argument('-j', '--json', dest='json', action='store_true')

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # help message
    if opt.help:
        print(__doc__, file=sys.stderr)
        sys.exit(0)

    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    np.seterr(all='ignore')

    main(opt)
