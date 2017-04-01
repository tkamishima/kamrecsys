#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate evaluation metrics

See a description of :func:`kamrecsys.metrics.score_predictor_statistics`
to the detail of metrices 

Options
=======

-i <INPUT>, --in <INPUT>
    score predictor .result file
-o <OUTPUT>, --out <OUTPUT>
    statistics
-n, --no-timestamp or --timestamp
    specify whether .event files has 'timestamp' information,
    default=true
-d <DOMAIN>, --domain <DOMAIN>
    The domain of scores specified by three floats: min, max, increment
    (default=1.0, 5,0, 1.0)
-h, --help
    show this help message and exit
--version
    show program's version number and exit
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import sys
import argparse
import os
import json
import numpy as np

from kamrecsys.metrics import score_predictor_statistics

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014/06/17"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2014 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def main(opt):
    """ Main routine that exits with status code 0
    """

    # load data
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

    # calculate statistics
    stats = score_predictor_statistics(
        x['t_score'], x['p_score'],
        scores=np.arange(opt.domain[0], opt.domain[1], opt.domain[2]))

    # output statistics
    json.dump(stats, opt.outfile)

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

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
    ap.add_argument('-d', '--domain', nargs=3, default=[1, 5, 1], type=float)
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(timestamp=True)
    apg.add_argument('-n', '--no-timestamp', dest='timestamp',
                     action='store_false')
    apg.add_argument('--timestamp', dest='timestamp',
                     action='store_true')

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options

    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    np.seterr(all='ignore')

    main(opt)
