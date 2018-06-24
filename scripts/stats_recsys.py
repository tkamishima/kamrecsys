#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate evaluation metrics

Input file
----------

Input file is formatted in json.  Prediction data is stored in the entry with a
top-level key: `prediction`.  The sub-entries in this entry are as follows:
 
* `event` : a list of user-item pairs represented by external IDs. 
* `true` : true rating scores corresponding to an event
* `predicted` : predicted rating scores corresponding to an event
* `event_feature` : event features

Output file
-----------

Statistics of prediction is stored in a json formatted file. 
The details are in the descriptions of statistics functions:

* :func:`kamrecsys.metrics.score_predictor_statistics`
* :func:`kamrecsys.metrics.item_finder_statistics`

Options
=======

-i <INPUT>, --in <INPUT>
    .json file generated by a training script
-o <OUTPUT>, --out <OUTPUT>
    statistics
--no-keepdata or --keepdata
    specify whether to keep the input whole data or not.
    default=no-keepdata
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

import argparse
import json
import sys

import numpy as np

from kamrecsys.metrics import (
    item_finder_statistics, score_predictor_statistics)
from kamrecsys.utils import json_decodable

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014-06-17"
__version__ = "3.0.0"
__copyright__ = "Copyright (c) 2014 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['do_task']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def do_task(opt):
    """
    Main task

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed command-line arguments
    """

    # suppress warnings in numerical computation
    np.seterr(all='ignore')

    # load data
    info = json.load(opt.infile, encoding='utf-8')

    # merge multi-fold data
    if isinstance(info['prediction']['event'], dict):
        info['prediction']['event'] = np.vstack(
            [np.asarray(info['prediction']['event'][k])
             for k in info['prediction']['event']])
        info['prediction']['true'] = np.hstack(
            [np.asarray(info['prediction']['true'][k])
             for k in info['prediction']['true']])
        info['prediction']['predicted'] = np.hstack(
            [np.asarray(info['prediction']['predicted'][k])
             for k in info['prediction']['predicted']])
        if 'event_feature' in info['prediction']:
            ef = info['prediction']['event_feature']
            for i in ef:
                features = [np.asarray(ef[i][k]) for k in ef[i]]
                if features[0].ndim == 2:
                    ef[i] = np.vstack(features)
                else:
                    ef[i] = np.hstack(features)

    # calculate statistics
    if info['model']['task_type'] == 'score_predictor':
        stats = score_predictor_statistics(
            info['prediction']['true'],
            info['prediction']['predicted'],
            score_domain=info['condition']['score_domain'])
    elif info['model']['task_type'] == 'item_finder':
        stats = item_finder_statistics(
            info['prediction']['true'],
            info['prediction']['predicted'])
    else:
        raise TypeError('Unsupported type of recommendation models')
    info['statistics'] = stats
    json_decodable(info['statistics'])

    # remove input data
    # if not opt.keepdata:
    #     del info['prediction']
    # else:
    json_decodable(info['prediction'])

    # output statistics
    opt.outfile.write(json.dumps(info))

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def command_line_parser():
    """
    Parsing Command-Line Options

    Returns
    -------
    opt : argparse.Namespace
        Parsed command-line arguments
    """
    # import argparse
    # import sys

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
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(keepdata=False)
    apg.add_argument('--no-keepdata', dest='keepdata',
                     action='store_false')
    apg.add_argument('--keepdata', dest='keepdata',
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

    return opt


def main():
    """ Main routine
    """
    # command-line arguments
    opt = command_line_parser()

    # do main task
    do_task(opt)


# top level -------------------------------------------------------------------
# Call main routine if this is invoked as a top-level script environment.
if __name__ == '__main__':

    main()

    sys.exit(0)
