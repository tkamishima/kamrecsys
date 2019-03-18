#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate trained models for score predictors

SYNOPSIS::

    SCRIPT [options]

Options
-------
-i <INPUT>, --in <INPUT>
    file name of test data
-o <OUTPUT>, --out <OUTPUT>
    file name for writing evaluation statistics
-m <MODEL>, --model <MODEL>
    file name of a learned model
-q, --quiet
    set logging level to ERROR, no messages unless errors
-n, --notimestamp
    timestamps are not included in input file
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
--version
    show version

Notes
-----

An output format is as follows::

    <User ID><tab><Item ID><tab><True Score><tab><Estimated Score><tab>\
    <Absolute Residual><tab><Squared Residual><newline>
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/04/09"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2012 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import os
import platform
import commands
import logging
import datetime
import cPickle
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
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Main routine
#==============================================================================

def main(opt):
    """ Main routine that exits with status code 0
    """

    ### load data

    # load test data
    if opt.timestamp:
        dt = np.dtype([
            ('event', np.int, 2),
            ('score', np.float),
            ('event_feature', np.dtype([('timestamp', np.int)]))
        ])
    else:
        dt = np.dtype([
            ('event', np.int, 2),
            ('score', np.float)
        ])
    x = np.genfromtxt(fname=opt.infile, delimiter='\t', dtype=dt)
    ev = x['event']
    tsc = x['score']

    # load a learned model
    recommender = cPickle.load(opt.model)
    learning_opt = cPickle.load(opt.model)

    ### main process

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.start_time = start_time.isoformat()
    logger.info("start time = " + start_time.isoformat())

    # prediction
    esc = recommender.predict(ev)

    # output evaluation results
    for i in xrange(ev.shape[0]):
        opt.outfile.write("%d %d " % (ev[i, 0], ev[i, 1]) +
                          str(tsc[i]) + " " +
                          str(esc[i]) + " " +
                          str(np.abs(tsc[i] - esc[i])) + " " +
                          str((tsc[i] - esc[i]) ** 2) + "\n")

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    logger.info("end time = " + end_time.isoformat())
    opt.end_time = end_time.isoformat()
    logger.info("elapsed_time = " + str((end_time - start_time)))
    opt.elapsed_time = str((end_time - start_time))
    logger.info("elapsed_utime = " + str((end_utime - start_utime)))
    opt.elapsed_utime = str((end_utime - start_utime))

    # output option information
    for key, key_val in vars(opt).iteritems():
        opt.outfile.write("#%s=%s\n" % (key, str(key_val)))
    for key, key_val in learning_opt.iteritems():
        opt.outfile.write("#%s=%s\n" % (key, str(key_val)))

    ### post process

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


# Preliminary processes before executing a main routine

if __name__ == '__main__':
# set script name

    script_name = sys.argv[0].split('/')[-1]

    ### init logging system
    logger = logging.getLogger(script_name)
    logging.basicConfig(level=logging.INFO,
                        format='[%(name)s: %(levelname)s '\
                               '@ %(asctime)s] %(message)s')

    ### command-line option parsing
    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

    ap.add_argument("--rseed", type=int, default=None)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile',
                    default=None, type=argparse.FileType('r'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE',
                    default=sys.stdin, type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile',
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))
    ap.add_argument('-m', '--model', dest='model',
                    type=argparse.FileType('r'), required=True)

    # script specific options
    ap.add_argument('-n', '--notimestamp', dest='timastamp',
                    action='store_false')
    ap.set_defaults(timestamp=True)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # disable logging messages by changing logging level
    if not opt.verbose:
        logger.setLevel(logging.ERROR)

    ### set random seed
    np.random.seed(opt.rseed)

    ### basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    logger.info("input_file = " + opt.infile.name)
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']
    logger.info("output_file = " + opt.outfile.name)

    ### set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__
    opt.python_version = platform.python_version()
    opt.sys_uname = platform.uname()
    if platform.system() == 'Darwin':
        opt.sys_info =\
        commands.getoutput('system_profiler'
                           ' -detailLevel mini SPHardwareDataType').\
        split('\n')[4:-1]
    elif platform.system() == 'FreeBSD':
        opt.sys_info = commands.getoutput('sysctl hw').split('\n')
    elif platform.system() == 'Linux':
        opt.sys_info = commands.getoutput('cat /proc/cpuinfo').split('\n')

    ### suppress warnings in numerical computation
    np.seterr(all='ignore')

    ### call main routine
    main(opt)
