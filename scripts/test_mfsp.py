"""
Testing "matrix factorization score predictors" on a hold-out set

SYNOPSIS::

    SCRIPT [options]

Description
===========

Output
------

1. user ID
2. item ID
3. original score
4. predicted score
5. timestamp (if timestamp option is true)

Options
=======

-i <INPUT>, --in <INPUT>
    specify training file name
-t <TEST>, --test <TEST>
    specify testing file name
-o <OUTPUT>, --out <OUTPUT>
    specify output file name
-n, --no-timestamp or --timestamp
    specify whether .event files has 'timestamp' information,
    default=timestamp
--header or --no-header
    output column information or not
    default=no-header
-C <C>, --lambda <C>
    regularization parameter, default=0.01.
-k <K>, --dim <K>
    the number of latent factors, default=1.
-l <TOL>, --tol <TOL>
    optimization parameter. the size of norm of gradient. default=1e-05.
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
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
__date__ = "2014/07/06"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2014 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

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
import numpy as np

from kamrecsys.data import EventWithScoreData
from kamrecsys.mf.pmf import EventScorePredictor

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

def load_data(fp, ts):
    """
    load event with scores data

    Parameters
    ----------
    fp : string
        input file pointer
    ts : bool
        has timestamp field

    Returns
    -------
    x : array
        structured array containing event and related information
    """

    # have timestamp?
    if ts:
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

    # load training data
    x = np.genfromtxt(fname=fp, delimiter='\t', dtype=dt)

    # close file
    if fp is not sys.stdin:
        fp.close()

    return x


def training(opt, ev, tsc, event_feature=None):
    """
    training model

    Parameters
    ----------
    opt : dict
        parsed command line options
    ev : array, size=(n_events, 2), dtype=np.int
        array of events in external ids
    tsc : array, size=(n_events,), dtype=np.float
        true scores
    event_feature : optional, structured array
        structured array of event features

    Returns
    -------
    rcmdr : EventScorePredictor
        trained recommender
    """

    # generate event data
    data = EventWithScoreData(n_otypes=2, n_stypes=1)
    score_domain = (np.min(tsc), np.max(tsc))
    data.set_events(ev, tsc, score_domain=score_domain,
                    event_feature=event_feature)

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.training_start_time = start_time.isoformat()
    logger.info("training_start time = " + start_time.isoformat())

    # create and learing model
    rcmdr = EventScorePredictor(C=opt.C, k=opt.k, tol=opt.tol,
                                random_state=opt.rseed)
    rcmdr.fit(data)

    # preserve final loss
    opt.learning_f_loss = rcmdr.f_loss_

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    logger.info("training_end time = " + end_time.isoformat())
    opt.training_end_time = end_time.isoformat()
    logger.info("training_elapsed_time = " + str((end_time - start_time)))
    opt.training_elapsed_time = str((end_time - start_time))
    logger.info("training_elapsed_utime = " + str((end_utime - start_utime)))
    opt.training_elapsed_utime = str((end_utime - start_utime))

    return rcmdr


def testing(rcmdr, fp, opt, ev, tsc, ts=None):
    """
    test and output results

    Parameters
    ----------
    rcmdr : EventScorePredictor
        trained recommender
    fp : file
        output file pointer
    opt : Options
        parsed command line options
    ev : array, size=(n_events, 2), dtype=np.int
        array of events in external ids
    tsc : array, size=(n_events,), dtype=np.float
        true scores
    ts : optional, array, size=(n_events,), dtype=np.int
        timestamps if available
    """

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.testing_start_time = start_time.isoformat()
    logger.info("testing_start time = " + start_time.isoformat())

    # prediction
    esc = rcmdr.predict(ev)

    # output evaluation results
    if ts is None:
        if opt.header:
            print('User ID', 'Item ID', 'Original Score', 'Predicted Score',
                  file=fp, sep='\t')
        for i in xrange(ev.shape[0]):
            print(ev[i, 0], ev[i, 1], tsc[i], esc[i],
                  file=fp, sep='\t')
    else:
        if opt.header:
            print('User ID', 'Item ID', 'Original Score', 'Predicted Score',
                  'Time Stamp',
                  file=fp, sep='\t')
        for i in xrange(ev.shape[0]):
            print(ev[i, 0], ev[i, 1], tsc[i], esc[i], ts[i],
                  file=fp, sep='\t')

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    logger.info("testing_end time = " + end_time.isoformat())
    opt.testing_end_time = end_time.isoformat()
    logger.info("testing_elapsed_time = " + str((end_time - start_time)))
    opt.testing_elapsed_time = str((end_time - start_time))
    logger.info("testing_elapsed_utime = " + str((end_utime - start_utime)))
    opt.testing_elapsed_utime = str((end_utime - start_utime))


def finalize(fp, opt):
    """
    output meta information and close output file

    Parameters
    ----------
        fp : file
            output file pointer
        opt : Option
            parsed command line options
    """

    # output option information
    for key, key_val in vars(opt).iteritems():
        print("#{0}={1}".format(key, str(key_val)), file=fp)

    ### post process

    # close file
    if fp is not sys.stdout:
        fp.close()


def holdout_test(opt):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    opt : Optionn
        parsed command line options
    """

    # load training data
    train_x = load_data(opt.infile, opt)

    # load testing data
    if opt.testfile is None:
        raise IOError('hold-out test data is required')
    test_x = load_data(opt.testfile, opt)

    # training
    if opt.timestamp:
        rcmdr = training(opt, train_x['event'], train_x['score'],
                         event_feature=train_x['event_feature'])
    else:
        rcmdr = training(opt, train_x['event'], train_x['score'])

    # testing
    if opt.timestamp:
        testing(rcmdr, opt.outfile, opt,
                test_x['event'], test_x['score'],
                test_x['event_feature']['timestamp'])
    else:
        testing(rcmdr, opt.outfile, opt,
                test_x['event'], test_x['score'])

    # output tailing information
    finalize(opt.outfile, opt)


#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Main routine
#==============================================================================


def main(opt):
    """ Main routine that exits with status code 0
    """

    holdout_test(opt)

    sys.exit(0)


### Preliminary processes before executing a main routine
if __name__ == '__main__':
    ### set script name
    script_name = os.path.basename(sys.argv[0])

    ### init logging system
    logger = logging.getLogger(script_name)
    logging.basicConfig(level=logging.INFO,
                        format='[%(name)s: %(levelname)s'
                               ' @ %(asctime)s] %(message)s')

    ### command-line option parsing
    ap = argparse.ArgumentParser(add_help=False)

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)
    ap.add_argument('-h', '--help', action='store_true', dest='help')
    ap.set_defaults(help=False)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

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
    ap.add_argument('-t', '--test', dest='testfile', default=None,
                    type=argparse.FileType('r'))

    # script specific options
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(timestamp=True)
    apg.add_argument('-n', '--no-timestamp', dest='timestamp',
                     action='store_false')
    apg.add_argument('--timestamp', dest='timestamp',
                     action='store_true')
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(header=False)
    apg.add_argument('--no-header', dest='header', action='store_false')
    apg.add_argument('--header', dest='header', action='store_true')
    ap.add_argument('-C', '--lambda', dest='C', type=float, default=0.01)
    ap.add_argument('-k', '--dim', dest='k', type=int, default=1)
    ap.add_argument('--tol', type=float, default=1e-05)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # help message
    if opt.help:
        print(__doc__, file=sys.stderr)
        sys.exit(0)

    # disable logging messages by changing logging level
    if not opt.verbose:
        logger.setLevel(logging.ERROR)

    # basic file i/o
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
        opt.sys_info = commands.getoutput(
            'system_profiler'
            ' -detailLevel mini SPHardwareDataType').split('\n')[4:-1]
    elif platform.system() == 'FreeBSD':
        opt.sys_info = commands.getoutput('sysctl hw').split('\n')
    elif platform.system() == 'Linux':
        opt.sys_info = commands.getoutput(
            'cat /proc/cpuinfo').split('\n')

    ### suppress warnings in numerical computation
    np.seterr(all='ignore')

    ### output option information
    logger.info("list of options:")
    for key, key_val in vars(opt).iteritems():
        logger.info("{0}={1}".format(key, str(key_val)))

    ### call main routine
    main(opt)
