"""
test "matrix factorization score predictors"

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
    specify test file name
-o <OUTPUT>, --out <OUTPUT>
    specify output file name
-m <METHOD>, --method <METHOD>
    specify algorithm: default=pmf

    * pmf : probabilistic matrix factorization

-v <VALIDATION>, --validation <VALIDATION>
    validation scheme: default=holdout

    * holdout : tested on the specified hold-out data
    * cv : cross validation

-f <FOLD>, --fold <FOLD>
    the number of folds in cross validation, default=5
-n, --no-timestamp or --timestamp
    specify whether .event files has 'timestamp' information,
    default=timestamp
-d <DOMAIN>, --domain <DOMAIN>
    The domain of scores specified by three floats: min, max, increment
    default=auto
--header or --no-header
    output column information or not
    default=no-header
-C <C>, --lambda <C>
    regularization parameter, default=0.01.
-k <K>, --dim <K>
    the number of latent factors, default=1.
--tol <TOL>
    optimization parameter. the size of norm of gradient. default=1e-05.
--maxiter <MAXITER>
    maximum number of iterations is maxiter times the number of parameters.
    default=200
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
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import sys
import argparse
import os
import platform
import subprocess
import logging
import datetime
import numpy as np
import scipy as sp
import sklearn

from kamrecsys.data import EventWithScoreData
from kamrecsys.cross_validation import KFold

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014/07/06"
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
            ('event', int, 2),
            ('score', float),
            ('event_feature', np.dtype([('timestamp', int)]))
        ])
    else:
        dt = np.dtype([
            ('event', int, 2),
            ('score', float)
        ])

    # load training data
    x = np.genfromtxt(fname=fp, delimiter='\t', dtype=dt)

    # close file
    if fp is not sys.stdin:
        fp.close()

    return x


def training(opt, ev, tsc, event_feature=None, fold=0):
    """
    training model

    Parameters
    ----------
    opt : dict
        parsed command line options
    ev : array, size=(n_events, 2), dtype=int
        array of events in external ids
    tsc : array, size=(n_events,), dtype=float
        true scores
    event_feature : optional, structured array
        structured array of event features
    fold : int, default=0
        fold No.

    Returns
    -------
    rec : EventScorePredictor
        trained recommender
    """

    # generate event data
    data = EventWithScoreData(n_otypes=2, n_stypes=1)
    if np.all(opt.domain == [0, 0, 0]):
        score_domain = (
            np.min(tsc), np.max(tsc), np.min(np.diff(np.unique(tsc))))
    else:
        score_domain = tuple(opt.domain)
    logger.info("score_domain = " + str(score_domain))
    data.set_events(ev, tsc, score_domain=score_domain,
                    event_feature=event_feature)

    # init learning results
    if 'training_start_time' not in opt:
        opt.training_start_time = [0] * opt.fold
    if 'training_end_time' not in opt:
        opt.training_end_time = [0] * opt.fold
    if 'learning_i_loss' not in opt:
        opt.learning_i_loss = [np.inf] * opt.fold
    if 'learning_f_loss' not in opt:
        opt.learning_f_loss = [np.inf] * opt.fold
    if 'learning_opt_outputs' not in opt:
        opt.learning_opt_outputs = [None] * opt.fold

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    opt.training_start_time[fold] = start_time.isoformat()
    logger.info("training_start_time = " + start_time.isoformat())

    # create and learning model
    rec = EventScorePredictor(
        C=opt.C, k=opt.k, tol=opt.tol, maxiter=opt.maxiter,
        random_state=opt.rseed)
    rec.fit(data)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    elapsed_time = end_time - start_time
    elapsed_utime = end_utime - start_utime
    opt.training_end_time[fold] = end_time.isoformat()
    logger.info("training_end_time = " + end_time.isoformat())
    if 'training_elapsed_time' not in opt:
        opt.training_elapsed_time = elapsed_time
    else:
        opt.training_elapsed_time += elapsed_time
    logger.info("training_elapsed_time = " + str(opt.training_elapsed_time))
    if 'training_elapsed_utime' not in opt:
        opt.training_elapsed_utime = elapsed_utime
    else:
        opt.training_elapsed_utime += elapsed_utime
    logger.info("training_elapsed_utime = " + str(opt.training_elapsed_utime))

    # preserve optimizer's outputs
    opt.learning_i_loss[fold] = rec.i_loss_
    opt.learning_f_loss[fold] = rec.f_loss_
    opt.learning_opt_outputs[fold] = rec.opt_outputs_

    return rec


def testing(rec, fp, opt, ev, tsc, ts=None, fold=0):
    """
    test and output results

    Parameters
    ----------
    rec : EventScorePredictor
        trained recommender
    fp : file
        output file pointer
    opt : Options
        parsed command line options
    ev : array, size=(n_events, 2), dtype=int
        array of events in external ids
    tsc : array, size=(n_events,), dtype=float
        true scores
    ts : optional, array, size=(n_events,), dtype=int
        timestamps if available
    fold : int, default=0
        fold No.
    """

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    if 'test_start_time' not in opt:
        opt.test_start_time = [0] * opt.fold
    opt.test_start_time[fold] = start_time.isoformat()
    logger.info("test_start_time = " + start_time.isoformat())

    # prediction
    esc = rec.predict(ev)

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
    elapsed_time = end_time - start_time
    elapsed_utime = end_utime - start_utime
    if 'test_end_time' not in opt:
        opt.test_end_time = [0] * opt.fold
    opt.test_end_time[fold] = end_time.isoformat()
    logger.info("test_end_time = " + end_time.isoformat())
    if 'test_elapsed_time' not in opt:
        opt.test_elapsed_time = elapsed_time
    else:
        opt.test_elapsed_time += elapsed_time
    logger.info("test_elapsed_time = " + str(opt.test_elapsed_time))
    if 'test_elapsed_utime' not in opt:
        opt.test_elapsed_utime = elapsed_utime
    else:
        opt.test_elapsed_utime += elapsed_utime
    logger.info("test_elapsed_utime = " + str(opt.test_elapsed_utime))


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
    for (key_name, key_value) in vars(opt).items():
        print("#{0}={1}".format(key_name, str(key_value)), file=fp)

    # post process

    # close file
    if fp is not sys.stdout:
        fp.close()


def holdout_test(opt):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    opt : Option
        parsed command line options
    """

    # load training data
    train_x = load_data(opt.infile, opt.timestamp)

    # load test data
    if opt.testfile is None:
        raise IOError('hold-out test data is required')
    test_x = load_data(opt.testfile, opt.timestamp)
    if opt.timestamp:
        ef = train_x['event_feature']
    else:
        ef = None

    # training
    rec = training(opt, train_x['event'], train_x['score'], event_feature=ef)

    # test
    if opt.timestamp:
        ef = test_x['event_feature']['timestamp']
    else:
        ef = None

    testing(rec, opt.outfile, opt,
            test_x['event'], test_x['score'], ts=ef)

    # output tailing information
    finalize(opt.outfile, opt)


def cv_test(opt):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    opt : Option
        parsed command line options
    """

    # load training data
    x = load_data(opt.infile, opt.timestamp)
    n_events = x.shape[0]
    ev = x['event']
    tsc = x['score']
    if opt.timestamp:
        ef = x['event_feature']

    fold = 0
    for train_i, test_i in KFold(n_events, n_folds=opt.fold, interlace=True):

        # training
        if opt.timestamp:
            rec = training(opt, ev[train_i], tsc[train_i],
                             event_feature=ef[train_i], fold=fold)
        else:
            rec = training(opt, ev[train_i], tsc[train_i], fold=fold)

        # test
        if opt.timestamp:
            testing(rec, opt.outfile, opt,
                    ev[test_i], tsc[test_i],
                    ef[test_i]['timestamp'], fold=fold)
        else:
            testing(rec, opt.outfile, opt,
                    ev[test_i], tsc[test_i], fold=fold)

        fold += 1

    # output tailing information
    finalize(opt.outfile, opt)


def get_system_info():
    """
    Get System hardware information

    Returns
    -------
    info : dict
        Information about an operating system and a hardware.
    """
    # import subprocess
    # import platform

    info = {}

    # information collected by a platform package
    info['system'] = platform.system()
    info['node'] = platform.node()
    info['release'] = platform.release()
    info['version'] = platform.version()
    info['machine'] = platform.machine()
    info['processor'] = platform.processor()

    # obtain hardware information
    with open('/dev/null', 'w') as DEVNULL:
        if platform.system() == 'Darwin':
            process_pipe = subprocess.Popen(
                ['/usr/sbin/system_profiler',
                 '-detailLevel', 'mini', 'SPHardwareDataType'],
                stdout=subprocess.PIPE, stderr=DEVNULL)
            hard_info, _ = process_pipe.communicate()
            hard_info = hard_info.decode('utf-8').split('\n')[4:-2]
            hard_info = [i.lstrip(' ') for i in hard_info]
        elif platform.system() == 'FreeBSD':
            process_pipe = subprocess.Popen(
                ['/sbin/sysctl', 'hw'],
                stdout=subprocess.PIPE, stderr=DEVNULL)
            hard_info, _ = process_pipe.communicate()
            hard_info = hard_info.decode('utf-8').split('\n')
        elif platform.system() == 'Linux':
            process_pipe = subprocess.Popen(
                ['/bin/cat', '/proc/cpuinfo'],
                stdout=subprocess.PIPE, stderr=DEVNULL)
            hard_info, _ = process_pipe.communicate()
            hard_info = hard_info.decode('utf-8').split('\n')
        else:
            hard_info = []
    info['hardware'] = hard_info

    return info

def get_version_info():
    """
    Get version numbers of a Python interpreter and packages.  
    
    Returns
    -------
    info : dict
        Version numbers of a Python interpreter and packages. 
    """
    # import platform
    # import numpy as np
    # import scipy as sp
    # import sklearn

    info = {}

    info['python_compiler'] = platform.python_compiler()
    info['python_implementation'] = platform.python_implementation()
    info['python'] = platform.python_version()

    info['numpy'] = np.__version__
    info['scipy'] = sp.__version__
    info['sklearn'] = sklearn.__version__

    return info

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def main(opt):
    """ Main routine that exits with status code 0
    """

    # select validation scheme
    if opt.validation == 'holdout':
        opt.fold = 1
        logger.info("the nos of folds is set to 1")
        holdout_test(opt)
    elif opt.validation == 'cv':
        cv_test(opt)
    else:
        raise argparse.ArgumentTypeError(
            "Invalid validation scheme: {0:s}".format(opt.method))

    sys.exit(0)

# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    # set script name
    script_name = os.path.basename(sys.argv[0])

    # init logging system
    logger = logging.getLogger(script_name)
    logging.basicConfig(level=logging.INFO,
                        format='[%(name)s: %(levelname)s'
                               ' @ %(asctime)s] %(message)s')

    # command-line option parsing
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

    ap.add_argument("--rseed", type=int, default=None)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile', default=None,
                    type=argparse.FileType('rb'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE', default=sys.stdin,
                    type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile', default=None,
                    type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))
    ap.add_argument('-t', '--test', dest='testfile', default=None,
                    type=argparse.FileType('rb'))

    # script specific options
    ap.add_argument('-m', '--method', type=str, default='pmf',
                    choices=['pmf'])
    ap.add_argument('-v', '--validation', type=str, default='holdout',
                    choices=['holdout', 'cv'])
    ap.add_argument('-f', '--fold', type=int, default=5)
    ap.add_argument('-d', '--domain', nargs=3, default=[0, 0, 0], type=float)
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(timestamp=True)
    apg.add_argument('-n', '--no-timestamp',
                     dest='timestamp', action='store_false')
    apg.add_argument('--timestamp',
                     dest='timestamp', action='store_true')
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(header=False)
    apg.add_argument('--no-header', dest='header', action='store_false')
    apg.add_argument('--header', dest='header', action='store_true')
    ap.add_argument('-C', '--lambda', dest='C', type=float, default=0.01)
    ap.add_argument('-k', '--dim', dest='k', type=int, default=1)
    ap.add_argument('--tol', type=float, default=1e-05)
    ap.add_argument('--maxiter', type=float, default=200)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options

    # disable logging messages by changing logging level
    if not opt.verbose:
        logger.setLevel(logging.ERROR)
        logging.getLogger('kamrecsys').setLevel(logging.ERROR)

    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    # suppress warnings in numerical computation
    np.seterr(all='ignore')

    # select algorithm
    if opt.method == 'pmf':
        from kamrecsys.mf.pmf import EventScorePredictor
    else:
        raise argparse.ArgumentTypeError(
            "Invalid method name: {0:s}".format(opt.method))

    # output option information
    logger.info("list of options:")
    for key_name, key_value in vars(opt).items():
        logger.info("{0}={1}".format(key_name, str(key_value)))

    # call main routine
    main(opt)
