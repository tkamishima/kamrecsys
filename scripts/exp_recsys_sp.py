"""
Experimentation script for Score Predictors

Input Format
------------

An input dataset is a tab separated file.  Each line corresponds to one 
rating behavior. Each column represents is as follows: 

* 1. A user represented by External-ID
* 2. An item rated by the user, represented by External-ID,
* 3. A rating score given by the user to the item
* 4. A timestamp of rating behavior, optional.

Output Format
-------------

Outputs of prediction are stored in a `json` formatted file.  Top-level keys 
of the outputs are as follows: 

* `data` : the data, such as a domain of rating scores and availability of 
  timestamp. 
* `environment` : hardware, system software, and experimental script 
* `model` : model and its parameters used for prediction 
* `prediction` : predicted results, user-item pairs and predicted and true 
  rating scores. 
* `test` : conditions, time information in test
* `training` : conditions, time information in training

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
    * plsam : pLSA (multinomial / use expectation in prediction)
    * plsamm : pLSA (multinomial / use mode in prediction)

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
-C <C>, --lambda <C>
    regularization parameter, default=0.01.
-k <K>, --dim <K>
    the number of latent factors, default=1.
--alpha <ALPHA>
    smoothing parameter of multinomial pLSA
--tol <TOL>
    optimization parameter. the size of norm of gradient. default=1e-05.
--maxiter <MAXITER>
    maximum number of iterations is maxiter times the number of parameters.
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

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from kamrecsys import __version__ as kamrecsys_version
from kamrecsys.data import EventWithScoreData
from kamrecsys.datasets import event_dtype_timestamp
from kamrecsys.model_selection import interlace_group
from kamrecsys.utils import get_system_info, get_version_info, json_decodable

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014/07/06"
__version__ = "3.4.0"
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
            ('event_feature', event_dtype_timestamp)
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


def training(info, ev, tsc, event_feature=None, fold=0):
    """
    training model

    Parameters
    ----------
    info : dict
        Information about the target task
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

    # start new fold
    n_folds = info['test']['n_folds']
    logger.info("training fold = " + str(fold + 1) + " / " + str(n_folds))

    # generate event data
    data = EventWithScoreData(n_otypes=2)
    score_domain = info['data']['score_domain']
    if np.all(np.array(score_domain) == 0):
        score_domain = [
            np.min(tsc), np.max(tsc), np.min(np.diff(np.unique(tsc)))]
        info['data']['score_domain'] = score_domain
        logger.info("score domain is changed to " + str(score_domain))
    data.set_event(
        ev, tsc, score_domain=score_domain, event_feature=event_feature)

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    if 'start_time' not in info['training']:
        info['training']['start_time'] = [0] * n_folds
    info['training']['start_time'][fold] = start_time.isoformat()
    logger.info("training_start_time = " + start_time.isoformat())

    # create and learning model
    rec = info['model']['recommender'](**info['model']['options'])
    rec.fit(data)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    elapsed_time = end_time - start_time
    elapsed_utime = end_utime - start_utime
    if 'end_time' not in info['training']:
        info['training']['end_time'] = [0] * n_folds
    info['training']['end_time'][fold] = end_time.isoformat()
    logger.info("training_end_time = " + end_time.isoformat())

    if 'elapsed_time' not in info['training']:
        info['training']['elapsed_time'] = elapsed_time
    else:
        info['training']['elapsed_time'] += elapsed_time
    logger.info("training_elapsed_time = " +
                str(info['training']['elapsed_time']))
    if 'elapsed_utime' not in info['training']:
        info['training']['elapsed_utime'] = elapsed_utime
    else:
        info['training']['elapsed_utime'] += elapsed_utime
    logger.info("training_elapsed_utime = " +
                str(info['training']['elapsed_utime']))

    # preserve optimizer's outputs
    if 'results' not in info['training']:
        info['training']['results'] = [{}] * n_folds
    info['training']['results'][fold] = rec.fit_results_

    return rec


def testing(rec, info, ev, fold=0):
    """
    test and output results

    Parameters
    ----------
    rec : EventScorePredictor
        trained recommender
    info : dict
        Information about the target task
    ev : array, size=(n_events, 2), dtype=int
        array of events in external ids
    fold : int, default=0
        fold No.
    
    Returns
    -------
    esc : array, shape=(n_events,), dtype=float
        estimated scores
    """

    # start new fold
    n_folds = info['test']['n_folds']
    logger.info("test fold = " + str(fold + 1) + " / " + str(n_folds))

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    if 'start_time' not in info['test']:
        info['test']['start_time'] = [0] * n_folds
    info['test']['start_time'][fold] = start_time.isoformat()
    logger.info("test_start_time = " + start_time.isoformat())

    # prediction
    esc = rec.predict(ev)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    elapsed_time = end_time - start_time
    elapsed_utime = end_utime - start_utime

    if 'end_time' not in info['test']:
        info['test']['end_time'] = [0] * n_folds
    info['test']['end_time'][fold] = start_time.isoformat()
    logger.info("test_end_time = " + end_time.isoformat())
    if 'elapsed_time' not in info['test']:
        info['test']['elapsed_time'] = elapsed_time
    else:
        info['test']['elapsed_time'] += elapsed_time
    logger.info("test_elapsed_time = " + str(info['test']['elapsed_time']))
    if 'elapsed_utime' not in info['test']:
        info['test']['elapsed_utime'] = elapsed_utime
    else:
        info['test']['elapsed_utime'] += elapsed_utime
    logger.info("test_elapsed_utime = " + str(info['test']['elapsed_utime']))

    # preserve predictor's outputs
    if 'results' not in info['test']:
        info['test']['results'] = [{}] * n_folds
    info['test']['results'][fold] = {'n_events': ev.shape[0]}

    return esc


def holdout_test(info):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    """

    # prepare training data
    train_x = load_data(
        info['training']['file'],
        info['data']['has_timestamp'])

    # prepare test data
    if info['test']['file'] is None:
        raise IOError('hold-out test data is required')
    test_x = load_data(
        info['test']['file'],
        info['data']['has_timestamp'])
    if info['data']['has_timestamp']:
        ef = train_x['event_feature']
    else:
        ef = None

    # training
    rec = training(info, train_x['event'], train_x['score'], event_feature=ef)

    # test
    esc = testing(rec, info, test_x['event'])

    # set predicted result
    info['prediction']['event'] = test_x['event']
    info['prediction']['true'] = test_x['score']
    info['prediction']['predicted'] = esc
    if info['data']['has_timestamp']:
        info['prediction']['event_feature'] = (
            {'timestamp': test_x['event_feature']['timestamp']})


def cv_test(info):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    """

    # prepare training data
    x = load_data(
        info['training']['file'],
        info['data']['has_timestamp'])
    info['test']['file'] = info['training']['file']
    n_events = x.shape[0]
    ev = x['event']
    tsc = x['score']

    fold = 0
    esc = np.empty(n_events, dtype=float)
    cv = LeaveOneGroupOut()
    for train_i, test_i in cv.split(
            ev, groups=interlace_group(n_events, info['test']['n_folds'])):

        # training
        if info['data']['has_timestamp']:
            rec = training(
                info, ev[train_i], tsc[train_i],
                event_feature=x['event_feature'][train_i], fold=fold)
        else:
            rec = training(
                info, ev[train_i], tsc[train_i], fold=fold)

        # test
        esc[test_i] = testing(rec, info, ev[test_i], fold=fold)

        fold += 1

    # set predicted result
    info['prediction']['event'] = ev
    info['prediction']['true'] = tsc
    info['prediction']['predicted'] = esc
    if info['data']['has_timestamp']:
        info['prediction']['event_feature'] = {
            'timestamp': x['event_feature']['timestamp']}


def do_task(info):
    """
    Main task

    Parameters
    ----------
    info : dict
        Information about the target task
    """

    # suppress warnings in numerical computation
    np.seterr(all='ignore')

    # update information dictionary
    info['model']['type'] = 'score_predictor'
    info['model']['name'] = info['model']['recommender'].__name__
    info['model']['module'] = info['model']['recommender'].__module__

    info['environment']['script'] = {
        'name': os.path.basename(sys.argv[0]), 'version': __version__}
    info['environment']['system'] = get_system_info()
    info['environment']['version'] = get_version_info()
    info['environment']['version']['kamrecsys'] = kamrecsys_version

    # select validation scheme
    if info['test']['scheme'] == 'holdout':
        info['test']['n_folds'] = 1
        logger.info("the nos of folds is set to 1")
        holdout_test(info)
    elif info['test']['scheme'] == 'cv':
        cv_test(info)
    else:
        raise TypeError("Invalid validation scheme: {0:s}".format(opt.method))

    # output information
    outfile = info['prediction']['file']
    info['prediction']['file'] = str(outfile)
    for k in info.keys():
        if k not in ['prediction']:
            json_decodable(info)
    outfile.write(json.dumps(info))
    if outfile is not sys.stdout:
        outfile.close()

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
                    type=argparse.FileType('rb'))
    ap.add_argument('-o', '--out', dest='outfile', default=None,
                    type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))
    ap.add_argument('-t', '--test', dest='testfile', default=None,
                    type=argparse.FileType('rb'))

    # script specific options
    ap.add_argument('-m', '--method', type=str, default='pmf',
                    choices=['pmf', 'plsam', 'plsamm'])
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

    ap.add_argument('-C', '--lambda', dest='C', type=float, default=0.01)
    ap.add_argument('-k', '--dim', dest='k', type=int, default=1)
    ap.add_argument('--alpha', dest='alpha', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=1e-05)
    ap.add_argument('--maxiter', type=int, default=None)

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

    # disable logging messages by changing logging level
    if opt.verbose:
        logger.setLevel(logging.INFO)
        logging.getLogger('kamrecsys').setLevel(logging.INFO)

    # output option information
    logger.info("list of options:")
    for key_name, key_value in vars(opt).items():
        logger.info("{0}={1}".format(key_name, str(key_value)))

    return opt


def init_info(opt):
    """
    Initialize information dictionary

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed command-line options

    Returns
    -------
    info : dict
        Information about the target task
    """

    info = {'data': {}, 'environment': {}, 'training': {}, 'test': {},
            'model': {'options': {}}, 'prediction': {}}

    # files
    info['training']['file'] = opt.infile
    info['prediction']['file'] = opt.outfile
    info['test']['file'] = opt.testfile

    # model
    if opt.method == 'pmf':
        from kamrecsys.score_predictor import PMF
        info['model']['method'] = 'probabilistic matrix factorization'
        info['model']['recommender'] = PMF
        info['model']['options'] = {'C': opt.C, 'k': opt.k, 'tol': opt.tol}
        if opt.maxiter is not None:
            info['model']['options']['maxiter'] = opt.maxiter
    elif opt.method == 'plsam':
        from kamrecsys.score_predictor import MultinomialPLSA
        info['model']['method'] = 'multionomial pLSA - expectation'
        info['model']['recommender'] = MultinomialPLSA
        info['model']['options'] = {
            'alpha': opt.alpha, 'k': opt.k, 'use_expectation': True,
            'tol': opt.tol, 'maxiter': opt.maxiter}
    elif opt.method == 'plsamm':
        from kamrecsys.score_predictor import MultinomialPLSA
        info['model']['recommender'] = MultinomialPLSA
        info['model']['method'] = 'multionomial pLSA - mode'
        info['model']['options'] = {
            'alpha': opt.alpha, 'k': opt.k, 'use_expectation': False,
            'tol': opt.tol, 'maxiter': opt.maxiter}
    else:
        raise TypeError(
            "Invalid method name: {0:s}".format(info['model']['method']))
    info['model']['options']['random_state'] = opt.rseed

    # test
    info['test']['scheme'] = opt.validation
    info['test']['n_folds'] = opt.fold

    # data
    info['data']['score_domain'] = list(opt.domain)
    info['data']['has_timestamp'] = opt.timestamp
    info['data']['explicit_rating'] = True

    return info


def main():
    """ Main routine
    """
    # command-line arguments
    opt = command_line_parser()

    # collect assets and information
    info = init_info(opt)

    # do main task
    do_task(info)

# top level -------------------------------------------------------------------
# init logging system
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(level=logging.INFO,
                    format='[%(name)s: %(levelname)s'
                           ' @ %(asctime)s] %(message)s')
logger.setLevel(logging.ERROR)
logging.getLogger('kamrecsys').setLevel(logging.ERROR)

# Call main routine if this is invoked as a top-level script environment.
if __name__ == '__main__':

    main()

    sys.exit(0)
