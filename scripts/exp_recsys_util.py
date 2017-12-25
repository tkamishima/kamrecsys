#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common part of experimental scripts
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import json
import logging
import os
import sys
import datetime

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from kamrecsys import __version__ as kamrecsys_version
from kamrecsys.model_selection import interlace_group
from kamrecsys.utils import get_system_info, get_version_info, json_decodable

# =============================================================================
# Metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2017-12-25"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2017 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['do_task']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def training(rec, data):
    """
    training model

    Parameters
    ----------
    rec : EventScorePredictor
        recommender object
    data : :class:`kamrecsys.data.EventWithScoreData`
        training data

    Returns
    -------
    res_info : dict
        Info of training results
    """

    # info of results
    res_info = {}

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    res_info['start_time'] = start_time.isoformat()
    logger.info("training_start_time = " + res_info['start_time'])

    # create and learning model
    rec.fit(data)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    res_info['end_time'] = end_time.isoformat()
    res_info['elapsed_time'] = end_time - start_time
    res_info['elapsed_utime'] = end_utime - start_utime
    logger.info("training_end_time = " + res_info['end_time'])
    logger.info("training_elapsed_time = " + str(res_info['elapsed_time']))
    logger.info("training_elapsed_utime = " + str(res_info['elapsed_utime']))

    # preserve optimizer's outputs
    res_info.update(rec.fit_results_)

    return res_info


def testing(rec, ev):
    """
    test and output results

    Parameters
    ----------
    rec : EventScorePredictor
        trained recommender
    ev : array, size=(n_events, 2), dtype=int
        array of events in external ids

    Returns
    -------
    esc : array, shape=(n_events,), dtype=float
        Estimated scores
    res_info : dict
        Info of training results
    """

    # info of results
    res_info = {}

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    res_info['start_time'] = start_time.isoformat()
    logger.info("test_start_time = " + res_info['start_time'])

    # prediction
    esc = rec.predict(ev)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    res_info['end_time'] = start_time.isoformat()
    res_info['elapsed_time'] = end_time - start_time
    res_info['elapsed_utime'] = end_utime - start_utime
    logger.info("test_end_time = " + res_info['end_time'])
    logger.info("test_elapsed_time = " + str(res_info['elapsed_time']))
    logger.info("test_elapsed_utime = " + str(res_info['elapsed_utime']))

    # preserve test info
    res_info['n_events'] = ev.shape[0]

    return esc, res_info


def holdout_test(info, load_data):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    """

    # set information about data and conditions
    info['test']['n_folds'] = 1

    # prepare training data
    train_data = load_data(info['training']['file'], info)

    # prepare test data
    if info['test']['file'] is None:
        raise IOError('hold-out test data is required')
    test_data = load_data(info['test']['file'], info)
    test_ev = test_data.to_eid_event(test_data.event)

    # training
    rec = info['model']['recommender'](**info['model']['options'])
    training_info = training(rec, train_data)
    info['training']['results'] = {0: training_info}

    # test
    esc, test_info = testing(rec, test_ev)
    info['test']['results'] = {0: test_info}

    # set predicted result
    info['prediction']['event'] = test_data.to_eid_event(test_data.event)
    info['prediction']['true'] = test_data.score
    info['prediction']['predicted'] = esc
    if info['data']['has_timestamp']:
        info['prediction']['event_feature'] = (
            {'timestamp': test_data.event_feature['timestamp']})


def cv_test(info, load_data):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    """

    # set information about data and conditions
    info['test']['file'] = info['training']['file']

    # prepare training data
    data = load_data(info['training']['file'], info)
    n_events = data.n_events
    n_folds = info['test']['n_folds']
    ev = data.to_eid_event(data.event)

    fold = 0
    esc = np.zeros(n_events, dtype=float)
    cv = LeaveOneGroupOut()
    info['training']['results'] = {}
    info['test']['results'] = {}
    for train_i, test_i in cv.split(
            ev, groups=interlace_group(n_events, n_folds)):

        # training
        logger.info("training fold = " + str(fold + 1) + " / " + str(n_folds))
        training_data = data.filter_event(train_i)
        rec = info['model']['recommender'](**info['model']['options'])
        training_info = training(rec, training_data)
        info['training']['results'][fold] = training_info

        # test
        logger.info("test fold = " + str(fold + 1) + " / " + str(n_folds))
        esc[test_i], test_info = testing(rec, ev[test_i])
        info['test']['results'][fold] = test_info

        fold += 1

    # set predicted result
    info['prediction']['event'] = ev
    info['prediction']['true'] = data.score
    info['prediction']['predicted'] = esc
    if info['data']['has_timestamp']:
        info['prediction']['event_feature'] = {
            'timestamp': data.event_feature['timestamp']}


def do_task(info, load_data):
    """
    Main task

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    """

    # suppress warnings in numerical computation
    np.seterr(all='ignore')

    # update information dictionary
    rec = info['model']['recommender']
    info['model']['task_type'] = rec.task_type
    info['model']['explicit_ratings'] = rec.explicit_ratings
    info['model']['name'] = rec.__name__
    info['model']['module'] = rec.__module__

    info['environment']['script'] = {
        'name': os.path.basename(sys.argv[0]), 'version': __version__}
    info['environment']['system'] = get_system_info()
    info['environment']['version'] = get_version_info()
    info['environment']['version']['kamrecsys'] = kamrecsys_version

    # select validation scheme
    if info['test']['scheme'] == 'holdout':
        holdout_test(info, load_data)
    elif info['test']['scheme'] == 'cv':
        cv_test(info, load_data)
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
# Module initialization
# =============================================================================

# init logging system
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script

if __name__ == '__main__':
    _test()
