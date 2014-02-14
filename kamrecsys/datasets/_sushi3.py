#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load sample sushi3 data sets
"""

from __future__ import (
    print_function,
    division,
    absolute_import)

#==============================================================================
# Imports
#==============================================================================

import sys
import os
import codecs
import logging
import numpy as np

from ..data import EventWithScoreData
from ._base import SAMPLE_PATH

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['SUSHI3_INFO', 'load_sushi3b_score']

#==============================================================================
# Constants
#==============================================================================

# Conversion tables for mapping the numbers to names for the ``sushi3``
# data set. available tables are ``user_age``, ``user_prefecture``,
# ``user_region``, and ``item_genre``.
SUSHI3_INFO = {
    'user_age': np.array([
        '15-19', '20-29', '30-39', '40-49', '50-59', '60-'
    ]),
    'user_prefecture': np.array([
        'Hokkaido', 'Aomori', 'Iwate', 'Akita', 'Miyagi',
        'Yamagata', 'Fukushima', 'Niigata', 'Ibaraki', 'Tochigi',
        'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
        'Yamanashi', 'Shizuoka', 'Nagano', 'Aichi', 'Gifu',
        'Toyama', 'Ishikawa', 'Fukui', 'Shiga', 'Mie',
        'Kyoto', 'Osaka', 'Nara', 'Wakayama', 'Hyogo',
        'Okayama', 'Hiroshima', 'Tottori', 'Shimane', 'Yamaguchi',
        'Ehime', 'Kagawa', 'Tokushima', 'Kochi', 'Fukuoka',
        'Nagasaki', 'Saga', 'Kumamoto', 'Kagoshima', 'Miyazaki',
        'Oita', 'Okinawa', 'non-Japan'
    ]),
    'user_region': np.array([
        'Hokkaido', 'Tohoku', 'Hokuriku', 'Kanto+Shizuoka', 'Nagano+Yamanashi',
        'Chukyo', 'Kinki', 'Chugoku', 'Shikoku', 'Kyushu',
        'Okinawa', 'non-Japan'
    ]),
    'item_genre': np.array([
        'aomono (blue-skinned fish)',
        'akami (red meat fish)',
        'shiromi (white-meat fish)',
        'tare (something like baste; for eel or sea eel)',
        'clam or shell',
        'squid or octopus',
        'shrimp or crab',
        'roe',
        'other seafood',
        'egg',
        'non-fish meat',
        'vegetable'
    ])
}

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions 
#==============================================================================

def load_sushi3b_score(infile=None,
                       event_dtype=np.dtype([('timestamp', np.int)])):
    """ load the MovieLens 100k data set

    Original file ``ml-100k.zip`` is distributed by the Grouplens Research
    Project at the site:
    `MovieLens Data Sets <http://www.grouplens.org/node/73>`_.

    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.
    event_dtype : np.dtype
        dtype of extra event features. as default, it consists of only a
        ``timestamp`` feature.

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----
    Format of events:

    * each event consists of a vector whose format is [user, item].
    * 100,000 events in total
    * 943 users rate 1682 items (=movies)
    * dtype=np.int

    Format of scores:

    * one score is given to each event
    * domain of score is [1.0, 2.0, 3.0, 4.0, 5.0]
    * dtype=np.float

    Default format of event_features ( `data.event_feature` ):
    
    timestamp : int
        UNIX seconds since 1/1/1970 UTC

    Format of user's feature ( `data.feature[0]` ):

    age : int
        age of the user
    gender : int
        gender of the user, {0:male, 1:female}
    occupation : int
        the number indicates the occupation of the user as follows:
        0:None, 1:Other, 2:Administrator, 3:Artist, 4:Doctor, 5:Educator,
        6:Engineer, 7:Entertainment, 8:Executive, 9:Healthcare, 10:Homemaker,
        11:Lawyer, 12:Librarian, 13:Marketing, 14:Programmer, 15:Retired,
        16:Salesman, 17:Scientist, 18:Student, 19:Technician, 20:Writer
    zip : str, length=5
        zip code of 5 digits, which represents the residential area of the user

    Format of item's feature ( `data.feature[1]` ):

    name : str, length=[7, 81], dtype=np.dtype('S81')
        title of the movie with release year
    date : int * 3
        released date represented by a tuple (year, month, day)
    genre : np.dtype(i1) * 18
        18 binary nunbers represents a genre of the movie. 1 if the movie
        belongs to the genre; 0 other wise. All 0 implies unknown. Each column
        corresponds to the following genres:
        Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
        Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
        Thriller, War, Western
    imdb : str, length=[0, 134], dtype=np.dtype('S134')
         URL for the movie at IMDb http://www.imdb.com
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'movielens100k.event')
    dtype = np.dtype([('event', np.int, 2),
                      ('score', np.float),
                      ('event_feature', event_dtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, n_stypes=1,
                              event_otypes=np.array([0, 1]))
    data.set_events(x['event'], x['score'], score_domain=(1.0, 5.0),
                    event_feature=x['event_feature'])

    # load user's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens100k.user')
    fdtype = np.dtype([('age', np.int), ('gender', np.int),
                       ('occupation', np.int), ('zip', 'S5')])
    dtype = np.dtype([('eid', np.int), ('feature', fdtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data.set_features(0, x['eid'], x['feature'])

    # load item's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens100k.item')
    fdtype = np.dtype([('name', 'U81'),
                       ('day', np.int),
                       ('month', np.int),
                       ('year', np.int),
                       ('genre', 'i1', 18),
                       ('imdb', 'S134')])
    dtype = np.dtype([('eid', np.int), ('feature', fdtype)])
    infile = codecs.open(infile, 'r', 'utf_8')
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data.set_features(1, x['eid'], x['feature'])

    del x

    return data

#==============================================================================
# Module initialization 
#==============================================================================

# init logging system ---------------------------------------------------------

logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module
    """

    # perform doctest
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
