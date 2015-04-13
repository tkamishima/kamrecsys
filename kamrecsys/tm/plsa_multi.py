#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topic Model: probabilistic latent semantic analysis
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import logging
import sys
import numpy as np

from ._base import BaseTopicModel

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['EventScorePredictor']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class EventScorePredictor(BaseTopicModel):
    """
    A probabilistic latent semantic analysis model in [1]_ Figure 2(b).

    Parameters
    ----------
    k : int, optional
        the number of latent factors, default=1
    maxiter : int, default=100
        maximum number of iterations is maxiter times the number of parameters

    Parameters
    ----------
    k : int, default=1
        nos of latent factors
    maxiter : int, default=100
        maximum number of iterations
    alpha : float, default=1.0
        Laplace smoothing parameter

    Attributes
    ----------
    `pz_` : array_like
        Latent distribution: Pr[Z]
    `pxgz_` : array_like
        User distribution: Pr[X | Z]
    `pygz_` : array_like
        Item distribution: Pr[Y | Z]
    `prgz_` : array_like
        Rating distribution: Pr[R | Z]

    Notes
    -----

    3-way topic model: user x item x rating

    .. math::

       \Pr[X, Y, R] = \sum_{Z} \Pr[X | Z] \Pr[Y | Z] \Pr[R | Z] Pr[Z]

    References
    ----------
    .. [1] T. Hofmann and J. Puzicha. "Latent Class Models for Collaborative
        Filtering", IJCAI 1999
    """


    def __init__(
            self, k=1, tol=1e-5, maxiter=100, alpha=1.0, random_state=None):

        super(EventScorePredictor, self).__init__(random_state=random_state)

        # parameters
        self.k = k
        self.tol=tol
        self.maxiter=maxiter
        self.alpha = alpha

        # attributes
        self.pz_ = None
        self.pxgz_ = None
        self.pygz_ = None
        self.prgz_ = None
        self.n_users_ = 0
        self.n_items_ = 0
        self.n_score_levels_ = 0
        self.n_events_ = 0

        # internal vars
        self._q = None  # p[z | x, y]

    def _init_model(
            self, data,
            user_index=0, item_index=1, score_index=0):
        """
        model initialization
        """
        self.n_users_ = data.shape[0]
        self.n_items_ = data.shape[1]

        # responsibilities
        self._q = self._rng.dirichlet(alpha=np.ones(self.n_z),
                                      size=(self.n_x_, self.n_y_))

        # model parameters
        self.pxgz_ = np.tile(self.alpha / self.n_x_, (self.n_x_, self.n_z))
        self.pygz_ = np.tile(self.alpha / self.n_y_, (self.n_y_, self.n_z))
        self.pz_ = np.tile(self.alpha / self.n_z, self.n_z)

    def _likelihood(self, data):
        """
        likelihood

        Parameters
        ----------
        data : array, dtype=int, shape=(n_x, n_y)
            rows = x, columns = y, elements are the frequencies

        Returns
        -------
        likelihood : float
            negative log-likelihood of current model
        """

        l = np.sum(
            self.pz_[np.newaxis, np.newaxis, :] *
            self.pxgz_[:, np.newaxis, :] *
            self.pygz_[np.newaxis, :, :], axis=2)
        return - np.sum(data * np.log(l))

    def fit(
            self, data, user_index=0, item_index=1, score_index=0,
            disp=False, random_state=None):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventWithScoreData`
            data to fit
        user_index : optional, int
            Index to specify the position of a user in an event vector.
            (default=0)
        item_index : optional, int
            Index to specify the position of a item in an event vector.
            (default=1)
        score_index : optional, int
            Ignored if score of data is a single criterion type. In a multi-
            criteria case, specify the position of the target score in a score
            vector. (default=0)
        disp : bool, default=False
            print intermediate states

        Notes
        -----
        Currently `score_index` must be 0
        """

        # initialization
        super(EventScorePredictor, self).fit(random_state=random_state)
        ev, sc, n_objects = (
            self._get_event_and_score(
                data, (user_index, item_index), score_index))
        sc = data.digitize(sc)
        self._init_model(ev, sc, n_objects)



        self._total = np.sum(data)
        self.i_loss_ = self._likelihood(data)

        if disp:
            print("initial:", self.i_loss_, file=sys.stderr)

        # main loop
        for iter_no in xrange(self.n_iter):

            # M-step ----------------------------------------------------------

            # p[x | z]
            self.pxgz_[:, :] = (
                np.sum(data[:, :, np.newaxis] * self._q, axis=1)
                + self.alpha / self.n_x_)
            self.pxgz_ /= np.sum(self.pxgz_, axis=0, keepdims=True)

            # p[y | z]
            self.pygz_[:, :] = (
                np.sum(data[:, :, np.newaxis] * self._q, axis=0)
                + self.alpha / self.n_y_)
            self.pygz_ /= np.sum(self.pygz_, axis=0, keepdims=True)

            # p[z]
            self.pz_[:] = (
                np.sum(data[:, :, np.newaxis] * self._q, axis=(0, 1))
                + self.alpha / self.n_z)
            self.pz_[:] /= np.sum(self.pz_[:])

            # E-Step ----------------------------------------------------------

            # p[z | y, z]
            self._q = (
                self.pz_[np.newaxis, np.newaxis, :] *
                self.pxgz_[:, np.newaxis, :] * self.pygz_[np.newaxis, :, :])
            self._q /= (np.sum(self._q, axis=2, keepdims=True))

            if disp:
                print("iter ", iter_no + 1, ":", self._likelihood(data),
                      file=sys.stderr)

        self.f_loss_ = self._likelihood(data)
        if disp:
            print("final:", self.f_loss_, file=sys.stderr)

        # clean garbage variables
        del self.__dict__['_rng']
        del self.__dict__['_q']
        del self.__dict__['_total']

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
            a target user's and item's ids. unknwon objects assumed to be
            represented by n_object[event_otype]

        Returns
        -------
        sc : float
            score for a target pair of user and item

        Raises
        ------
        TypeError
            shape of an input array is illegal
        """

        if ev.ndim == 1:
            return (self.mu_[0] + self.bu_[ev[0]] + self.bi_[ev[1]] +
                    np.dot(self.p_[ev[0]], self.q_[ev[1]]))
        elif ev.ndim == 2:
            return (self.mu_[0] + self.bu_[ev[:, 0]] + self.bi_[ev[:, 1]] +
                    np.sum(self.p_[ev[:, 0], :] * self.q_[ev[:, 1], :],
                           axis=1))
        else:
            raise TypeError('argument has illegal shape')

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Module initialization 
# =============================================================================

# init logging system ---------------------------------------------------------
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

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

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
