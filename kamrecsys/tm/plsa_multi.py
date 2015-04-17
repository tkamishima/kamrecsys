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

from ..recommenders import BaseEventScorePredictor

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


class EventScorePredictor(BaseEventScorePredictor):
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
        Raring distribution: Pr[R | Z]
    `n_iter_` : int
        nos of iteration after convergence
    `n_users_` : int
        nos of users
    `n_items_` : int
        nos of items
    `n_score_levels_` : int
        nos of score levels
    `n_events_` : int
        nos of events in training data

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
        self.tol = tol
        self.maxiter = maxiter
        self.alpha = alpha

        # attributes
        self.i_loss_ = np.inf
        self.f_loss_ = np.inf
        self.n_iter_ = 0
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

    def _init_model(self):
        """
        model initialization
        """

        # responsibilities
        self._q = self._rng.dirichlet(
            alpha=np.ones(self.k),
            size=(self.n_score_levels_, self.n_users_, self.n_items_))

        # model parameters
        self.pxgz_ = np.tile(
            self.alpha / self.n_users_, (self.n_users_, self.k))
        self.pygz_ = np.tile(
            self.alpha / self.n_items_, (self.n_items_, self.k))
        self.prgz_ = np.tile(
            self.alpha / self.n_score_levels_, (self.n_score_levels_, self.k))
        self.pz_ = np.tile(self.alpha / self.k, self.k)

    def _likelihood(self, ev, sc):
        """
        likelihood

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            digitized scores corresponding to events

        Returns
        -------
        likelihood : float
            negative log-likelihood of current model
        """

        l = np.sum(
            self.pz_[np.newaxis, :] *
            self.prgz_[sc, :] *
            self.pxgz_[ev[:, 0], :] *
            self.pygz_[ev[:, 1], :], axis=1)
        l = -np.sum(np.log(l)) / self.n_events_

        #----------------------------
        ll = 0
        for i in xrange(self.n_events_):
            ll += np.log(np.sum(
                self.pz_ *
                self.prgz_[sc[i], :] *
                self.pxgz_[ev[i, 0], :] *
                self.pygz_[ev[i, 1], :]))
        ll = - ll / self.n_events_
        if np.abs(l - ll) > 1e-14:
            logger.error("{:g} {:g}\n".format(l, ll))
        #----------------------------

        return l

    def fit(
            self, data, user_index=0, item_index=1, score_index=0,
            random_state=None):
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

        Notes
        -----
        * Currently `score_index` must be 0.
        * output intermediate results, if the logging level is lower than INFO
        """

        # initialization
        super(EventScorePredictor, self).fit(random_state=random_state)
        ev, sc, n_objects = (
            self._get_event_and_score(
                data, (user_index, item_index), score_index))
        self.n_users_ = n_objects[0]
        self.n_items_ = n_objects[1]
        self.n_score_levels_ = data.n_score_levels
        self.n_events_ = ev.shape[0]
        sc = data.digitize_score(sc)

        self._init_model()
        self.i_loss_ = self._likelihood(ev, sc)
        logger.info("initial: {:g}".format(self.i_loss_))
        pre_loss = self.i_loss_

        # main loop
        for iter_no in xrange(self.maxiter):

            # M-step ----------------------------------------------------------

            # n[r, x, y] P[z | r, x, y]
            n_rxyz = self._q[sc, ev[:, 0], ev[:, 1], :]
            n_total = np.sum(n_rxyz, axis=0, keepdims=True) + 1

            # p[r | z]
            self.prgz_ = (
                np.array([
                    np.bincount(
                        sc,
                        weights=n_rxyz[:, k],
                        minlength=self.n_score_levels_
                    ) for k in xrange(self.k)]).T
                + self.alpha / self.n_score_levels_) / n_total

            # p[x | z]
            self.pxgz_ = (
                np.array([
                    np.bincount(
                        ev[:, 0],
                        weights=n_rxyz[:, k],
                        minlength=self.n_users_
                    ) for k in xrange(self.k)]).T
                + self.alpha / self.n_users_) / n_total

            # p[y | z]
            self.pygz_ = (
                np.array([
                    np.bincount(
                        ev[:, 1],
                        weights=n_rxyz[:, k],
                        minlength=self.n_items_
                    ) for k in xrange(self.k)]).T
                + self.alpha / self.n_items_) / n_total

            # p[z]
            self.pz_[:] = np.sum(n_rxyz, axis=0) + self.alpha / self.k
            self.pz_[:] /= np.sum(self.pz_[:])

            # E-Step ----------------------------------------------------------

            # p[z | r, y, z]
            self._q = (
                self.pz_[np.newaxis, np.newaxis, np.newaxis, :] *
                self.prgz_[:, np.newaxis, np.newaxis, :] *
                self.pxgz_[np.newaxis, :, np.newaxis, :] *
                self.pygz_[np.newaxis, np.newaxis, :, :])
            self._q /= (np.sum(self._q, axis=3, keepdims=True))

            cur_loss = self._likelihood(ev, sc)
            logger.info("iter {:d}: {:g}".format(iter_no + 1, cur_loss))
            precision = np.abs((cur_loss - pre_loss) / cur_loss)
            if precision < self.tol:
                logger.info(
                    "Reached to specified tolerance: {:g}".format(precision))
                break
            pre_loss = cur_loss

        if iter_no >= self.maxiter - 1:
            logger.warning(
                "Exceeded the maximum number of iterations".format(
                    self.maxiter))

        self.f_loss_ = cur_loss
        logger.info("final: {:g}".format(self.f_loss_))
        self.n_iter_ = iter_no + 1
        logger.info("nos of iterations: {:d}".format(self.n_iter_))

        # clean garbage variables
        del self.__dict__['_q']

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

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
