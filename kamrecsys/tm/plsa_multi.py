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
    k : int, default=1
        nos of latent factors
    maxiter : int, default=100
        maximum number of iterations
    alpha : float, default=1.0
        Laplace smoothing parameter
    tol : float
        tolerance parameter of conversion
    use_expectation : bool, default=True
        use expecation in prediction if True, use mode if False

    Attributes
    ----------
    pz_ : array_like
        Latent distribution: Pr[Z]
    pxgz_ : array_like
        User distribution: Pr[X | Z]
    pygz_ : array_like
        Item distribution: Pr[Y | Z]
    prgz_ : array_like
        Raring distribution: Pr[R | Z]
    n_iter_ : int
        nos of iteration after convergence
    n_users_ : int
        nos of users
    n_items_ : int
        nos of items
    n_score_levels_ : int
        nos of score levels
    score_levels_ : array, dtype=float, shape=(n_score_levels_,)
        1d-array of score levels corresponding to each digitized score
    n_events_ : int
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
            self, k=1, tol=1e-5, maxiter=100, alpha=1.0, use_expectation=True,
            random_state=None):

        super(EventScorePredictor, self).__init__(random_state=random_state)

        # parameters
        self.k = k
        self.tol = tol
        self.maxiter = maxiter
        self.alpha = alpha
        self.use_expectation = use_expectation

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
        self.score_levels_ = None
        self.n_score_levels_ = 0
        self.n_events_ = 0

        # internal vars
        self._q = None  # p[z | x, y]

    def loss(self, ev, sc):
        """
        negative log-likelihood

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

        return l

    def _init_params(self, ev, sc):
        """
        initialize latent variables

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            digitized scores corresponding to events
        """
        self._q = self._rng.dirichlet(
            alpha=np.ones(self.k), size=self.n_events_)

    def maximization_step(self, ev, sc):
        """
        maximization step

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            digitized scores corresponding to events
        """

        # p[r | z]
        self.prgz_ = (
            np.array([
                         np.bincount(
                             sc,
                             weights=self._q[:, k],
                             minlength=self.n_score_levels_
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.prgz_ /= self.prgz_.sum(axis=1, keepdims=True)

        # p[x | z]
        self.pxgz_ = (
            np.array([
                         np.bincount(
                             ev[:, 0],
                             weights=self._q[:, k],
                             minlength=self.n_users_
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.pxgz_ /= self.pxgz_.sum(axis=1, keepdims=True)

        # p[y | z]
        self.pygz_ = (
            np.array([
                         np.bincount(
                             ev[:, 1],
                             weights=self._q[:, k],
                             minlength=self.n_items_
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.pygz_ /= self.pygz_.sum(axis=1, keepdims=True)

        # p[z]
        self.pz_ = np.sum(self._q, axis=0) + self.alpha
        self.pz_ /= np.sum(self.pz_)

        print("pZ =", self.pz_, sep='\n')
        print("pXgZ =\n",
              self.pxgz_[:3, ],
              self.pxgz_[497:503, :],
              self.pxgz_[-3:, :],
              sep='\n')
        print("pYgZ =\n",
              self.pygz_[:3, ],
              self.pygz_[497:503, :],
              self.pygz_[-3:, :],
              sep='\n')
        print("pRgZ =",
              self.prgz_,
              sep='\n')

    def fit(self, data, user_index=0, item_index=1, score_index=0,
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

        # initialization #####
        super(EventScorePredictor, self).fit(random_state=random_state)
        ev, sc, n_objects = (
            self._get_event_and_score(
                data, (user_index, item_index), score_index))
        self.n_users_ = n_objects[0]
        self.n_items_ = n_objects[1]
        self.n_score_levels_ = data.n_score_levels
        self.score_levels_ = np.linspace(
            data.score_domain[0], data.score_domain[1], self.n_score_levels_)
        self.n_events_ = ev.shape[0]
        sc = data.digitize_score(sc)
        self._init_params(ev, sc)

        # first m-step
        self.maximization_step(ev, sc)

        self.i_loss_ = self.loss(ev, sc)
        logger.info("initial: {:.15g}".format(self.i_loss_))
        pre_loss = self.i_loss_

        # main loop
        iter_no = 0
        cur_loss = np.inf
        for iter_no in xrange(self.maxiter):

            # E-Step

            # p[z | r, y, z]
            self._q = (
                self.pz_[np.newaxis, :] *
                self.prgz_[sc, :] *
                self.pxgz_[ev[:, 0], :] *
                self.pygz_[ev[:, 1], :])
            self._q /= (self._q.sum(axis=1, keepdims=True))

            # M-step
            self.maximization_step(ev, sc)

            # check loss
            cur_loss = self.loss(ev, sc)
            logger.info("iter {:d}: {:.15g}".format(iter_no + 1, cur_loss))
            precision = np.abs((cur_loss - pre_loss) / cur_loss)
            if precision < self.tol:
                logger.info(
                    "Reached to specified tolerance:"
                    " {:.15g}".format(precision))
                break
            pre_loss = cur_loss

        if iter_no >= self.maxiter - 1:
            logger.warning(
                "Exceeded the maximum number of iterations".format(
                    self.maxiter))

        self.f_loss_ = cur_loss
        logger.info("final: {:.15g}".format(self.f_loss_))
        self.n_iter_ = iter_no + 1
        logger.info("nos of iterations: {:d}".format(self.n_iter_))

        # clean garbage variables
        del self.__dict__['_q']

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        ev : array_like
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

        n_events = ev.shape[0]
        missing_values = self.n_objects[self.event_otypes]
        pRgXY = np.empty((n_events, self.n_score_levels_), dtype=float)

        for i in xrange(n_events):
            xi, yi = ev[i, :]

            if xi < missing_values[0] and yi < missing_values[1]:
                # known user and item: Pr[R | x, y]
                pRgXY[i, :] = np.sum(
                    self.pz_[np.newaxis, :] *
                    self.prgz_[:, :] *
                    self.pxgz_[xi, :][np.newaxis, :] *
                    self.pygz_[yi, :][np.newaxis, :], axis=1)
                pRgXY[i, :] /= pRgXY[i, :].sum()
            elif xi < missing_values[0]:
                # known user and unknwon item
                # marginal score over items: Pr[R | x]
                pRgXY[i, :] = np.sum(
                    self.pz_[np.newaxis, :] *
                    self.prgz_[:, :] *
                    self.pxgz_[xi, :][np.newaxis, :], axis=1)
                pRgXY[i, :] /= pRgXY[i, :].sum()
            elif yi < missing_values[1]:
                # unknown user and knwon item
                # marginal score over users: Pr[R | y]
                pRgXY[i, :] = np.sum(
                    self.pz_[np.newaxis, :] *
                    self.prgz_[:, :] *
                    self.pygz_[yi, :][np.newaxis, :], axis=1)
                pRgXY[i, :] /= pRgXY[i, :].sum()
            else:
                # unknown user and item: P[R]
                pRgXY[i, :] = np.sum(
                    self.pz_[np.newaxis, :] *
                    self.prgz_[:, :], axis=1)
                pRgXY[i, :] /= pRgXY[i, :].sum()

        if self.use_expectation:
            sc = np.dot(pRgXY, self.score_levels_[:, np.newaxis])
        else:
            sc = self.score_levels_[np.argmax(pRgXY, axis=1)]

        return sc

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
