#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix Factorization: probabilistic matrix factorization model
"""

from __future__ import (
    print_function,
    division,
    absolute_import)

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import logging
import sys
import numpy as np
from scipy.optimize import fmin_cg
from sklearn.utils import check_random_state

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
    A probabilistic matrix factorization model proposed in [1]_.
    A method of handling bias terms is defined by equation (5) in [2]_.

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1
    tol : optional, float
        tolerance parameter for optimizer
    maxiter : int, default=200
        maximum number of iterations is maxiter times the number of parameters

    Attributes
    ----------
    `mu_` : array_like
        global bias
    `bu_` : array_like
        users' biases
    `bi_` : array_like
        items' biases
    `p_` : array_like
        latent factors of users
    `q_` : array_like
        latent factors of items
    `i_loss_` : float
        the loss value after initialization
    `f_loss_` : float
        the loss value after fitting
    `opt_outputs_` : tuple
        extra outputs of an optimizer

    Notes
    -----
    Preference scores are modeled by the sum of bias terms and the cross
    product of users' and items' latent factors with L2 regularizers.

    .. math::

       \hat{y} =
        \sum_{(u,i)\in\mathcal{D}}
        \mu + b_u + c_i + \mathbf{p}_u^\top \mathbf{q}_i
        + \lambda (\|P_u\|_2^2 + \|Q_u\|_2^2
        + \|\mathbf{b}\|_2^2 + \|\mathbf{c}\|_2^2)

    For computational reasons, a loss term is scaled by the number of
    events, and a regularization term is scaled by the number of model
    parameters.

    References
    ----------
    .. [1] R. Salakhutdinov and A. Mnih. "Probabilistic matrix factorization"
        NIPS 20
    .. [2] Y. Koren, "Factorization Meets the Neighborhood: A Multifaceted
        Collaborative Filtering Model", KDD2008
    """

    def __init__(self, C=1.0, k=1, tol=None, maxiter=200, random_state=None):
        super(EventScorePredictor, self).__init__(random_state=random_state)

        self.C = np.float(C)
        self.k = np.int(k)
        self.tol = tol
        self.maxiter = maxiter
        self.mu_ = None
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None
        self.i_loss_ = np.inf
        self.f_loss_ = np.inf
        self.opt_outputs_ = None

        # private instance variables
        self._coef = None
        self._dt = None

    def _init_coef(self, ev, sc, n_objects):
        """
        Initialize model parameters

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            scores attached to events
        n_objects : array, shape(2,)
            vector of numbers of objects
        """
        # constants
        n_events = ev.shape[0]
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', np.float, (1,)),
            ('bu', np.float, n_users + 1),
            ('bi', np.float, n_items + 1),
            ('p', np.float, (n_users + 1, k)),
            ('q', np.float, (n_items + 1, k))
        ])

        # memory allocation
        self._coef = np.zeros(1 + (n_users + 1) + (n_items + 1) +
                              (n_users + 1) * k + (n_items + 1) * k,
                              dtype=np.float)

        # set array's view
        self.mu_ = self._coef.view(self._dt)['mu'][0]
        self.bu_ = self._coef.view(self._dt)['bu'][0]
        self.bi_ = self._coef.view(self._dt)['bi'][0]
        self.p_ = self._coef.view(self._dt)['p'][0]
        self.q_ = self._coef.view(self._dt)['q'][0]

        # set bias term
        self.mu_[0] = np.sum(sc) / n_events
        for i in xrange(n_users):
            j = np.nonzero(ev[:, 0] == i)[0]
            if len(j) > 0:
                self.bu_[i] = np.sum(sc[j] - self.mu_[0]) / len(j)
        for i in xrange(n_items):
            j = np.nonzero(ev[:, 1] == i)[0]
            if len(j) > 0:
                self.bi_[i] = \
                    np.sum(sc[j] - (self.mu_[0] + self.bu_[ev[j, 0]])) / len(j)

        # fill cross terms by normal randoms whose s.d.'s are mean residuals
        var = 0.0
        for i in xrange(n_events):
            var += \
                (sc[i] -
                 (self.mu_[0] + self.bu_[ev[i, 0]] + self.bi_[ev[i, 1]])) ** 2
        var = var / n_events
        self.p_[0:n_users, :] = \
            self._rng.normal(0.0, np.sqrt(var), (n_users, k))
        self.q_[0:n_items, :] = \
            self._rng.normal(0.0, np.sqrt(var), (n_items, k))

        # scale a regularization term by the number of parameters
        self._reg = self.C / (1 + (k + 1) * (n_users + n_items))

    def loss(self, coef, ev, sc, n_objects):
        """
        loss function to optimize

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        sc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        loss : float
            value of loss function
        """
        # constants
        n_events = ev.shape[0]

        # set array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # loss term
        esc = (mu[0] + bu[ev[:, 0]] + bi[ev[:, 1]] +
               np.sum(p[ev[:, 0], :] * q[ev[:, 1], :], axis=1))
        loss = np.sum((sc - esc) ** 2)

        # regularization term
        reg = (np.sum(bu ** 2) + np.sum(bi ** 2) +
               np.sum(p ** 2) + np.sum(q ** 2))

        return loss / n_events + self._reg * reg

    def grad_loss(self, coef, ev, sc, n_objects):
        """
        gradient of loss function

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        sc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        grad : array_like, shape=coef.shape
            the first gradient of loss function by coef
        """
        # constants
        n_events = ev.shape[0]
        n_users = n_objects[0]
        n_items = n_objects[1]

        # set input array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # create empty gradient
        grad = np.zeros_like(coef)
        grad_mu = grad.view(self._dt)['mu'][0]
        grad_bu = grad.view(self._dt)['bu'][0]
        grad_bi = grad.view(self._dt)['bi'][0]
        grad_p = grad.view(self._dt)['p'][0]
        grad_q = grad.view(self._dt)['q'][0]

        # gradient of loss term
        neg_res = -(sc - (mu[0] + bu[ev[:, 0]] + bi[ev[:, 1]] +
                          np.sum(p[ev[:, 0], :] * q[ev[:, 1], :], axis=1)))
        grad_mu[0] = np.sum(neg_res)
        grad_bu[:] = np.bincount(ev[:, 0], weights=neg_res,
                                 minlength=n_users + 1)
        grad_bi[:] = np.bincount(ev[:, 1], weights=neg_res,
                                 minlength=n_items + 1)
        weights = neg_res[:, np.newaxis] * q[ev[:, 1], :]
        for i in xrange(self.k):
            grad_p[:, i] = np.bincount(ev[:, 0], weights=weights[:, i],
                                       minlength=n_users + 1)
        weights = neg_res[:, np.newaxis] * p[ev[:, 0], :]
        for i in xrange(self.k):
            grad_q[:, i] = np.bincount(ev[:, 1], weights=weights[:, i],
                                       minlength=n_items + 1)

        # re-scale gradients
        grad[:] = grad[:] / n_events

        # gradient of regularization term
        grad_bu[:] += self._reg * bu
        grad_bi[:] += self._reg * bi
        grad_p[:, :] += self._reg * p
        grad_q[:, :] += self._reg * q

        return grad

    def fit(self, data, user_index=0, item_index=1, score_index=0, tol=None,
            maxiter=None, random_state=None, **kwargs):
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
        random_state: RandomState or an int seed (None by default)
            A random number generator instance. If None is given, the
            object's random_state is used
        kwargs : keyword arguments
            keyword arguments passed to optimizers
        """

        # set random state
        if random_state is None:
            random_state = self.random_state
        self._rng = check_random_state(random_state)

        # get input data
        ev, sc, n_objects = \
            self._get_event_and_score(data,
                                      (user_index, item_index),
                                      score_index)

        # initialize coefficients
        self._init_coef(ev, sc, n_objects)

        # check optimization parameters
        if 'disp' not in kwargs:
            kwargs['disp'] = False
        if 'gtol' in kwargs:
            del kwargs['gtol']
        if self.tol is not None:
            kwargs['gtol'] = self.tol
        if maxiter is None:
            kwargs['maxiter'] = int(self.maxiter * self._coef.shape[0])
        else:
            kwargs['maxiter'] = int(maxiter * self._coef.shape[0])

        # get final loss
        self.i_loss_ = self.loss(self._coef, ev, sc, n_objects)

        # optimize model
        # fmin_bfgs is slow for large data, maybe because due to the
        # computation cost for the Hessian matrices.
        res = fmin_cg(self.loss,
                      self._coef,
                      fprime=self.grad_loss,
                      args=(ev, sc, n_objects),
                      full_output=True,
                      **kwargs)

        # get parameters
        self._coef[:] = res[0]
        self.f_loss_ = res[1]
        self.opt_outputs_ = res[2:]

        # clean up temporary instance variables
        del self._reg

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
