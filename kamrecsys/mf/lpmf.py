#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix Factorization: logistic probabilistic matrix factorization model
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

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

from ..recommenders import BaseEventItemFinder

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['EventItemFinder']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class EventItemFinder(BaseEventItemFinder):
    """
    A probabilistic matrix factorization model proposed in [1]_.
    A method of handling bias terms is defined by equation (5) in [2]_.
    However, to deal with implicit ratings, a sigmoid function is additionally
    used.

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
    mu_ : array_like
        global bias
    bu_ : array_like
        users' biases
    bi_ : array_like
        items' biases
    p_ : array_like
        latent factors of users
    q_ : array_like
        latent factors of items
    i_loss_ : float
        the loss value after initialization
    f_loss_ : float
        the loss value after fitting
    opt_outputs_ : tuple
        extra outputs of an optimizer

    Notes
    -----
    Rating scores are modeled by the sum of bias terms and the cross
    product of users' and items' latent factors.
    To constrain that ratings so as to lie between zero to one, a sigmoid
    function is applied.
    
    .. math::
    
        \hat{r}_{xy} =  \sigma(
        \mu + b_x + c_y + \mathbf{p}_x^\top \mathbf{q}_y
        )
       
    Parameters of this model is estimated by optimizing a cross-entropy loss
    function with L2 regularizer

    .. math::

        \sum_{(x,y)}
        \frac{1}{N_x N_y}
        - \Big( r_{xy} \log \hat{r}_{xy} + 
                (1 - r_{xy}) \log(1 - \hat{r}_{xy})\Big) 
        + \lambda \Big(
        \|\mathbf{b}\|_2^2 + \|\mathbf{c}\|_2^2 +
        \|\mathbf{P}\|_2^2 + \|\mathbf{Q}\|_2^2
        \Big)

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

    # constant for clipping inputs in a logistic function
    sigmoid_range = 34.538776394910684

    def __init__(self, C=1.0, k=1, tol=None, maxiter=200, random_state=None):
        super(EventItemFinder, self).__init__(random_state=random_state)

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

    def _init_coef(self, ev, n_objects):
        """
        Initialize model parameters

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        n_objects : array, shape(2,)
            vector of numbers of objects
        """
        # constants
        n_positives = ev.count_nonzero()
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', np.float, (1,)),
            ('bu', np.float, n_users),
            ('bi', np.float, n_items),
            ('p', np.float, (n_users, k)),
            ('q', np.float, (n_items, k))
        ])

        # memory allocation
        self._coef = np.zeros(1 + n_users + n_items +
                              n_users * k + n_items * k,
                              dtype=float)

        # set array's view
        self.mu_ = self._coef.view(self._dt)['mu'][0]
        self.bu_ = self._coef.view(self._dt)['bu'][0]
        self.bi_ = self._coef.view(self._dt)['bi'][0]
        self.p_ = self._coef.view(self._dt)['p'][0]
        self.q_ = self._coef.view(self._dt)['q'][0]

        # set bias term
        self.mu_[0] = n_positives / (n_users * n_items)
        self.bu_[:] = ev.sum(axis=1).ravel() / n_items
        self.bi_[:] = ev.sum(axis=0).ravel() / n_users

        # fill cross terms by normal randoms whose s.d.'s are mean residuals
        self.p_[0:n_users, :] = (self._rng.normal(0.0, 1.0, (n_users, k)))
        self.q_[0:n_items, :] = (self._rng.normal(0.0, 1.0, (n_items, k)))

        # scale a regularization term by the number of parameters
        self._reg = self.C / (1 + (k + 1) * (n_users + n_items))

    def sigmoid(self, x):
        """
        sigmoid function

        Parameters
        ----------
        x : array_like, shape=(n_data), dtype=float
            arguments of function

        Returns
        -------
        sig : array, shape=(n_data), dtype=float
            1.0 / (1.0 + exp(- x))
        """

        # restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
        x = np.clip(x, -self.sigmoid_range, self.sigmoid_range)

        return 1.0 / (1.0 + np.exp(-x))

    def loss(self, coef, ev, n_objects):
        """
        loss function to optimize

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        loss : float
            value of loss function
        """
        # constants
        n_users = n_objects[0]
        n_events = n_objects[0] * n_objects[1]

        # set array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # loss term
        loss = 0.0
        for i in xrange(n_users):
            evi = ev.getrow(i).toarray().reshape(-1)
            esc = self.sigmoid(
                mu[0] + bu[i] + bi[:] +
                np.sum(p[i, :][np.newaxis, :] * q, axis=1))
            loss = loss - np.sum(
                evi * np.log(esc) + (1 - evi) * np.log(1. - esc))
        loss = loss / n_events

        # regularization term
        reg = (np.sum(bu ** 2) + np.sum(bi ** 2) +
              np.sum(p ** 2) + np.sum(q ** 2))

        return loss + self._reg * reg

    def grad_loss(self, coef, ev, n_objects):
        """
        gradient of loss function

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        grad : array_like, shape=coef.shape
            the first gradient of loss function by coef
        """
        # constants
        n_users = n_objects[0]
        n_events = n_objects[0] * n_objects[1]

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
        for i in xrange(n_users):
            evi = ev.getrow(i).toarray().reshape(-1)
            esc = self.sigmoid(
                mu[0] + bu[i] + bi[:] +
                np.sum(p[i, :][np.newaxis, :] * q, axis=1))
            common_term = esc - evi

            grad_mu[0] += np.sum(common_term)
            grad_bu[i] = np.sum(common_term)
            grad_bi[:] += common_term
            grad_p[i, :] = np.sum(common_term[:, np.newaxis] * q, axis=0)
            grad_q[:, :] += common_term[:, np.newaxis] * p[i, :][np.newaxis, :]

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

        # call super class
        super(EventItemFinder, self).fit(random_state=random_state)

        # get input data
        ev, n_objects = self._get_event_array(
            data, (user_index, item_index), sparse_type='csr')

        # initialize coefficients
        self._init_coef(ev, n_objects)

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
        self.i_loss_ = self.loss(self._coef, ev, n_objects)

        # optimize model
        # fmin_bfgs is slow for large data, maybe because due to the
        # computation cost for the Hessian matrices.
        res = fmin_cg(self.loss,
                      self._coef,
                      fprime=self.grad_loss,
                      args=(ev, n_objects),
                      full_output=True,
                      **kwargs)

        # get parameters
        self._coef[:] = res[0]
        self.f_loss_ = res[1]
        self.opt_outputs_ = res[2:]

        # add parameters for unknown users and items
        self.mu_ = self._coef.view(self._dt)['mu'][0].copy()
        self.bu_ = np.r_[self._coef.view(self._dt)['bu'][0], 0.0]
        self.bi_ = np.r_[self._coef.view(self._dt)['bi'][0], 0.0]
        self.p_ = np.r_[self._coef.view(self._dt)['p'][0],
                        np.zeros((1, self.k), dtype=np.float)]
        self.q_ = np.r_[self._coef.view(self._dt)['q'][0],
                        np.zeros((1, self.k), dtype=np.float)]

        # clean up temporary instance variables
        del self._coef
        del self._reg
        del self._dt
        del self._rng

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
            a target user's and item's ids. unknown objects assumed to be
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

        sc = self.sigmoid(
            self.mu_[0] + self.bu_[ev[:, 0]] + self.bi_[ev[:, 1]] +
            np.sum(self.p_[ev[:, 0], :] * self.q_[ev[:, 1], :], axis=1))

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
