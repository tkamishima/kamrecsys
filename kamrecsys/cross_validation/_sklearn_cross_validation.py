"""
originally copied from sklearn.cross_validation.KFold
commit 877d4711bde70df121c6dffa9dc46a110836721f


The :mod:`sklearn.cross_validation` module includes utilities for cross-
validation and performance evaluation.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_random_state
from sklearn.externals.six import with_metaclass


__all__ = ['KFold']


class _PartitionIterator(with_metaclass(ABCMeta)):
    """Base class for CV iterators where train_mask = ~test_mask

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.

    Parameters
    ----------
    n : int
        Total number of elements in dataset.
    """

    def __init__(self, n, indices=None):
        if indices is None:
            indices = True
        else:
            warnings.warn("The indices parameter is deprecated and will be "
                          "removed (assumed True) in 0.17", DeprecationWarning,
                          stacklevel=1)
        if abs(n - int(n)) >= np.finfo('f').eps:
            raise ValueError("n must be an integer")
        self.n = int(n)
        self._indices = indices

    @property
    def indices(self):
        warnings.warn("The indices attribute is deprecated and will be "
                      "removed (assumed True) in 0.17", DeprecationWarning,
                      stacklevel=1)
        return self._indices

    def __iter__(self):
        indices = self._indices
        if indices:
            ind = np.arange(self.n)
        for test_index in self._iter_test_masks():
            train_index = np.logical_not(test_index)
            if indices:
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices()
        """
        for test_index in self._iter_test_indices():
            test_mask = self._empty_mask()
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def _empty_mask(self):
        return np.zeros(self.n, dtype=np.bool)


class _BaseKFold(with_metaclass(ABCMeta, _PartitionIterator)):
    """Base class to validate KFold approaches"""

    @abstractmethod
    def __init__(self, n, n_folds, indices, shuffle, random_state):
        super(_BaseKFold, self).__init__(n, indices)

        if abs(n_folds - int(n_folds)) >= np.finfo('f').eps:
            raise ValueError("n_folds must be an integer")
        self.n_folds = n_folds = int(n_folds)

        if n_folds <= 1:
            raise ValueError(
                "k-fold cross validation requires at least one"
                " train / test split by setting n_folds=2 or more,"
                " got n_folds={0}.".format(n_folds))
        if n_folds > self.n:
            raise ValueError(
                ("Cannot have number of folds n_folds={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))
        self.shuffle = shuffle
        self.random_state = random_state


class KFold(_BaseKFold):
    """K-Folds cross validation iterator.

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds (without shuffling). In a interlaced mode,
    each fold consists of interlaced data, instead of consecutive ones.

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    n : int
        Total number of elements.

    n_folds : int, default=3
        Number of folds. Must be at least 2.

    interlace : boolean, optional, default=False
        Interlaced mode.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : None, int or RandomState
        Pseudo-random number generator state used for random
        sampling. If None, use default numpy RNG for shuffling

    Examples
    --------
    >>> from kamrecsys import cross_validation
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = cross_validation.KFold(4, n_folds=2, interlace=True)
    >>> for train_index, test_index in kf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]

    Notes
    -----
    The first n % n_folds folds have size n // n_folds + 1, other folds have
    size n // n_folds.

    See also
    --------
    StratifiedKFold: take label information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).
    """

    def __init__(self, n, n_folds=3, indices=None,
                 interlace=False, shuffle=False, random_state=None):
        super(KFold, self).__init__(n, n_folds, indices, shuffle, random_state)
        self.interlace = interlace
        self.idxs = np.arange(n)
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        n = self.n
        n_folds = self.n_folds
        if self.interlace:
            for fold in range(n_folds):
                yield self.idxs[fold:n:n_folds]
        else:
            fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
            fold_sizes[:n % n_folds] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                yield self.idxs[start:stop]
                current = stop

    def __repr__(self):
        return ('%s.%s(n=%i, n_folds=%i, interlace=%s, shuffle=%s, '
                'random_state=%s)' % (
                    self.__class__.__module__,
                    self.__class__.__name__,
                    self.n,
                    self.n_folds,
                    self.interlace,
                    self.shuffle,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_folds
