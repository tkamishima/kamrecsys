KamRecSys
=========

`KamRecSys` is a python package of algorithms for recommender systems.

This package targets experimental algorithm used for research purpose.
We concentrate on the flexibility of data structure or easiness for implementing new algorithm rather than scalablity or efficiency of algorithms.

Requirements
------------

We tested this module under the following packages:

* Python 2.7.x (Python 3.x's are not supported)
* NumPy
* SciPy
* Scikit-learn
* numexpr

Install
-------

First, generate sample data sets that you need. Read a `readme.md` file in a `data` directory.
You then build and install by using a `setup.py` script.

Algorithms
----------

* Matrix Factorization

    * Probabilistic Matrix Factorization

* Topic Model

    * Probabilistic Latent Semantic Analysis (Multinomial)

DataSets
--------

* Programming the Collective Intelligence by Toby Segaran

    * sample_movies

* [Movielens Data Sets](http://www.grouplens.org/node/73)

    * Movielens 100k Data Set
    * Movielens 1m Data Set

* [SUSHI Data Sets](http://www.kamishima.net/sushi/)

    * Sushi3b Score Data Set

* [Flixster Data Sets](http://www.sfu.ca/~sja25/datasets/)

    * Flixster Rating Data Set
