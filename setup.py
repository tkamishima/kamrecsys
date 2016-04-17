from setuptools import setup, find_packages
from kamrecsys import (__version__, __license__, __author__)

import re
(author, author_email) = \
    re.search('([^\(\)]+)\s+\(\s+(.+)\s+\)', __author__).groups()

with open('readme.md') as in_file:
    long_description = in_file.read()

setup(
    name='kamrecsys',
    version=__version__,
    download_url='https://github.com/tkamishima/kamrecsys/archive/master.zip',
    license=__license__,
    author='Toshihiro Kamishima',
    author_email='http://www.kamishima.net/',
    url='https://github.com/tkamishima/kamrecsys',
    description='KamRecSys: Algorithms for recommender systems',
    long_description=long_description,
    keywords='recommender systems',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'sklearn'],
    test_suite='nose.collector',
    package_data={
        'kamrecsys.datasets':
            ['samples/flixster*',
             'samples/movielens*',
             'samples/pci.*',
             'samples/sushi*']
    },
)
