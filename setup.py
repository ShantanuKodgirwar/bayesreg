import os
import sys
import numpy

from distutils.core import setup, Extension

try:
    __doc__ = open('readme.md').read()
except IOError:
    pass

NAME = "bayesian_linear_regression"
VERSION = "0.1"
AUTHOR = "Shantanu Kodgirwar, Michael Habeck"
EMAIL = "shantanu.kodgirwar@uni-jena.de, michael.habeck@uni-jena.de"
DESCRIPTION = __doc__
LICENSE = 'MIT'
REQUIRES = ['numpy', 'scipy', 'matplotlib', 'sklearn']

setup(
    name=NAME,
    packages=[NAME],
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6+',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries']
)
