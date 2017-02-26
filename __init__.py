"""
Algorithms and utilities for diffeomorphic density matching.
============================================================

A Python library for diffeomorphic density matching, based on
the Fisher-Rao distance function.

Routine listings
----------------
N/A

See also
--------
Documentation guidelines `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

References
----------
N/A

Examples
--------
N/A
"""

from .core import TwoComponentDensityMatching, CompatibleConformalDensityMatching
from .display import *


__version__ = '0.0.1'
