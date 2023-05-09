"""
.. include:: ../README.md
"""

from .grammar import parse
from exact import exact, count
from ground import ground
from .program import Program
from sample import sample
from .wlearn import learn

import numpy as np

__version__ = "0.0.3"
