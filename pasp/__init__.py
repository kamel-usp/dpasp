"""
.. include:: ../README.md
"""

from .grammar import parse
from exact import exact, count
from ground import ground
from .program import Program
from sample import sample

import numpy as np

__version__ = "0.0.2-2"

def learn(P: Program, D: np.ndarray, A: np.ndarray = None, niters: int = 30, alg: str = "fixpoint",
          eta: float = 0.001, batch: int = 100, lstable_sat: bool = True):
  if A is None:
    if type(D) is np.ndarray: data = D if np.issubdtype(A.dtype, bytes) else A.astype(bytes)
    else: data = D
    from learn import learn_batch as clearn_batch
    return clearn_batch(P, data, niters = niters, alg = alg, eta = eta, batch = batch, lstable_sat = lstable_sat)

  if type(A) is not np.ndarray: atoms = np.array(A, dtype = bytes)
  else: atoms = A if np.issubdtype(A.dtype, bytes) else A.astype(bytes)
  if type(D) is not np.ndarray: data = np.array(D, dtype = np.uint8)
  else: data = D if np.issubdtype(D.dtype, np.uint8) else D.astype(np.uint8)

  obs, obs_counts = np.unique(data, axis = 0, return_counts = True)
  from learn import learn as clearn
  clearn(P, obs, obs_counts, atoms, niters = niters, alg = alg, eta = eta, lstable_sat = lstable_sat)
