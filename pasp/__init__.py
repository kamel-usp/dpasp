"""
.. include:: ../README.md
"""

from .grammar import parse
from exact import exact, count
from ground import ground
from .program import Program
from sample import sample

import numpy as np

def learn(P: Program, D: np.ndarray, A: np.ndarray, niters: int = 30, alg: str = "fixpoint",
          eta: float = 0.001, lstable_sat: bool = True):
  if type(A) is not np.ndarray: atoms = np.array(A, dtype = bytes)
  else: atoms = A if np.issubdtype(A.dtype, bytes) else A.astype(bytes)
  if type(D) is not np.ndarray: data = np.array(D, dtype = np.uint8)
  else: data = D if np.issubdtype(D.dtype, np.uint8) else D.astype(np.uint8)

  obs, obs_counts = np.unique(data, axis = 0, return_counts = True)
  from learn import learn as clearn
  clearn(P, obs, obs_counts, atoms, niters = niters, alg = alg, eta = eta, lstable_sat = lstable_sat)
