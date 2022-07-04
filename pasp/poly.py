import itertools, math

import numpy as np
from scipy.optimize import minimize

from .program import CredalFact

Product = list[bool]
Polynomial = list[Product]
Coefficients = list[float]

def _obj_func(X: np.array, S: np.array, C: np.array):
  X_cmpl = 1-X
  return sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(len(C)))

def _obj_func_neg(X: np.array, S: np.array, C: np.array):
  X_cmpl = 1-X
  return -sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(len(C)))

def extrema(P: Polynomial, K: Coefficients, C: list[CredalFact], method: str = "Nelder-Mead") -> tuple[float, float]:
  B = [(c.l, c.u) for c in C]
  S, C = np.array(P), np.array(K)
  l = minimize(_obj_func, x0 = np.fromiter(((l+u)/2 for l, u in B), dtype = float), \
               args = (S, C), method = method, bounds = B)
  u = minimize(_obj_func_neg, x0 = np.fromiter(((l+u)/2 for l, u in B), dtype = float), \
               args = (S, C), method = method, bounds = B)
  return (l.fun, -u.fun)
