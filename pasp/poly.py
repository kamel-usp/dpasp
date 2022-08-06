###########################################################
# THE FUNCTIONS CONTAINED WITHIN THIS FILE ARE DEPRECATED #
###########################################################

import itertools, math

import numpy as np
from scipy.optimize import brute

from .program import CredalFact

Product = list[bool]
Polynomial = list[Product]
Coefficients = list[float]

def _smp_obj_func(X: np.array, S: np.array, C: np.array) -> float:
  n, X_cmpl = len(C), 1-X
  return sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(n))

def _smp_obj_func_neg(X: np.array, S: np.array, C: np.array) -> float:
  n, X_cmpl = len(C), 1-X
  return -sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(n))

def _obj_func(X: np.array, S: np.array, T: np.array, C: np.array, K: np.array) -> float:
  n, m, X_cmpl = len(C), len(K), 1-X
  x = 0 if len(S) == 0 else sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(n))
  y = 0 if len(T) == 0 else sum(np.prod(np.where(T[i], X, X_cmpl))*K[i] for i in range(m))
  return 0 if x+y == 0 else x/(x+y)

def _obj_func_neg(X: np.array, S: np.array, T: np.array, C: np.array, K: np.array) -> float:
  n, m, X_cmpl = len(C), len(K), 1-X
  x = 0 if len(S) == 0 else sum(np.prod(np.where(S[i], X, X_cmpl))*C[i] for i in range(n))
  y = 0 if len(T) == 0 else sum(np.prod(np.where(T[i], X, X_cmpl))*K[i] for i in range(m))
  return 0 if x+y == 0 else -(x/(x+y))

def minimize_smp(P: Polynomial, K: Coefficients, B: list[tuple[float, float]]) -> float:
  return brute(_smp_obj_func, ranges = B, args = (np.array(P), np.array(K)), Ns = 2, finish = None, \
               full_output = True)[1]

def maximize_smp(P: Polynomial, K: Coefficients, B: list[tuple[float, float]]) -> float:
  return -brute(_smp_obj_func_neg, ranges = B, args = (np.array(P), np.array(K)), Ns = 2, finish = None, \
                full_output = True)[1]

def minimize(P: Polynomial, Q: Polynomial, K: Coefficients, F: Coefficients, B: list[tuple[float, float]]) -> float:
  return brute(_obj_func, ranges = B, args = (np.array(P), np.array(Q), np.array(K), np.array(F)), \
               Ns = 2, finish = None, full_output = True)[1]

def maximize(P: Polynomial, Q: Polynomial, K: Coefficients, F: Coefficients, B: list[tuple[float, float]]) -> float:
  return -brute(_obj_func_neg, ranges = B, args = (np.array(P), np.array(Q), np.array(K), np.array(F)), \
                Ns = 2, finish = None, full_output = True)[1]

def extrema(P: Polynomial, Q: Polynomial, K: Coefficients, F: Coefficients, B: list[CredalFact]) -> tuple[float, float]:
  S1, S2, K1, K2 = np.array(P), np.array(Q), np.array(K), np.array(F)
  _, l, _, _ = brute(_obj_func, ranges = B, args = (S1, K1, S2, K2), Ns = 2, finish = None, \
                     full_output = True)
  _, u, _, _ = brute(_obj_func_neg, ranges = B, args = (S1, K1, S2, K2), Ns = 2, finish = None, \
                     full_output = True)
  return (l, -u)

def print_poly(P: Polynomial, C: Coefficients):
  print(" + ".join("·".join((f"{C[i]}", *(f"x_{j}" if P[i][j] else f"(1-x_{j})" for j in range(len(P[i]))))) for i in range(len(P))))
