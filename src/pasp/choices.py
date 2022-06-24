from clingo.symbol import Function
from .program import ProbFact

"""
Returns the probability of the total choice θ asserted through the probabilistic facts ϕ.
"""
def ℙ(ϕ: list[ProbFact], θ: iter) -> float:
  n = len(ϕ)
  p = 1.0
  for i in range(n):
    q, t = float(ϕ[i].p), θ[i]
    p *= t*q + (not t)*(1-q) # ϕ[i].p if θ[i] is true, otherwise 1-ϕ[i].p.
  return p
