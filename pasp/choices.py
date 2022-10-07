from .program import ProbFact

def pr(phi: list[ProbFact], theta: iter) -> float:
  """
  Returns the probability of the total choice θ asserted through the probabilistic facts ϕ.
  """
  n = len(phi)
  p = 1.0
  for i in range(n):
    q, t = float(phi[i].p), theta[i]
    p *= t*q + (not t)*(1-q) # ϕ[i].p if θ[i] is true, otherwise 1-ϕ[i].p.
  return p
