import sys
import itertools
import math
import numpy as np

import clingo
from clingo.symbol import Function
from clingo.control import Control

from .program import Program
from . import choices
from .utils import suppress_err, suppress_ext_err

"""
Runs exact inference in order to answer the queries in `P`.
"""
def exact(P: Program) -> list[tuple[float, float]]:
  # Get all probabilistic facts.
  PF = np.array(P.PF)
  # Get all queries.
  queries = P.Q
  # Query results.
  R = [None for i in range(len(queries))]
  # For each query, run the algorithm due to Calì et al, 2009.
  for i, query in enumerate(queries):
    a, b, c, d = .0, .0, .0, .0
    Q, E = query.Q, query.E
    # We iterate through each total choice θ (itertools.product is written in C, so this loop
    # somewhat efficient.)
    for θ in itertools.product([False, True], repeat = len(PF)):
      # Transform into list to enable fast indexing through numpy.
      θ = list(θ)
      # F are the facts produced by the total choice θ.
      F = [PF[i].cl_f[x] for i, x in enumerate(θ)]
      # Initialize a clingo Control.
      C = Control(logger = lambda x, y: None if x == clingo.MessageCode.AtomUndefined else print(y, file = sys.stderr))
      # Force solver to output all stable models.
      C.configuration.solve.models = 0
      # Input the logic program into the clingo Control.
      C.add("base", [], P.P)
      # Add probabilistic facts according to θ.
      with C.backend() as B:
        for x in F: B.add_rule((B.add_atom(x),))
      # Ground atoms.
      C.ground([("base", [])])
      # m is the number of stable models according to <P,θ>, i.e. m = |Γ(θ)|.
      m = 0
      # Condition 1: if every stable model in Γ(θ) satisfies Q and E.
      cond_1 = False
      # Condition 2: if some stable model in Γ(θ) satisfies Q and E.
      cond_2 = False
      # Condition 3: if every stable model in Γ(θ) satisfies E but fails Q.
      cond_3 = False
      # Condition 4: if some stable model in Γ(θ) satisfies E but fails Q.
      cond_4 = False
      # How many models satisfy Q and E, and how many models satisfy E.
      count_q_e, count_e = 0, 0
      # How many models satisfy E but do not satisfy Q completely.
      count_partial_q_e = 0
      # Count which models satisfy Q and/or E.
      def count_sat(σ: clingo.solving.Model):
        nonlocal m, count_q_e, count_e, count_partial_q_e
        nonlocal cond_2, cond_4
        m += 1
        all_e = all(σ.contains(e) if t else not σ.contains(e) for e, t in E) # if e = true, check if e ∈ σ.
        if not all_e: return
        all_q = all(σ.contains(q) if t else not σ.contains(q) for q, t in Q) # if q = true, check if q ∈ σ.
        count_e += 1
        if all_q: cond_2 = True; count_q_e += 1
        else: cond_4 = True; count_partial_q_e += 1
      # Solve for <P,θ>, running on_model for every stable model σ found.
      C.solve(on_model = count_sat)
      # Evaluate counts to judge whether cond_1 and/or cond_3 are true.
      if count_e == m or len(E) == 0:
        # All stable models satisfy Q and E completely.
        if count_q_e == m: cond_1 = True
        # All stable models satisfy E, but none satisfies Q completely.
        if count_partial_q_e == m: cond_3 = True
      # Add probability ℙ(θ) according to model satisfiabilities.
      p = choices.ℙ(PF, θ)
      a += cond_1*p
      b += cond_2*p
      c += cond_3*p
      d += cond_4*p
    # Evaluate a, b, c, d values and return ℙ(Q|E) as a tuple of lower and upper probabilities.
    #print(a, b, c, d)
    #print(query)
    if len(E) == 0: R[i] = (a, b)
    else:
      if b + d == 0:
        print("Fail: ℙ(E) = 0!")
        R[i] = (-math.inf, math.inf)
      else:
        if b + c == 0 and d > 0: R[i] = (0, 0)
        elif a + d == 0 and b > 0: R[i] = (1, 1)
        else: R[i] = (a/(a+d), b/(b+c))
    print(f"{query} = {R[i]}")
  return R
