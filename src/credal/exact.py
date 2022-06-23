import sys
import itertools
import math
import numpy as np

import clingo
from clingo.symbol import Function
from clingo.control import Control

from .program import Program
from . import choices
from .utils import new_list

"""
Runs exact inference in order to answer the queries in `P`.
"""
def exact(P: Program) -> list[tuple[float, float]]:
  # Get all probabilistic facts.
  PF = np.array(P.PF)
  # Get all queries.
  queries = P.Q
  # Query results.
  n_queries = len(queries)
  R = new_list(n_queries, None)

  # Condition 1: if every stable model in Γ(θ) satisfies Q and E.
  cond_1 = new_list(n_queries, False)
  # Condition 2: if some stable model in Γ(θ) satisfies Q and E.
  cond_2 = new_list(n_queries, False)
  # Condition 3: if every stable model in Γ(θ) satisfies E but fails Q.
  cond_3 = new_list(n_queries, False)
  # Condition 4: if some stable model in Γ(θ) satisfies E but fails Q.
  cond_4 = new_list(n_queries, False)
  # How many models satisfy Q and E, and how many models satisfy E.
  count_q_e, count_e = new_list(n_queries, 0), new_list(n_queries, 0)
  # How many models satisfy E but do not satisfy Q completely.
  count_partial_q_e = new_list(n_queries, 0)
  # Model counts
  a, b = new_list(n_queries, .0), new_list(n_queries, .0)
  c, d = new_list(n_queries, .0), new_list(n_queries, .0)

  # We iterate through each total choice θ (itertools.product is written in C, so this loop
  # somewhat efficient.)
  for θ in itertools.product([False, True], repeat = len(PF)):
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
    # Zero-initialize counters.
    for i in range(n_queries):
      cond_1[i] = cond_2[i] = cond_3[i] = cond_4[i] = False
      count_q_e[i] = count_e[i] = count_partial_q_e[i] = 0
    # Count which models satisfy Q and/or E.
    def count_sat(σ: clingo.solving.Model):
      nonlocal m
      m += 1
      for i, query in enumerate(queries):
        Q, E = query.Q, query.E
        all_e = all(σ.contains(e) if t else not σ.contains(e) for e, t in E) # if e = true, check if e ∈ σ.
        if not all_e: continue
        all_q = all(σ.contains(q) if t else not σ.contains(q) for q, t in Q) # if q = true, check if q ∈ σ.
        count_e[i] += 1
        if all_q: cond_2[i] = True; count_q_e[i] += 1
        else: cond_4[i] = True; count_partial_q_e[i] += 1
    # Solve for <P,θ>, running on_model for every stable model σ found.
    C.solve(on_model = count_sat)
    p = choices.ℙ(PF, θ)
    for i in range(n_queries):
      # Evaluate counts to judge whether cond_1 and/or cond_3 are true.
      if count_e[i] == m or len(queries[i].E) == 0:
        # All stable models satisfy Q and E completely.
        if count_q_e[i] == m: cond_1[i] = True
        # All stable models satisfy E, but none satisfies Q completely.
        if count_partial_q_e[i] == m: cond_3[i] = True
      # Add probability ℙ(θ) according to model satisfiabilities.
      a[i] += cond_1[i]*p
      b[i] += cond_2[i]*p
      c[i] += cond_3[i]*p
      d[i] += cond_4[i]*p
  for i in range(n_queries):
    # Evaluate a, b, c, d values and return ℙ(Q|E) as a tuple of lower and upper probabilities.
    _a, _b, _c, _d = a[i], b[i], c[i], d[i]
    if len(queries[i].E) == 0: R[i] = (_a, _b)
    else:
      if _b + _d == 0:
        print("Fail: ℙ(E) = 0!")
        R[i] = (-math.inf, math.inf)
      else:
        if _b + _c == 0 and _d > 0: R[i] = (0, 0)
        elif _a + _d == 0 and _b > 0: R[i] = (1, 1)
        else: R[i] = (_a/(_a+_d), _b/(_b+_c))
    print(f"{queries[i]} = {R[i]}")
  return R
