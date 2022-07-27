# cython: c_string_type=unicode, c_string_encoding=utf8

import re

import clingo
from clingo.symbol import Function

"""
Creates a new unique fact for probabilistic rules. To do this, we update a counter `unique_fact.i`
in a way equivalent to C's `static` variables.
"""
def unique_fact(i: int = None) -> str:
  if i is None:
    unique_fact.i += 1
    return f"__unique_id_{unique_fact.i}"
  return f"__unique_id_{i}"
unique_fact.i = 0

"""
A Probabilistic Fact (PF) is a (Logic Program) fact which is "chosen" with some probability.
"""
class ProbFact:
  "Constructs a PF out of a probability `p` and fact `f`."
  def __init__(self, p: str, f: str):
    self.p = float(p)
    self.f = f
    # Construct a clingo.symbol.Function from this fact.
    self.cl_f = clingo.parse_term(f)

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

"""
A Probabilistic Rule (PR) is syntactic sugar for constructing a (Logic Program) rule equipped
with a (unique) PF as one of its subgoals. To reflect this, `ProbRule` is actually a function that
returns a rule and a PF.
"""
def ProbRule(p: float, r: str) -> tuple[str, ProbFact]:
  f = unique_fact()
  return f"{r}, {f}", ProbFact(p, f)

"""
A Credal Fact (CF) consists of a fact `f` attached to a probability interval `[l, u]`, where `l ∈
[0, 1]` is the lowest probability `f` may attain and `u ≥ l ∈ [0, 1]` is the highest.
"""
class CredalFact:
  def __init__(self, l: float, u: float, f: str):
    self.l, self.u = float(l), float(u)
    self.f = f
    self.cl_f = clingo.parse_term(f)

  def __getitem__(self, i: bool) -> float: return self.l if False else self.u
  def __str__(self) -> str: return f"[{self.l}, {self.u}]::{self.f}"
  def __repr__(self) -> str: return self.__str__()

"""
String formats a query tuple `(f, t)`, where `f` is an atom and `t` is whether it should appear
in the program or not.
"""
def _str_query_assignment(f: Function, t: bool) -> str: return str(f) if t else "not " + str(f)

"""
A query is a meta-command within a PLP to signal the solver to produce and output a probabilistic
query. A query follows a modified PASOCS [1] syntax, that is, the query (not necessarily in this
order)

```
#query(q1; ...; qk; not p1; ...; not pm | e1; ...; en; not v1; ...; not vt)
```

of ground atoms `q1, ..., qk`, `p1, ..., pm`, `e1, ..., en`, `v1, ..., vt` is equivalent to asking
the probability

```
P({q1, ..., qk} = true, {p1, ..., pm} = false | {e1, ..., en} = true, {v1, ..., vt} = false).
```

See concrete examples in the `/examples` folder.

[1] - PASOCS: A Parallel Approximate Solver for Probabilistic Logic Programs under the Credal
Semantics. Tuckey et al, 2021. URL: https://arxiv.org/abs/2105.10908.
"""
class Query:
  """
  Constructs a query from query (`Q`) and evidence (`E`) assignments.

  We use the notation `iter` as a type hinting to mean `Q` and `E` are iterables.
  """
  def __init__(self, Q: iter, E: iter = []):
    self.Q = [None for _ in range(len(Q))]
    for i, q in enumerate(Q):
      t, n = REGEX_QUERY_NOT.subn("", q)
      self.Q[i] = (clingo.parse_term(t), n == 0)
    self.E = [None for _ in range(len(E))]
    for i, e in enumerate(E):
      t, n = REGEX_QUERY_NOT.subn("", e)
      self.E[i] = (clingo.parse_term(t), n == 0)

  def __str__(self) -> str:
    qs = f"ℙ({', '.join(_str_query_assignment(q, t) for q, t in self.Q)}"
    if len(self.E) != 0: return qs + f" | {', '.join(_str_query_assignment(e, t) for e, t in self.E)})"
    return qs + ")"
  def __repr__(self) -> str: return self.__str__()

"""
A Probabilistic Logic Program (PLP) usually configures a triple `<P,PF,CF>`, where `P` is a logic
program, `PF` are probabilistic facts and `CF` are credal facts. We extend a PLP into a tuple
`<P,PF,CF,Q>`, where `Q` are the queries to be asked from `P`, `PF` and `CF`.

We accept ProbLog's syntactic sugar for probabilistic rules,

```
p::h(X) :- b1(X), b2(X), ..., bn(X).
```

meaning that if `b1(X), b2(X), ..., bn(X)` is true, `h(X)` is added with probability `p`. This is
equivalent to

```
p::a.
h(X) :- b1(X), b2(X), ..., bn(X), a.
```

where `a` is a unique probabilistic fact added with probability `p`.
"""
class Program:
  """
  Constructs a PLP out of a logic program `P`, probabilistic facts `PF`, credal facts `CF` and
  queries `Q`.
  """
  def __init__(self, P: str, PF: list[ProbFact], Q: list[Query], CF: list[CredalFact]):
    self.P = P
    self.PF = PF
    self.Q = Q
    self.CF = CF

  def __str__(self) -> str:
    return f"<Logic Program:\n{self.P},\nProbabilistic Facts:\n{self.PF},\nCredal Facts:\n{self.CF}\nQueries\n{self.Q}>"
  def __repr__(self) -> str: return self.__str__()

REGEX_QUERY_NOT  = re.compile(r"\s*not\s*")
