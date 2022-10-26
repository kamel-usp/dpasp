# cython: c_string_type=unicode, c_string_encoding=utf8

import enum

import clingo
from clingo.symbol import Function

def unique_fact(i: int = None) -> str:
  """
  Creates a new unique fact for probabilistic rules. To do this, we update a counter `unique_fact.i`
  in a way equivalent to C's `static` variables.
  """
  if i is None:
    unique_fact.i += 1
    return f"__unique_id_{unique_fact.i}"
  return f"__unique_id_{i}"
unique_fact.i = 0

class ProbFact:
  """
  A Probabilistic Fact (PF) is a (Logic Program) fact which is "chosen" with some probability.
  """

  def __init__(self, p: str, f: str):
    "Constructs a PF out of a probability `p` and fact `f`."
    self.p = float(p)
    self.f = f
    # Construct a clingo.symbol.Function from this fact.
    self.cl_f = clingo.parse_term(f)

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

class ProbRule:
  """
  A Probabilistic Rule (PR) is a (Logic Program) rule that (when propositional) may be chosen with
  some probability `p`. A non-propositional PR must be grounded first.
  """

  def __init__(self, p: str, f: str, is_prop: bool = True, unify: str = None, ufact: str = None, \
               partial: str = None):
    self.p = p
    self.f = f
    self.is_prop = is_prop
    self.unify = unify
    self.prop_pf = ProbFact(p, unique_fact() if ufact is None else ufact)
    self.prop_f = f"{f}, {self.prop_pf.f}."
    #self.partial = f"{partial}, {self.prop_pf.f}."

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

class CredalFact:
  """
  A Credal Fact (CF) consists of a fact `f` attached to a probability interval `[l, u]`, where `l ∈
  [0, 1]` is the lowest probability `f` may attain and `u ≥ l ∈ [0, 1]` is the highest.
  """

  def __init__(self, l: float, u: float, f: str):
    self.l, self.u = float(l), float(u)
    self.f = f
    self.cl_f = clingo.parse_term(f)

  def __getitem__(self, i: bool) -> float: return self.l if False else self.u
  def __str__(self) -> str: return f"[{self.l}, {self.u}]::{self.f}"
  def __repr__(self) -> str: return self.__str__()

def _str_query_assignment(f: Function, t: bool) -> str:
  """
  String formats a query tuple `(f, t)`, where `f` is an atom and `t` is whether it should appear
  in the program or not.
  """
  return str(f) if t == Query.TERM_POS else ("not " + str(f) if t == Query.TERM_NEG else "undef " + str(f))

class AnnotatedDisjunction:
  def __init__(self, P: list[float], F: list[str]):
    self.P = P
    self.F = F
    self.cl_F = [clingo.parse_term(f) for f in F]

  def __getitem__(self, i: int) -> tuple[float, str]:
    return self.P[i], self.F[i]
  def __str__(self) -> str:
    return "; ".join([f"{self.P[i]}::{self.F[i]}" for i in range(len(self.P))])
  def __repr__(self) -> str: return self.__str__()

class Semantics(enum.IntEnum):
  STABLE = 0
  PARTIAL = 1
  LSTABLE = 2

class Query:
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

  TERM_NEG = 0
  TERM_POS = 1
  TERM_UND = 2

  def __init__(self, Q: iter, E: iter = [], semantics: Semantics = Semantics.STABLE):
    """
    Constructs a query from query (`Q`) and evidence (`E`) assignments.

    We use the notation `iter` as a type hinting to mean `Q` and `E` are iterables.
    """
    self.Q = [Query.parse_term(q, semantics) for q in Q]
    self.E = [Query.parse_term(e, semantics) for e in E]

  @staticmethod
  def parse_term(u: str, s: Semantics):
    if u.startswith("not "): t, n = u[4:], Query.TERM_NEG
    elif u.startswith("undef "): t, n = u[6:], Query.TERM_UND
    else: t, n = u, Query.TERM_POS
    return clingo.parse_term(t), n, None if s == Semantics.STABLE else clingo.parse_term(f"_{t}")

  def __str__(self) -> str:
    qs = f"ℙ({', '.join(_str_query_assignment(q, t) for q, t, _ in self.Q)}"
    if len(self.E) != 0: return qs + f" | {', '.join(_str_query_assignment(e, t) for e, t, _ in self.E)})"
    return qs + ")"
  def __repr__(self) -> str: return self.__str__()

class Program:
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

  def __init__(self, P: str, PF: list[ProbFact], PR: list[ProbRule], Q: list[Query], \
               CF: list[CredalFact], AD: list[AnnotatedDisjunction], \
               semantics: Semantics = Semantics.STABLE, stable_p = None):
    """
    Constructs a PLP out of a logic program `P`, probabilistic facts `PF`, credal facts `CF` and
    queries `Q`.
    """
    self.P = P
    self.PF = PF
    self.PR = PR
    self.Q = Q
    self.CF = CF
    self.AD = AD

    self.gr_P = None
    self.gr_PF = None
    self.gr_pr = None

    self.semantics = semantics
    self.stable = stable_p

  def __str__(self) -> str:
    return f"<Logic Program:\n{self.P},\nProbabilistic Facts:\n{self.PF},\nCredal Facts:\n{self.CF},\nAnnotated Disjunctions:\n{self.AD},\nProbabilistic Rules:\n{self.PR},\nQueries\n{self.Q}>"
  def __repr__(self) -> str: return self.__str__()
