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
    return f"__unique_id_{unique_fact.i}."
  return f"__unique_id_{i}."
unique_fact.i = 0

"""
A Probabilistic Fact (PF) is a (Logic Program) fact which is "chosen" with some probability.
"""
class ProbFact:
  "Constructs a PF out of a probability `p` and fact `f`."
  def __init__(self, p: str, f: str):
    self.p = p
    self.f = f
    # Construct a clingo.symbol.Function from this fact.
    self.cl_f = clingo.parse_term(f[:-1]) # remove the dot at the end.
    #self.cl_f = [Function(clf.name, clf.arguments, not clf.positive), clf]

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

"""
A Probabilistic Rule (PR) is syntactic sugar for constructing a (Logic Program) rule equipped
with a (unique) PF as one of its subgoals. To reflect this, `ProbRule` is actually a function that
returns a rule and a PF.
"""
def ProbRule(p: float, r: str) -> tuple[str, ProbFact]:
  f = unique_fact()
  return f"{r[:-1]}, {f}", ProbFact(p, f)

"""
A Credal Fact (CF) is syntactic sugar for constructing four (Logic Program) rules, two of them
being part of a probabilistic rule. Suppose we have a CF `f` whose lower and upper probabilities
are 0.25 and 0.75 respectively. We syntactically write this as:

```
[0.25, 0.75]::f.
```

This is equivalent to the following rules:

```
u :- not v. v :- not u.
0.25::f :- u. 0.75::f :- v.
```

Facts `u` and `v` encode the lower and upper cases respectively of `f`. The first two rules state
that both `u` and `v` must never occur simultaneously (i.e., they are mutually exclusive). The
following two (probabilistic) rules set the probabilities of `f` according to the choice of `u` and
`v`.
"""
def CredalFact(p: float, q: float, f: str) -> tuple[str, str, ProbFact, str, ProbFact]:
  u, v = unique_fact()[:-1], unique_fact()[:-1]
  return f"{u} :- not {v}. {v} :- not {u}.", \
         *ProbRule(p, f"{f} :- {u}."), *ProbRule(q, f"{f} :- {v}.")

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
A Probabilistic Logic Program (PLP) usually configures a tuple `<P,PF>`, where `P` is a logic
program and `PF` are probabilistic facts. We extend a PLP into a triple `<P,PF,Q>`, where `Q` are
the queries to be asked from `P` and `PF`.

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
  "Constructs a PLP out of a logic program `P`, probabilistic facts `PF` and queries `Q`."
  def __init__(self, P: str, PF: list[ProbFact], Q: list[Query]):
    self.P = P
    self.PF = PF
    self.Q = Q

  def __str__(self) -> str: return f"<{self.P[:-1]},\n{self.PF},\n{self.Q}>"
  def __repr__(self) -> str: return self.__str__()

REGEX_PROB_CMNT  = re.compile(r"\%.*$", flags = re.MULTILINE)
REGEX_PROB_FACT  = re.compile(r"\d+(?:\.\d*)?\:\:[a-zA-Z]\w*(?:\((?:[a-z]\w*\s*\,?\s*)+\))?\s*\.")
REGEX_PROB_RULE  = re.compile(r"\d+(?:\.\d*)?\:\:[a-zA-Z]\w*(?:\([a-zA-Z]\w*\))?\s*\:\-.*?\.")
REGEX_PROB_QUERY = re.compile(r"^\#query\(.+\)", flags = re.MULTILINE)
REGEX_PROB_TOKEN = re.compile(r"\:\:")
REGEX_CRED_PROBS = re.compile(r"\[\s*(\d+(?:\.\d*)?)\s*\,\s*(\d+(?:\.\d*)?)\s*\]")
REGEX_CRED_FACT  = re.compile(r"\[\s*\d+(?:\.\d*)?\s*\,\s*\d+(?:\.\d*)?\s*\]\:\:[a-zA-Z]\w*(?:\((?:[a-z]\w*\s*\,?\s*)+\))?\s*\.")
REGEX_BEG_WSPACE = re.compile(r"^\s*", flags = re.MULTILINE)
REGEX_QUERY_COND = re.compile(r"\s*\|\s*")
REGEX_QUERY_ARGS = re.compile(r"\s*\;\s*")
REGEX_QUERY_NOT  = re.compile(r"\s*not\s*")

def parse(filename: str) -> Program:
  # Logic Program
  P = None
  # Probabilistic Facts
  PF = None
  # For now, dump file entirely into memory (this isn't too much of a problem, since PLPs are
  # usually small). In the future, consider streaming batches of text instead for large files.
  data = None
  try:
    with open(filename, "r") as f: data = f.read()
  except Exception as ex:
    raise ex

  # Remove comments
  data = REGEX_PROB_CMNT.sub("", data)

  # Parse probabilistic facts and probabilistic rules.
  PF = [ProbFact(*REGEX_PROB_TOKEN.split(x)) for x in REGEX_PROB_FACT.findall(data)]
  data = REGEX_PROB_FACT.sub("", data)
  PR = [(*ProbRule(*REGEX_PROB_TOKEN.split(x)), x) for x in REGEX_PROB_RULE.findall(data)]
  # r - logic rule, pf - probabilistic fact, o - original probabilistic rule
  for r, pf, o in PR:
    data = data.replace(o, f"{r} % {o}")
    PF.append(pf)

  # Parse credal facts.
  CF = [(*CredalFact(*REGEX_CRED_PROBS.match(x).groups(), REGEX_PROB_TOKEN.split(x)[1][:-1]), x) for x in REGEX_CRED_FACT.findall(data)]
  for xor_rules, lower_fact, lower_pf, upper_fact, upper_pf, o in CF:
    data = data.replace(o, f"{xor_rules} {lower_fact} {upper_fact} % {o}")
    PF.extend((lower_pf, upper_pf))

  # Parse query commands.
  Q = [Query(*(REGEX_QUERY_ARGS.split(y) for y in REGEX_QUERY_COND.split(x[7:-1]))) for x in REGEX_PROB_QUERY.findall(data)]
  data = REGEX_PROB_QUERY.sub("", data)
  P = REGEX_BEG_WSPACE.sub("", data)
  return Program(P, PF, Q)
