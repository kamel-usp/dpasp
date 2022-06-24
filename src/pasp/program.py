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
    clf = clingo.parse_term(f[:-1]) # remove the dot at the end.
    self.cl_f = [Function(clf.name, clf.arguments, not clf.positive), clf]

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

"""
A Probabilistic Rule (PR) is a syntactic sugar for constructing a (Logic Program) rule equipped
with a (unique) PF as one of its subgoals. To reflect this, `ProbRule` is actually a function that
returns a rule and a PF.
"""
def ProbRule(p: float, r: str) -> tuple[str, ProbFact]:
  f = unique_fact()
  return f"{r[:-1]}, {f}", ProbFact(p, f)

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
    self.Q = [(clingo.parse_term((t := REGEX_QUERY_NOT.subn("", q))[0]), t[1] == 0) for q in Q]
    self.E = [(clingo.parse_term((t := REGEX_QUERY_NOT.subn("", e))[0]), t[1] == 0) for e in E]

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

REGEX_PROB_CMNT  = re.compile("\%.*$", flags = re.MULTILINE)
REGEX_PROB_FACT  = re.compile("[0-9]*\.[0-9]*\:\:[a-zA-Z]\w*(?:\((?:[a-z]\w*\s*\,{0,1}\s*)+\)){0,1}\s*\.")
REGEX_PROB_RULE  = re.compile("[0-9]*\.[0-9]*\:\:[a-zA-Z]\w*(?:\([a-zA-Z]\w*\)){0,1}\s*\:\-.*?\.")
REGEX_PROB_QUERY = re.compile("^\#query\(.+\)", flags = re.MULTILINE)
REGEX_PROB_TOKEN = re.compile("\:\:")
REGEX_BEG_WSPACE = re.compile("^\s*", flags = re.MULTILINE)
REGEX_QUERY_COND = re.compile("\s*\|\s*")
REGEX_QUERY_ARGS = re.compile("\s*\;\s*")
REGEX_QUERY_NOT = re.compile("\s*not\s*")

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
    data = data.replace(o, r)
    PF.append(pf)

  # Parse query commands.
  Q = [Query(*(REGEX_QUERY_ARGS.split(y) for y in REGEX_QUERY_COND.split(x[7:-1]))) for x in REGEX_PROB_QUERY.findall(data)]
  data = REGEX_PROB_QUERY.sub("", data)
  P = REGEX_BEG_WSPACE.sub("", data)
  return Program(P, PF, Q)
