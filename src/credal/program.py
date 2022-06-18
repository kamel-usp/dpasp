import re

"""
Creates a new unique fact for probabilistic rules. To do this, we update a counter `unique_fact.i`
in a way equivalent to C's `static` variables.
"""
def unique_fact(i: int = None):
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

  def __str__(self) -> str: return f"{self.p}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

"""
A Probabilistic Rule (PR) is a syntactic sugar for constructing a (Logic Program) rule equipped
with a (unique) PF as one of its subgoals. To reflect this, `ProbRule` is actually a function that
returns a rule and a PF.
"""
def ProbRule(p: float, r: str): return r, ProbFact(p, unique_fact())

"""
A Probabilistic Logic Program (PLP) is a tuple `<P,PF>`, where `P` is a logic program and `PF` are
probabilistic facts.

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
  "Constructs a PLP out of a file named `filename`."
  def __init__(self, filename: str):
    self.P, self.PF = parse(filename)

REGEX_PROB_FACT  = re.compile("[0-9]*\.[0-9]*\:\:[a-zA-Z][a-zA-Z0-9]*\.")
REGEX_PROB_RULE  = re.compile("[0-9]*\.[0-9]*\:\:[a-zA-Z0-9]+\([A-Z][a-zA-Z0-9]*\)\s*\:\-.*?\.")
REGEX_PROB_TOKEN = re.compile("\:\:")
REGEX_BEG_WSPACE = re.compile("^\s*", flags = re.MULTILINE)

def parse(filename: str) -> tuple[str, str]:
  # Logic Program
  P = None
  # Probabilistic Facts
  PF = None
  # For now, dump file entirely into memory (this isn't too much of a problem, since PLPs are
  # usually small). In the future, consider streaming batches of text instead for large files.
  data = None
  with open(filename, "r") as f: data = f.read()
  PF = [ProbFact(*REGEX_PROB_TOKEN.split(x)) for x in REGEX_PROB_FACT.findall(data)]
  data = REGEX_PROB_FACT.sub("", data)
  PR = [(*ProbRule(*REGEX_PROB_TOKEN.split(x)), x) for x in REGEX_PROB_RULE.findall(data)]
  # r - logic rule, pf - probabilistic fact, o - original probabilistic rule
  for r, pf, o in PR:
    data = data.replace(o, r)
    PF.append(pf)
  P = REGEX_BEG_WSPACE.sub("", data)
  return P, PF
