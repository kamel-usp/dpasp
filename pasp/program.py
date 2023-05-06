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

def unique_pgrule_id(gen: bool = True):
  if gen:
    unique_pgrule_id.i += 1
    return unique_pgrule_id.i
  return unique_pgrule_id.i
unique_pgrule_id.i = -1

class ProbFact:
  """
  A Probabilistic Fact (PF) is a (Logic Program) fact which is "chosen" with some probability.
  """

  def __init__(self, p: str, f: str, learnable: bool = False):
    "Constructs a PF out of a probability `p` and fact `f`."
    self.p = float(p)
    self.f = f
    # Construct a clingo.symbol.Function from this fact.
    self.cl_f = clingo.parse_term(f)
    self.learnable = learnable

  def __str__(self) -> str: return f"{round(self.p, ndigits = 3)}{'?' if self.learnable else ''}::{self.f}"
  def __repr__(self) -> str: return self.__str__()

class ProbRule:
  """
  A Probabilistic Rule (PR) is a (Logic Program) rule that (when propositional) may be chosen with
  some probability `p`. A non-propositional PR must be grounded first.
  """

  def __init__(self, p: str, f: str, is_prop: bool = True, unify: str = None, ufact: str = None,
               learnable: bool = False, sharing: bool = False):
    self.p = p
    self.f = f
    self.is_prop = is_prop
    self.learnable = learnable
    self.unify = unify
    self.sharing = sharing # sharing parameter i.e. parameter tying.
    self.prop_pf = ProbFact(p, unique_fact() if ufact is None else ufact,
                            learnable = learnable and sharing)
    self.prop_f = f"{f}, {self.prop_pf.f}."
    self.pf_ids = None

  def __str__(self) -> str:
    return f"{self.p}{('' if (self.sharing or self.is_prop) else '+') + ('?' if self.learnable else '')}::{self.f}"
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
  def __init__(self, P: list[float], F: list[str], learnable: bool = False):
    self.P = P
    self.F = F
    self.cl_F = [clingo.parse_term(f) for f in F]
    self.learnable = learnable

  def __getitem__(self, i: int) -> tuple[float, str]:
    return self.P[i], self.F[i]
  def __str__(self) -> str:
    return "; ".join([f"{(p if (p := round(self.P[i], ndigits = 3)) > 0 else '*')}{'?' if self.learnable else ''}::{self.F[i]}" for i in range(len(self.P))])
  def __repr__(self) -> str: return self.__str__()

try:
  import torch
except ModuleNotFoundError:
  print("PyTorch not found! PyTorch must be installed for neural rules and neural ADs to be used "
        "in programs.")

class Data:
  def __init__(self, name: str, arg: str, test, train = None):
    self.name = name
    self.arg = arg
    import pandas
    if issubclass(type(test), pandas.DataFrame): self.test = torch.tensor(test.to_numpy())
    else: self.test = test
    if issubclass(type(train), pandas.DataFrame): self.train = torch.tensor(train.to_numpy())
    else: self.train = train

    if self.train is not None:
      assert self.test.shape[1:] == self.train.shape[1:], \
        "Train and test sets must have same shape (excluding the first dimension)!"

  def __str__(self):
    if self.train is None: return f"{self.name}({self.arg}) ~ test({self.test.shape})"
    else: return f"{self.name}({self.arg}) ~ test({self.test.shape}), train({self.train.shape})"
  def __repr__(self): return self.__str__()

class Neural:
  def __init__(self, net, data: Data, learnable: bool, rep: str, nvals: int,
               opt_params: dict, outcomes: list):
    self.net = net
    self.learnable = learnable
    self.rep = rep
    self.data = data
    self.outcomes = 1 if outcomes is None else len(outcomes)
    self.nvals = nvals

    # Update default opt_params with given params.
    _opt_params = {"lr": 1., "maximize": True}
    _opt_params.update(opt_params)
    self.opt = torch.optim.SGD(net.parameters(), **_opt_params)

    self.test = torch.cat(tuple(d.test for d in data), dim = 0)
    self.train = None if data[0].train is None else torch.cat(tuple(d.train for d in data), dim = 0)
    self.out = None
    # Derivatives of the logic program to be passed to backwards.
    self.dw = torch.zeros(self.out_shape()) if learnable else None

  def __str__(self): return self.rep
  def __repr__(self): return self.__str__()

  def set_train(self): self.net.train()
  def set_eval(self): self.net.eval()

  def out_shape(self) -> tuple:
    "The output tensor shape."
    raise NotImplementedError("Neural components must override this method accordingly!")

  def pr(self):
    "Retrieves the probabilities of the neural rule from the test set."
    with torch.inference_mode():
      return self.net(self.test).cpu().numpy()

  def forward(self, start: int = 0, end: int = None):
    "Retrieves the probabilities of the neural rule from the train set."
    if self.learnable:
      self.out = self.net(self.train[start:end])
      return self.out.data.cpu().numpy()
    with torch.inference_mode():
      return self.net(self.train[start:end]).data.cpu().numpy()

  def backward(self):
    """ Performs backpropagation and runs the optimizer step.
    Argument `dl` is the derivative of the program as a `numpy.ndarray`.
    """
    self.out.backward(self.dw[:len(self.out)])
    self.opt.step()
    self.opt.zero_grad()

  def ntest(self): return self.data[0].test.shape[0]
  def ntrain(self): return self.data[0].train.shape[0] if self.learnable else 0

class NeuralRule(Neural):
  def __init__(self, heads: list, bodies: list, signs: list, name: str, net, rep: str, data: list,
               learnable: bool, params: dict, outcomes: list):
    super().__init__(net, data, learnable, rep, 1, params, outcomes)
    # Heads and bodies must be numpy.uint64 values representing _rep, not Symbols.
    self.H = heads
    self.B = bodies
    self.S = signs
    self.name = name

    # Validate net during parsing so that it won't blow up in our faces during inference or learning.
    p = self.pr()
    assert p.ndim == 2, \
           "Networks embedded onto neural rules must output a single probability!"

  def out_shape(self) -> tuple:
    return (self.train.shape[0], self.outcomes)

class NeuralAD(Neural):
  def __init__(self, heads: list, bodies: list, signs: list, name: str, vals: list, net, rep: str, \
               data: list, learnable: bool, params: dict, outcomes: list):
    super().__init__(net, data, learnable, rep, len(vals), params, outcomes)
    self.H = heads
    self.B = bodies
    self.S = signs
    self.name  = name
    self.vals  = vals

    # Validate net during parsing so that it won't blow up in our faces during inference or learning.
    p = self.pr()
    assert p.ndim == 2, \
           "Networks embedded onto neural rules must output a 1D probability tensor!"

  def out_shape(self):
    return (self.train.shape[0]*self.outcomes, self.nvals)

class Semantics(enum.IntEnum):
  STABLE = 0
  PARTIAL = 1
  LSTABLE = 2
  SMPROBLOG = 3

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
               CF: list[CredalFact], AD: list[AnnotatedDisjunction], NR: list[NeuralRule], \
               NA: list[NeuralAD], semantics: Semantics = Semantics.STABLE, stable_p = None, \
               directives: list = None):
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
    self.NR = NR
    self.NA = NA

    # Number of instances in data.
    self.m_test = 0
    self.m_train = 0
    if (len(NR) > 0) or (len(NA) > 0):
      self.m_test  = NR[0].ntest() if len(NR) > 0 else NA[0].ntest()
      for nr in (NR + NA):
        if nr.learnable:
          self.m_train = nr.ntrain()
          break

    self.gr_P = ""

    self.semantics = semantics
    self.stable = stable_p

    self.directives = directives

  def train(self):
    for N in self.NR:
      if N.learnable: N.set_train()
    for N in self.NA:
      if N.learnable: N.set_train()

  def eval(self):
    for N in self.NR:
      if N.learnable: N.set_eval()
    for N in self.NA:
      if N.learnable: N.set_eval()

  @staticmethod
  def str_if_contains(s: str, L):
    return f"\n{s}:\n{L}," if len(L) > 0 else ""

  def __str__(self) -> str:
    return f"<Logic Program:\n{self.P}," + \
           self.str_if_contains("Probabilistic Facts", self.PF) + \
           self.str_if_contains("Credal Facts", self.CF) + \
           self.str_if_contains("Annotated Disjunctions", self.AD) + \
           self.str_if_contains("Probabilistic Rules", self.PR) + \
           self.str_if_contains("Neural Rules", self.NR) + \
           self.str_if_contains("Neural Annotated Disjunctions", self.NA) + \
           f"\nQueries:\n{self.Q}>"
  def __repr__(self) -> str: return self.__str__()

  def __call__(self, **kwargs):
    if self.directives is not None:
      if "learn" in self.directives:
        f, A = self.directives["learn"]
        D = f()
        from .wlearn import learn
        if isinstance(D, tuple): learn(self, *D, **A)
        else: learn(self, D, **A)
    if len(self.Q) > 0:
      from exact import exact
      A = {"quiet": False}
      A.update(kwargs)
      if "psemantics" in self.directives: A.update(self.directives["psemantics"])
      return exact(self, **A)
