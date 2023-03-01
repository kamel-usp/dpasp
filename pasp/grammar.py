import pathlib, enum, math, collections.abc
import lark, lark.reconstruct, clingo, numpy
from numpy import ascontiguousarray as contiguous
from .program import ProbFact, Query, ProbRule, Program, CredalFact, unique_fact, Semantics, Data
from .program import AnnotatedDisjunction, NeuralRule, NeuralAD

def is_fact(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a fact."
  return isinstance(x, lark.Tree) and x.data == "fact"
def is_pfact(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a probabilistic fact."
  return isinstance(x, lark.Tree) and x.data == "pfact"
def is_rule(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a rule."
  return isinstance(x, lark.Tree) and x.data == "rule"
def is_prule(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a probabilistic rule."
  return isinstance(x, lark.Tree) and x.data == "prule"
def is_atom(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a(n) (grounded) atom."
  return isinstance(x, lark.Tree) and (x.data == "atom" or x.data == "gratom")
def is_pred(x: lark.Tree) -> bool:
  "Returns whether a node in the AST is a (grounded) predicate."
  return isinstance(x, lark.Tree) and (x.data == "pred" or x.data == "grpred")

def atom_sign(x: lark.Tree) -> bool:
  "Returns whether a(n) (grounded) atom is negative (`False`), positive (`True`), or not an atom (`None`)."
  return x.children[0].type != "NEG" if is_atom(x) else None
def pred_sign(x: lark.Tree) -> bool:
  "Returns whether a (grounded) predicate is negative (`False`), positive (`True`) or not a predicate (`None`)."
  return x.children[0].type != "NEG" if is_pred(x) else None

def is_prob(x: lark.Tree) -> iter:
  "Returns whether a node (in this case a fact or rule) is probabilistic."
  return isinstance(x, lark.Tree) and len(x.children) > 0 and isinstance(x.children[0], lark.Token) and x.children[0].type == "prob"

def facts(x: lark.Tree) -> iter:
  "Returns an iterator containing all facts in the AST."
  return x.find_pred(is_fact)
def pfacts(x: lark.Tree) -> iter:
  "Returns an iterator containing all probabilistic facts in the AST."
  return x.find_pred(is_pfact)
def rules(x: lark.Tree) -> iter:
  "Returns an iterator containing all rules in the AST."
  return x.find_pred(is_rule)
def prules(x: lark.Tree) -> iter:
  "Returns an iterator containing all probabilistic rules in the AST."
  return x.find_pred(is_prule)
def probs(x: lark.Tree) -> iter:
  "Returns an iterator containing all probabilistic facts and rules in the AST."
  return x.find_pred(lambda x: is_prule(x) or is_pfact(x))

def expand_interval(x: lark.Tree) -> tuple[int, int]:
  assert isinstance(x, lark.Tree) and x.data == "interval", "Given AST node is not an Interval."
  return int(x.children[0].value), int(x.children[1].value)

def tree_contains(x: lark.Tree, f) -> bool:
  "Performs a depth-first search, returning (and stopping the search) node `y` when `f(y) == True`."
  V = set()
  def visit(x: lark.Tree | lark.Token) -> bool:
    V.add(x)
    if f(x): return True
    if isinstance(x, lark.Tree):
      for c in x.children:
        if (c not in V) and visit(c): return True
    return False
  return visit(x)

def is_var(x: lark.Token): return isinstance(x, lark.Token) and x.type == "VAR"

def is_nonground(x: lark.Tree) -> bool:
  "Returns whether node `x` is not grounded, i.e. whether any node in the subtree of `x` is a variable."
  try: iter(x)
  except TypeError: return tree_contains(x, is_var)
  else: return any(tree_contains(x, is_var))
def is_ground(x: lark.Tree) -> bool:
  "Returns whether node `x` is grounded, i.e. whether no node in the subtree of `x` is a variable."
  return not is_nonground(x)

def read(*files: str, G: lark.Lark = None, from_str: bool = False) -> lark.Tree:
  "Read all `files` and parse them with grammar `G`, returning a single `lark.Tree`."
  if G is None:
    try:
      with open(pathlib.Path(__file__).resolve().parent.joinpath("grammar.lark"), "r") as f:
        G = lark.Lark(f, start = "plp")
    except Exception as ex:
      raise ex
  T = None
  if from_str: T = G.parse("\n".join(files)); return T
  for fname in files:
    try:
      # For now, dump files entirely into memory (this isn't too much of a problem, since PLPs are
      # usually small). In the future, consider streaming batches of text instead for large files.
      with open(fname, "r") as f:
        if T is None: T = G.parse(f.read())
        else:
          U = G.parse(f.read())
          T.children.extend(u for u in U.children if u not in T.children)
    except Exception as ex:
      raise ex
  assert T is not None, "No file read."
  return T

def getnths(X: iter, i: int) -> iter: return (x[i] for x in X)
def find(X: iter, v, d = None):
  try:
    i = X.index(v)
    return i, X[i]
  except ValueError: return -1, d
def push(L: list, X: iter):
  "Pushes X to L. If X is a list, then push all elements of X to L instead of the list object itself."
  if isinstance(X, list): L.extend(X)
  else: L.append(X)

def lit2atom(x: str) -> str: return x[4:] if x[:4] == "not " else x

class StableTransformer(lark.Transformer):
  def __init__(self, _):
    super().__init__()
    self.semantics = Semantics.STABLE
    self.torch_scope = {}

  @staticmethod
  def pack(t: str, rep: str = None, val = None, scope: dict = {}) -> tuple[str, str, str, dict]:
    class Pack(tuple):
      @staticmethod
      def __new__(cls, tp: str, r: str = None, v = None, sc: dict = {}):
        return super(Pack, cls).__new__(cls, (tp, str(v) if r is None else r, r if v is None else v, sc))
      def __str__(self): return self[1]
      def __repr__(self): return self.str()
    return Pack(t, rep, val, scope)

  @staticmethod
  def join_scope(A: list) -> dict: return dict((y, None) for S in A for y in S[3])

  @staticmethod
  def find_data_pred(D: dict, body: list, which: str, name: str) -> list:
    t = None
    for d in body:
      if d[2][1] in D:
        t = D[d[2][1]]
        break
    if t is None: raise ValueError(f"Neural {which} {name} must contain a data predicate!")
    return t

  @staticmethod
  def check_data(D: list):
    "Checks if all data have same first dimension size."
    n, m = -1, -1
    for X in D.values():
      if n < 0: n = X[0].test.shape[0]
      if (X[0].train is not None) and (m < 0): m = X[0].train.shape[0]
      for x in X:
        if x.test.shape[0] != n:
          raise ValueError("Test data must have same number of instances!")
        if (x.train is not None) and (x.train.shape[0] != m):
          raise ValueError("Train data must have same number of instances!")

  @staticmethod
  def register_nrule(TNR: list, NR: list, D: list):
    for name, inp, net, body, rep, learnable, params in TNR:
      t = StableTransformer.find_data_pred(D, body, "rule", name)
      # Ground rules.
      H = contiguous(tuple(clingo.parse_term(f"{name}({t[i].arg})")._rep for i in range(len(t))), \
                      dtype = numpy.uint64)
      B, S = None, None
      if len(body) > 1:
        body_no_data = [b for b in body if b[2][1] != t[0].name]
        B = contiguous(tuple(clingo.parse_term(f"{b[2][1]}({t[i].arg})" if len(b[3]) > 0 \
                                                else lit2atom(b[1]))._rep for i in range(len(t)) \
                              for b in body_no_data), dtype = numpy.uint64)
        S = contiguous(tuple(b[2][0] for i in range(len(t)) for b in body_no_data), dtype = bool)
      NR.append(NeuralRule(H, B, S, name, net, rep, t, learnable, params))

  @staticmethod
  def register_nad(TNA: list, NA: list, D: list):
    for name, inp, vals, net, body, rep, learnable, params in TNA:
      t = StableTransformer.find_data_pred(D, body, "AD", name)
      # Ground rules.
      V = list(vals.keys())
      H = contiguous(tuple(clingo.parse_term(f"{name}({t[i].arg}, {V[j]})")._rep \
                            for i in range(len(t)) for j in range(len(V))))
      B, S = None, None
      if len(body) > 1:
        body_no_data = [b for b in body if b[2][1] != t[0].name]
        # B and S do not depend on the number of values |V|, only on |t| and |body|.
        B = contiguous(tuple(clingo.parse_term(f"{b[2][1]}({t[i].arg})" if len(b[3]) > 0 \
                                                else lit2atom(b[1]))._rep for i in range(len(t)) \
                              for b in body_no_data), dtype = numpy.uint64)
        S = contiguous(tuple(b[2][0] for i in range(len(t)) for b in body_no_data), dtype = bool)
      NA.append(NeuralAD(H, B, S, name, V, net, rep, t, learnable, params))

  # Components which are directly translated to clingo.
  def CMP_OP(self, o): return self.pack("CMP_OP", str(o))
  def aggr(self, A): return self.pack("aggr", "".join(str(x) for x in A))

  # Terminals.
  def UND(self, u): return self.pack("UND", str(u))
  def WORD(self, c): return self.pack("WORD", str(c))
  def NEG(self, n): return self.pack("NEG", str(n))
  def VAR(self, v):
    x = str(v); X = {v: None}
    return self.pack("VAR", x, scope = X)
  def ID(self, i): return self.pack("ID", val = int(i))
  def OP(self, o): return self.pack("OP", str(o))
  def REAL(self, r): return self.pack("REAL", val = float(r))
  def frac(self, f): return self.pack("frac", val = f[0][2]/f[1][2])
  def prob(self, p): return self.pack("prob", val = p[0][2])
  def EXPAND(self, e): return self.pack("EXPAND",  str(e))
  def LEARN(self, l): return self.pack("LEARN", str(l))
  def CONST(self, c): return self.pack("CONST", str(c))
  def BOOL(self, b): return self.pack("BOOL", v := b.lower(), v != "false")
  def NULL(self, n): return self.pack("NULL", None, None)

  # Path.
  def path(self, p): return self.pack(p[0].type, p[0].value)

  # Set.
  def set(self, S):
    M = dict((x[1], None) for x in S)
    if len(M) != len(S): raise ValueError("set must contain only unique constants!")
    return self.pack("set", f"{{{','.join(x for x in M.keys())}}}", M)

  # Intervals.
  def interval(self, I): return self.pack("interval", f"{I[0][2]}..{I[1][2]}")

  # Predicates.
  def pred(self, P):
    name = P[0][1]
    return self.pack("pred", f"{name}({', '.join(getnths(P[1:], 1))})", name, self.join_scope(P))
  def grpred(self, P): return self.pred(P)

  # Literals.
  def lit(self, P):
    s = P[0][0] != "NEG"
    return self.pack("lit", " ".join(getnths(P, 1)), (s, P[0][2] if s else P[1][2]), self.join_scope(P))
  def grlit(self, P): return self.lit(P)

  # Binary operations.
  def bop(self, B) -> str: return self.pack("bop", " ".join(getnths(B, 1)))

  # Facts.
  def fact(self, F):
    f = f"{''.join(getnths(F, 1))}"
    # Facts are always grounded.
    return self.pack("fact", f + ".", f)
  def pfact(self, PF):
    p, f = PF[0][2], PF[1][1]
    return self.pack("pfact", "", ProbFact(p, f))
  def cfact(self, CF):
    l, u, f = CF[0][2], CF[1][2], CF[2][1]
    return self.pack("cfact", "", CredalFact(l, u, f))
  def lpfact(self, PF):
    if PF[0][0] == "prob": p, f = PF[0][2], PF[1][1]
    else: p, f = 0.5, PF[0][2]
    return self.pack("pfact", "", ProbFact(p, f, learnable = True))

  # Heads.
  def head(self, H): return self.pack("head", ", ".join(getnths(H, 1)), H, self.join_scope(H))
  def ohead(self, H): return self.pack("head", H[0][1], H[0][2], H[0][3])
  # Body.
  def body(self, B): return self.pack("body", ", ".join(getnths(B, 1)), B, self.join_scope(B))

  # Rules.
  def rule(self, R): return self.pack("rule", " :- ".join(getnths(R, 1)) + ".")
  def prule(self, R):
    l = "LEARN" in getnths(R, 0)
    e = "EXPAND" in getnths(R, 0)
    h, b = R[-2], R[-1]
    o = f"{h[1]} :- {b[1]}"
    p = R[0][2]
    if e:
      S = self.join_scope(R)
      if len(S) == 0:
        pr = ProbRule(p, o, is_prop = True)
        return self.pack("prule", pr.prop_f, pr)
      # Invariant: len(b) > 0, otherwise the rule is unsafe.
      name = h[2]
      # hscope is guaranteed to be ordered by Python dict's definition.
      hscope = h[3]
      body_preds = [x for x in b[2] if x[0] != "bop"]
      h_s = ", ".join(hscope) + ", " if len(hscope) > 0 else ""
      b_s = ", ".join(map(lambda x: f"1, {x[1]}" if x[2][0] else f"0, {x[1][4:]}", body_preds))
      # The number of body arguments is twice as we need to store the sugoal's sign and symbol.
      u = f"{name}(@unify(\"{p}\", {name}, {int(l)}, {len(hscope)}, {2*len(body_preds)}, {h_s}{b_s})) :- {b[1]}."
      return self.pack("prule", "", ProbRule(p, o, is_prop = False, unify = u, learnable = l))
    else:
      # TODO: parameter tying
      pr = ProbRule(p, o, is_prop = True, learnable = l)
      return self.pack("prule", pr.prop_f, pr)

  # Annotated disjunction head.
  def ad_head(self, H):
    P, F = [], []
    for i in range(0, len(H), 2):
      P.append(H[i][2])
      F.append(H[i+1][1])
    return self.pack("ad_head", F, P, self.join_scope(H))
  # Learnable annotated disjunction head.
  def lad_head(self, H: list):
    P, F = [], []
    i, o, j = 0, 0, 0
    last = None
    while i < len(H):
      a = H[i]
      if a[0] == "prob":
        P.append(a[2])
        F.append(H[i+1][1])
        i += 2
      else:
        P.append(-1)
        F.append(a[1])
        i += 1; o += 1
        last = j
      j += 1
    if o > 0:
      P_s = sum(P)+o
      # If probs were not explicitly given, assume maximum uncertainty and set to uniform.
      s = round((1.0-P_s)/o, ndigits = 15)
      ts = P_s+s*(o-1)
      for i, p in enumerate(P):
        if i == last: P[i] = 1.0-ts
        elif p < 0: P[i] = s
    return self.pack("lad_head", F, P, self.join_scope(H))
  # Annotated disjunctions.
  def ad(self, AD):
    P, F, learnable = AD[0][2], AD[0][1], AD[0][0] == "lad_head"
    if not math.isclose(s := sum(P), 1.0):
      P.append(1-s)
      F.append(unique_fact())
    return self.pack("ad", "", AnnotatedDisjunction(P, F, learnable), AD[0][3])
  def adr(self, AD):
    raise NotImplementedError

  def _data2tensor(self, D):
    tp, path_or_func = D[0][0], D[0][2]
    import pandas, numpy
    # Is an external file or URL.
    if tp != "PY_FUNC": data = pandas.read_csv(path_or_func, dtype = numpy.float32)
    else: # is a function.
      if path_or_func not in self.torch_scope:
        raise ValueError(f"No data definition {path_or_func} found! Either define it in a PyTorch "
                         "block or specify a file or URL to read from.")
      data = self.torch_scope[path_or_func]()
      try:
        import torch
      except ModuleNotFoundError:
        raise ModuleNotFoundError("PyTorch not found! PyTorch must be installed for neural rules "
                                  "and neural ADs.")
      if not issubclass(type(data), torch.Tensor): data = torch.tensor(data)
    return data

  # Test set special predicate.
  def test(self, T): return self.pack("test", "", self._data2tensor(T))
  # Train set special predicate.
  def train(self, R): return self.pack("train", "", self._data2tensor(R))

  # Data special predicate.
  def data(self, D):
    name, arg = D[0][1], D[1][1]
    test, train = D[2][2], D[3][2] if len(D) > 3 else None
    return self.pack("data", f"{name}({arg}).", Data(name, arg, test, train))

  # Torch block.
  def torch(self, T):
    exec("import torch\n\n" + T[0].value, self.torch_scope)
    return self.pack("torch", "")

  # Local hubconf repo.
  def LOCAL_NET(self, L): return self.pack("LOCAL_NET", str(L))
  # GitHub hubconf repo.
  def GITHUB(self, H): return self.pack("GITHUB", str(H))
  # Python function.
  def PY_FUNC(self, P): return self.pack("PY_FUNC", str(P))

  # Hub network.
  def hub(self, H):
    # Function name or entrypoint.
    func = H[0][1]
    # Network is coming from a Torch block.
    if len(H) == 1:
      if func not in self.torch_scope:
        raise ValueError(f"No network definition {func} found! Either define it in a PyTorch "
                         "block or specify a PyTorch Hub model (local or from GitHub).")
      N = self.torch_scope[func]()
      rep = f"@{func}"
    # Network is coming from PyTorch Hub.
    else:
      try:
        import torch
      except ModuleNotFoundError:
        raise ModuleNotFoundError("PyTorch not found! PyTorch must be installed for neural rules "
                                  "and neural ADs.")
      path, source = H[1][1], "github" if H[1][0] == "GITHUB" else "local"
      N = torch.hub.load(path, func, source = source, trust_repo = "check")
      rep = f"@{func} on \"{path}\" at \"{source}\""
    return self.pack("hub", "", (N, rep))

  # Optimizer parameters.
  def params(self, P):
    return self.pack("params", "", {P[i][1]: P[i+1][2] for i in range(0, len(P), 2)})

  # Neural rule.
  def nrule(self, A):
    learnable = A[0][0] == "LEARN"
    name = A[1][1]
    inp = A[2][1]
    net, hub_repr = A[3][2]
    if A[4][0] == "params":
      params = A[4][2]
      body = A[5:]
    else:
      params = {}
      body = A[4:]
    scope = self.join_scope(A)

    if len(scope) != 1:  raise ValueError(f"Neural rule {name} is not grounded!")
    if inp not in scope: raise ValueError(f"Neural rule {name} is unsafe!")

    rep = f"{A[0][1]}::{name}({inp}) as {hub_repr} :- {', '.join(getnths(body, 1))}."
    return self.pack("nrule", "", (name, inp, net, body, rep, learnable, params))

  # Neural annotated disjunction.
  def nad(self, A):
    learnable = A[0][0] == "LEARN"
    name = A[1][1]
    inp = A[2][1]
    vals = A[3][2]
    net, hub_repr = A[4][2]
    if A[5][0] == "params":
      params = A[5][2]
      body = A[6:]
    else:
      params = {}
      body = A[5:]
    scope = self.join_scope(A)

    if len(scope) != 1:  raise ValueError(f"Neural annotated disjunction {name} is not grounded!")
    if inp not in scope: raise ValueError(f"Neural annotated disjunction {name} is unsafe!")

    rep = f"{A[0][1]}::{name}({inp}, {A[2][1]}) as {hub_repr} :- {', '.join(getnths(body, 1))}."
    return self.pack("nad", "", (name, inp, vals, net, body, rep, learnable, params))

  # Constraint.
  def constraint(self, C): return self.pack("constraint", f":- {C[0][1]}.")

  # Query elements.
  def qelement(self, E): return self.pack("qelement", " ".join(getnths(E, 1)))
  # Interpretations.
  def interp(self, I): return self.pack("interp", "", getnths(I, 1))
  # Queries.
  def query(self, Q):
    return self.pack("query", "", Query(Q[0][2], Q[1][2] if len(Q) > 1 else [], semantics = self.semantics))

  # Constant definition.
  def constdef(self, C): return self.pack("constdef", f"#const {C[0][1]} = {C[1][1]}.")

  # Probabilistic Logic Program.
  def plp(self, C) -> Program:
    # Logic Program.
    P  = []
    # Probabilistic Facts.
    PF = []
    # Probabilistic Rules.
    PR = []
    # Queries.
    Q  = []
    # Credal Facts.
    CF = []
    # Annotated Disjunction.
    AD = []
    # Neural arguments and data.
    TNR, TNA = [], []
    D = {}
    # Actual neural rules and neural ADs.
    NR, NA = [], []
    # Mapping.
    M = {"pfact": PF, "prule": PR, "query": Q, "cfact": CF, "ad": AD, "nrule": TNR, "nad": TNA}
    for t, L, O, _ in C:
      if len(L) > 0: push(P, L)
      if t in M: push(M[t], O)
      if t == "data":
        if O.name in D: D[O.name].append(O)
        else: D[O.name] = [O]
    # Deal with ungrounded probabilistic rules.
    for r in PR:
      if r.is_prop: PF.append(r.prop_pf)
    self.check_data(D)
    self.register_nrule(TNR, NR, D)
    self.register_nad(TNA, NA, D)
    return Program("\n".join(P), PF, PR, Q, CF, AD, NR, NA, semantics = self.semantics)

class PartialTransformer(StableTransformer):
  def __init__(self, sem: str):
    super().__init__(sem)
    self.PT = set()
    self.semantics = Semantics.LSTABLE if sem == "lstable" else Semantics.PARTIAL
    self.o_tree = None

  @staticmethod
  def has_binop(x: str): return ("=" in x) or ("<" in x) or (">" in x)

  def fact(self, F):
    T = super().fact(F)
    self.PT.add(T[2])
    return T

  def pfact(self, PF):
    T = super().pfact(PF)
    self.PT.add(T[2].f)
    return T

  def cfact(self, CF):
    T = super().cfact(CF)
    self.PT.add(T[2].f)
    return T

  def rule(self, R):
    b1 = ", ".join(map(lambda x: x[1] if x[2][0] else f"not _{x[1][4:]}", R[1][2]))
    b2 = ", ".join(map(lambda x: f"_{x[1]}" if x[2][0] or PartialTransformer.has_binop(x) else x[1], R[1][2]))
    h1, h2 = R[0][1], ", ".join(map(lambda x: f"_{x[1]}", R[0][2]))
    for h in R[0][2]: self.PT.add(h[1])
    # for x in r[1][3]:
      # if not PartialTransformer.has_binop(x): self.PT.add(x[4:] if x[:4] == "not " else x)
    return self.pack("rule", [f"{h1} :- {b1}.", f"{h2} :- {b2}."])

  def prule(self, R):
    l = "LEARN" in getnths(R, 0)
    e = "EXPAND" in getnths(R, 0)
    p = R[0][2]
    h, b = R[-2], R[-1]
    tr_negs = lambda x: x[1] if x[2][0] else f"not _{x[1][4:]}"
    tr_pos  = lambda x: f"_{x[1]}" if x[2][0] or PartialTransformer.has_binop(x) else x[1]
    b1 = ", ".join(map(tr_negs, b[2]))
    b2 = ", ".join(map(tr_pos, b[2]))
    o1, o2 = f"{h[1]} :- {b1}", f"_{h[1]} :- {b2}"
    self.PT.add(h[1])
    uid = unique_fact()
    if e:
      S = self.join_scope(R)
      if len(S) == 0:
        pr1, pr2 = ProbRule(p, o1, ufact = uid), ProbRule(p, o2, ufact = uid)
        return self.pack("prule", [pr1.prop_f, pr2.prop_f], [pr1, pr2])
      # Invariant: len(b) > 0, otherwise the rule is unsafe.
      name = h[2]
      hscope = h[3]
      body_preds = [x for x in b[2] if x[0] != "bop"]
      h_s = ", ".join(hscope) + ", " if len(hscope) > 0 else ""
      b1_s = ", ".join(map(lambda x: f"1, {x[1]}" if x[2][0] else f"0, _{x[1][4:]}", body_preds))
      # Let the grounder deal with the _f rule.
      u1 = f"{name}(@unify(\"{p}\", {name}, {int(l)}, {len(hscope)}, {2*len(body_preds)}, {h_s}{b1_s})) :- {b1}."
      return self.pack("prule", "", ProbRule(p, o1, is_prop = False, unify = u1, learnable = l))
    else:
      pr1, pr2 = ProbRule(p, o1, ufact = uid, learnable = l), ProbRule(p, o2, ufact = uid)
      return self.pack("prule", [pr1.prop_f, pr2.prop_f], [pr1, pr2])

  def plp(self, C: list[tuple]) -> Program:
    # Logic Program.
    P  = []
    # Probabilistic Facts.
    PF = []
    # Probabilistic Rules.
    PR = []
    # Queries.
    Q  = []
    # Credal Facts.
    CF = []
    # Annotated Disjunction.
    AD = []
    # Neural rules and ADs.
    NR, NA = [], []
    # Mapping.
    M = {"pfact": PF, "prule": PR, "query": Q, "cfact": CF, "ad": AD}
    for t, L, O, _ in C:
      if len(L) > 0: push(P, L)
      if t in M: push(M[t], O)
      if t == "prule" and isinstance(O, collections.abc.Iterable) and O[0].is_prop:
        PF.append(O[0].prop_pf)
    P.extend(f"_{x} :- {x}." for x in self.PT)
    return Program("\n".join(P), PF, PR, Q, CF, AD, NR, NA, semantics = self.semantics, \
                   stable_p = self.stable_p)

  def transform(self, tree):
    self.o_tree = tree
    self.stable_p = StableTransformer(self.semantics).transform(tree)
    return super().transform(tree)

def parse(*files: str, G: lark.Lark = None, from_str: bool = False, semantics: str = "stable") -> Program:
  """Either parses `streams` as blocks of text containing the PLP when `from_str = True`, or
  interprets `streams` as filenames to be read and parsed into a `Program`."""
  if semantics not in parse.trans_map:
    raise ValueError("semantics not supported (must either be 'stable', 'partial' or 'lstable')!")
  return parse.trans_map[semantics](semantics).transform(read(*files, G = G, from_str = from_str))
parse.trans_map = {}
parse.trans_map["stable"] = StableTransformer
parse.trans_map["lstable"] = PartialTransformer
parse.trans_map["partial"] = PartialTransformer

