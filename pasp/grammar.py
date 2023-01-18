import pathlib, enum, fractions, math, collections.abc
import lark, lark.reconstruct
from .program import ProbFact, Query, ProbRule, Program, CredalFact, unique_fact, Semantics
from .program import AnnotatedDisjunction

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
  return isinstance(x, lark.Tree) and len(x.children) > 0 and isinstance(x.children[0], lark.Token) and x.children[0].type == "PROB"

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
  if isinstance(X, collections.abc.Iterable) and (type(X) != str): L.extend(X)
  else: L.append(X)

class Command(enum.Enum):
  FACT = 0
  PROB_FACT = 1
  RULE = 2
  PROB_RULE = 3
  QUERY = 4
  CRED_FACT = 5
  CONSTRAINT = 6
  CONSTDEF = 7
  ANNOTATED_DISJUNCTION = 8
  NEURAL_RULE = 11
  NEURAL_AD = 12

class Terms(enum.Enum):
  UND    = "undef"
  NEG    = "not"
  ADD    = "+"
  SUB    = "-"
  DIV    = "/"
  MOD    = "\\"
  MUL    = "*"
  NEQ    = "!="
  EQQ    = "="
  LES    = "<"
  GRT    = ">"
  LEQ    = "<="
  GEQ    = ">="
  EXPAND = "+"
  LEARN  = "?"
  LPATH = "LPATH"
  URL = "URL"

class StableTransformer(lark.Transformer):
  def __init__(self, _):
    super().__init__()
    self.semantics = Semantics.STABLE

  @staticmethod
  def pack(t: str, rep: str = None, val = None, scope: dict = {}) -> tuple[str, str, str, dict]:
    return t, str(val) if rep is None else rep, rep if val is None else val, scope

  @staticmethod
  def join_scope(A: list) -> dict: return dict((y, None) for S in A for y in S[3])

  # Terminals.
  def UND(self, u): return self.pack("UND", str(u))
  def CONST(self, c): return self.pack("CONST", str(c))
  def NEG(self, n): return self.pack("NEG", str(n))
  def VAR(self, v):
    x = str(v); X = {v: None}
    return self.pack("VAR", x, scope = X)
  def ID(self, i): return self.pack("ID", val = int(i))
  def OP(self, o): return self.pack("OP", str(o))
  def PROB(self, p): return self.pack("PROB", val = float(fractions.Fraction(p.value)))
  def PATH(self, p): return self.pack("LPATH" if p[0].type == Terms.LPATH else "URL", str(p))
  def EXPAND(self, e): return self.pack("EXPAND",  str(e))
  def LEARN(self, l): return self.pack("LEARN", str(l))

  # Set.
  def set(self, S):
    L = [str(x) for x in S]
    M = set()
    for x in L:
      if x in M: raise ValueError("set must contain only unique constants!")
      M.append(x)
    return self.pack("set", val = M)

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
    return self.pack("lit", " ".join(getnths(P, 1)), P[0][2] if s else P[1][2], self.join_scope(P))
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
    if PF[0][0] == "PROB": p, f = PF[0][2], PF[1][1]
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
      b_s = ", ".join(map(lambda x: f"0, {x[1][4:]}" if x[1][:4] == "not " else f"1, {x[1]}", body_preds))
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
      if a[0] == "PROB":
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

  # Neural constructs.
  def nrule(self, A: list):
    name = A[0][0]
    inp = A[1][0]
    where_from, net = A[2]
    body = A[3:]

    V = [v[3] for v in body]
    if inp not in V:
      raise ValueError(f"Neural rule {name} is unsafe!")
    for a in body:
      pass

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
    # Mapping.
    M = {"pfact": PF, "prule": PR, "query": Q, "cfact": CF, "ad": AD}
    for t, L, O, _ in C:
      if len(L) > 0: push(P, L)
      if t in M: push(M[t], O)
    for r in PR:
      if r.is_prop: PF.append(r.prop_pf)
    return Program("\n".join(P), PF, PR, Q, CF, AD, semantics = self.semantics)

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
    b1 = ", ".join(map(lambda x: f"not _{x[1][4:]}" if x[1][:4] == "not " else x[1], R[1][2]))
    b2 = ", ".join(map(lambda x: x[1] if (x[1][:4] == "not ") or PartialTransformer.has_binop(x) else f"_{x[1]}", R[1][2]))
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
    tr_negs = lambda x: f"not _{x[1][4:]}" if x[1][:4] == "not " else x[1]
    tr_pos  = lambda x: x[1] if (x[1][:4] == "not ") or PartialTransformer.has_binop(x) else f"_{x[1]}"
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
      b1_s = ", ".join(map(lambda x: f"0, _{x[1][4:]}" if x[1][:4] == "not " else f"1, {x[1]}", body_preds))
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
    # Mapping.
    M = {"pfact": PF, "prule": PR, "query": Q, "cfact": CF, "ad": AD}
    for t, L, O, _ in C:
      if len(L) > 0: push(P, L)
      if t in M: push(M[t], O)
      if t == "prule" and isinstance(O, collections.abc.Iterable) and O[0].is_prop:
        PF.append(O[0].prop_pf)
    P.extend(f"_{x} :- {x}." for x in self.PT)
    return Program("\n".join(P), PF, PR, Q, CF, AD, semantics = self.semantics, \
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

