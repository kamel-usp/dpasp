import pathlib, enum
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

class PLPTransformer(lark.Transformer):
  def __init__(self, _):
    super().__init__()
    self.semantics = Semantics.STABLE

  # Terminals.
  def UND(self, a: list[lark.Token]) -> tuple[str, bool]: return a, True
  def CONST(self, a: list[lark.Token]) -> tuple[str, bool]: return a, True
  def NEG(self, a: list[lark.Token]) -> tuple[str, bool]: return a, True
  def VAR(self, a: list[lark.Token]) -> tuple[str, bool]: return a, False
  def ID(self, a: list[lark.Token]) -> tuple[str, bool]: return a, True
  def OP(self, a: list[lark.Token]) -> tuple[str, bool]: return a, True

  # Atoms.
  def atom(self, a: list[lark.Tree]) -> tuple[str, bool]:
    return " ".join(getnths(a, 0)), all(getnths(a, 1)), True
  def gratom(self, a: list[lark.Tree]) -> tuple[str, bool]: return self.atom(a)[0], True
  def pgratom(self, a): return self.atom(a)[0], True

  # Intervals.
  def interval(self, i: list[lark.Token]) -> tuple[str, bool]:
    return "..".join(getnths(i, 0)), all(getnths(i, 1))

  # Predicates.
  def pred(self, p: list[tuple[str, bool]]) -> tuple[str, bool]:
    sign = p[0][0].type == "NEG"
    if sign:
      name, args = p[1], p[2:]
      return f"not {name}({', '.join(getnths(args, 0))})"
    name, args = p[0], p[1:]
    return f"{name[0]}({', '.join(getnths(args, 0))})", all(getnths(args, 1)), True
  def grpred(self, p: list[tuple[str, bool]]) -> tuple[str, bool]: return self.pred(p)[0], True
  def pgrpred(self, p): return self.pred(p)[0], True

  # Binary operations.
  def bop(self, b: list[lark.Tree]) -> str:
    return " ".join(getnths(b, 0)), all(getnths(b, 1)), False

  # Facts.
  def fact(self, f: list[lark.Tree]) -> tuple[Command, str]:
    return Command.FACT, "".join(getnths(f, 0)) + "."
  def pfact(self, f: list[lark.Tree]) -> tuple[Command, ProbFact]:
    return Command.PROB_FACT, ProbFact(f[0], f[1][0])
  def cfact(self, f: list[lark.Tree]) -> tuple[Command, CredalFact]:
    return Command.CRED_FACT, CredalFact(str(f[0]), str(f[1]), f[2][0])

  # Heads.
  def head(self, h: list[str]) -> str:
    return ", ".join(getnths(h, 0)), all(getnths(h, 1)), [x for x in getnths(h, 0)]
  def ohead(self, h: list[str]) -> str:
    if len(h) == 1: return str(h[0][0]), True, h[0][0], ()
    u = h[1:]
    return f"{h[0][0]}({', '.join(getnths(u, 0))})", all(getnths(u, 1)), h[0][0], list(getnths(u, 0))
  # Bodies.
  def body(self, b: list[str]) -> str:
    return ", ".join(getnths(b, 0)), all(getnths(b, 1)), list(x for x, t in zip(getnths(b, 0), getnths(b, 2)) if t), list(getnths(b, 0))

  # Rules.
  def rule(self, r: list[str]) -> tuple[Command, str]:
    return Command.RULE, f"{r[0][0]} :- {r[1][0]}."
  def prule(self, r: list[str]) -> tuple[Command, str, ProbFact]:
    o = f"{r[1][0]} :- {r[2][0]}"
    prop = r[1][1] and r[2][1]
    if prop: return Command.PROB_RULE, ProbRule(r[0], o, is_prop = True)
    h, b = r[1][3], r[2][2]
    # Invariant: len(b) > 0, otherwise the rule is unsafe.
    h_s = ", ".join(h) + ", " if len(h) > 0 else ""
    b_s = ", ".join(map(lambda x: f"0, {x[4:]}" if x[:4] == "not " else f"1, {x}", b))
    u = f"{r[1][2]}(@unify(\"{r[0]}\", {r[1][2]}, {len(h)}, {2*len(b)}, {h_s}{b_s})) :- {r[2][0]}."
    return Command.PROB_RULE, ProbRule(r[0], o, is_prop = False, unify = u)

  # Annotated disjunction head.
  def adhead(self, h: list):
    return [float(h[i]) for i in range(0, len(h), 2)], [h[i][0] for i in range(1, len(h), 2)]
  # Annotated disjunctions.
  def ad(self, d: list):
    return Command.ANNOTATED_DISJUNCTION, AnnotatedDisjunction(d[0][0], d[0][1])

  # Constraint.
  def constraint(self, b: list[str]) -> tuple[Command, str]:
    return Command.CONSTRAINT, f":- {b[0][0]}."

  # Interpretations.
  def interp(self, i: list[str]) -> list[str]: return list(getnths(i, 0))

  # Queries.
  def query(self, q: list[list[str]]) -> tuple[str, Query]:
    return Command.QUERY, Query(q[0], q[1] if len(q) > 1 else [], semantics = self.semantics)

  def constdef(self, t: list) -> tuple[Command, str]:
    return Command.CONSTDEF, f"#const {t[0][0]} = {t[1][0]}."

  # Probabilistic Logic Program.
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
    for t, *c in C:
      if t == Command.FACT: P.extend(c)
      elif t == Command.PROB_FACT: PF.extend(c)
      elif t == Command.RULE: P.extend(c)
      elif t == Command.PROB_RULE:
        for r in c:
          PR.append(r)
          if r.is_prop:
            P.append(r.prop_f)
            PF.append(r.prop_pf)
      elif t == Command.QUERY: Q.extend(c)
      elif t == Command.CRED_FACT: CF.extend(c)
      elif t == Command.CONSTRAINT: P.extend(c)
      elif t == Command.ANNOTATED_DISJUNCTION: AD.extend(c)
      else: P.extend(c)
    return Program("\n".join(P), PF, PR, Q, CF, AD, semantics = self.semantics)

class PartialTransformer(PLPTransformer):
  def __init__(self, sem: str):
    super().__init__(sem)
    self.PT = set()
    self.semantics = Semantics.LSTABLE if sem == "lstable" else Semantics.PARTIAL
    self.o_tree = None

  @staticmethod
  def has_binop(x: str): return ("=" in x) or ("<" in x) or (">" in x)

  def fact(self, f) -> tuple[Command, str]:
    u = "".join(getnths(f, 0))
    self.PT.add(u)
    return Command.FACT, u + "."

  def pfact(self, f) -> tuple[Command, ProbFact]:
    self.PT.add(f[1][0])
    return Command.PROB_FACT, ProbFact(f[0], f[1][0])

  def cfact(self, f) -> tuple[Command, CredalFact]:
    self.PT.add(f[2][0])
    return Command.CRED_FACT, CredalFact(str(f[0]), str(f[1]), f[2][0])

  def rule(self, r) -> tuple[Command, str, str, str]:
    b1 = ", ".join(map(lambda x: f"not _{x[4:]}" if x[:4] == "not " else x, r[1][3]))
    b2 = ", ".join(map(lambda x: x if (x[:4] == "not ") or PartialTransformer.has_binop(x) else f"_{x}", r[1][3]))
    h1, h2 = ", ".join(r[0][2]), ", ".join(map(lambda x: f"_{x}", r[0][2]))
    for h in r[0][2]: self.PT.add(h)
    # for x in r[1][3]:
      # if not PartialTransformer.has_binop(x): self.PT.add(x[4:] if x[:4] == "not " else x)
    return Command.RULE, f"{h1} :- {b1}.", f"{h2} :- {b2}."

  def prule(self, r) -> tuple[Command, str, str, str]:
    tr_negs = lambda x: f"not _{x[4:]}" if x[:4] == "not " else x
    tr_pos  = lambda x: x if (x[:4] == "not ") or PartialTransformer.has_binop(x) else f"_{x}"
    b1 = ", ".join(map(tr_negs, r[2][3]))
    b2 = ", ".join(map(tr_pos, r[2][3]))
    h = r[1][0]
    o1, o2 = f"{h} :- {b1}", f"_{h} :- {b2}"
    self.PT.add(h)
    # for x in r[2][3]:
      # if not PartialTransformer.has_binop(x): self.PT.add(x[4:] if x[:4] == "not " else x)
    prop = r[1][1] and r[2][1]
    uid = unique_fact()
    if prop: return Command.PROB_RULE, ProbRule(r[0], o1, ufact = uid), ProbRule(r[0], o2, ufact = uid)
    h_a, b = r[1][3], r[2][2]
    # Invariant: len(b) > 0, otherwise the rule is unsafe.
    h_s = ", ".join(h_a) + ", " if len(h_a) > 0 else ""
    b1_s = ", ".join(map(lambda x: f"0, _{x[4:]}" if x[:4] == "not " else f"1, {x}", b))
    # Let the grounder deal with the _f rule.
    u1 = f"{r[1][2]}(@unify(\"{r[0]}\", {r[1][2]}, {len(h_a)}, {2*len(b)}, {h_s}{b1_s})) :- {b1}."
    return Command.PROB_RULE, ProbRule(r[0], o1, is_prop = False, unify = u1)

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
    for t, *c in C:
      if t == Command.FACT: P.extend(c)
      elif t == Command.PROB_FACT: PF.extend(c)
      elif t == Command.RULE: P.extend(c)
      elif t == Command.PROB_RULE:
        if len(c) == 1:
          PR.append(c[0])
          continue
        r1, r2 = c
        PR.extend((r1, r2))
        if r1.is_prop:
          P.append(r1.prop_f)
          PF.append(r1.prop_pf)
        if r2.is_prop:
          P.append(r2.prop_f)
      elif t == Command.QUERY: Q.extend(c)
      elif t == Command.CRED_FACT: CF.extend(c)
      elif t == Command.CONSTRAINT: P.extend(c)
      elif t == Command.ANNOTATED_DISJUNCTION: AD.extend(c)
      else: P.extend(c)
    P.extend(f"_{x} :- {x}." for x in self.PT)
    return Program("\n".join(P), PF, PR, Q, CF, AD, semantics = self.semantics, \
                   stable_p = self.stable_p)

  def transform(self, tree):
    self.o_tree = tree
    self.stable_p = PLPTransformer(self.semantics).transform(tree)
    return super().transform(tree)

def parse(*files: str, G: lark.Lark = None, from_str: bool = False, semantics: str = "stable") -> Program:
  """Either parses `streams` as blocks of text containing the PLP when `from_str = True`, or
  interprets `streams` as filenames to be read and parsed into a `Program`."""
  if semantics not in parse.trans_map:
    raise ValueError("semantics not supported (must either be 'stable', 'partial' or 'lstable')!")
  return parse.trans_map[semantics](semantics).transform(read(*files, G = G, from_str = from_str))
parse.trans_map = {}
parse.trans_map["stable"] = PLPTransformer
parse.trans_map["lstable"] = PartialTransformer
parse.trans_map["partial"] = PartialTransformer

