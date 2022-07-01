import pathlib, enum
import lark, lark.reconstruct
from .program import ProbFact, Query, ProbRule, Program

"Returns whether a node in the AST is a fact."
def is_fact(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and x.data == "fact"
"Returns whether a node in the AST is a probabilistic fact."
def is_pfact(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and x.data == "pfact"
"Returns whether a node in the AST is a rule."
def is_rule(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and x.data == "rule"
"Returns whether a node in the AST is a probabilistic rule."
def is_prule(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and x.data == "prule"
"Returns whether a node in the AST is a(n) (grounded) atom."
def is_atom(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and (x.data == "atom" or x.data == "gratom")
"Returns whether a node in the AST is a (grounded) predicate."
def is_pred(x: lark.Tree) -> bool: return isinstance(x, lark.Tree) and (x.data == "pred" or x.data == "grpred")

"Returns whether a(n) (grounded) atom is negative (`False`), positive (`True`), or not an atom (`None`)."
def atom_sign(x: lark.Tree) -> bool: return x.children[0].type != "NEG" if is_atom(x) else None
"Returns whether a (grounded) predicate is negative (`False`), positive (`True`) or not a predicate (`None`)."
def pred_sign(x: lark.Tree) -> bool: return x.children[0].type != "NEG" if is_pred(x) else None

"Returns whether a node (in this case a fact or rule) is probabilistic."
def is_prob(x: lark.Tree) -> iter:
  return isinstance(x, lark.Tree) and len(x.children) > 0 and isinstance(x.children[0], lark.Token) and x.children[0].type == "PROB"

"Returns an iterator containing all facts in the AST."
def facts(x: lark.Tree) -> iter: return x.find_pred(is_fact)
"Returns an iterator containing all probabilistic facts in the AST."
def pfacts(x: lark.Tree) -> iter: return x.find_pred(is_pfact)
"Returns an iterator containing all rules in the AST."
def rules(x: lark.Tree) -> iter: return x.find_pred(is_rule)
"Returns an iterator containing all probabilistic rules in the AST."
def prules(x: lark.Tree) -> iter: return x.find_pred(is_prule)
"Returns an iterator containing all probabilistic facts and rules in the AST."
def probs(x: lark.Tree) -> iter: return x.find_pred(lambda x: is_prule(x) or is_pfact(x))

def expand_interval(x: lark.Tree) -> tuple[int, int]:
  assert isinstance(x, lark.Tree) and x.data == "interval", "Given AST node is not an Interval."
  return int(x.children[0].value), int(x.children[1].value)

"Performs a depth-first search, returning (and stopping the search) node `y` when `f(y) == True`."
def tree_contains(x: lark.Tree, f) -> bool:
  V = set()
  def visit(x: lark.Tree | lark.Token) -> bool:
    V.add(x)
    if f(x): return True
    if isinstance(x, lark.Tree):
      for c in x.children:
        if (c not in V) and visit(c): return True
    return False
  return visit(x)

"Returns whether node `x` is grounded, i.e. whether any node in the subtree of `x` is a variable."
def is_ground(x: lark.Tree) -> bool:
  return not tree_contains(x, lambda x: isinstance(x, lark.Token) and x.type == "VAR")

"Read all `files` and parse them with grammar `G`, returning a single `lark.Tree`."
def read(*files: str, G: lark.Lark = None, from_str: bool = False) -> lark.Tree:
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

class Command(enum.Enum):
  FACT = 0
  PROB_FACT = 1
  RULE = 2
  PROB_RULE = 3
  QUERY = 4

class PLPTransformer(lark.Transformer):
  # Atoms.
  def atom(self, a: list[lark.Token | lark.Tree]) -> str: return " ".join(a)
  def gratom(self, a: list[lark.Token | lark.Tree]) -> str: return self.atom(a)

  # Intervals.
  def interval(self, i: list[lark.Token]) -> str: return "..".join(i)

  # Predicates.
  def pred(self, p: list[str]) -> str:
    sign = p[0].type == "NEG"
    if sign:
      name, args = p[1], p[2:]
      return f"not {name}({', '.join(args)})"
    name, args = p[0], p[1:]
    return f"{name}({', '.join(args)})"
  def grpred(self, p: list[str]) -> str: return self.pred(p)

  # Binary operations.
  def bop(self, b: list[lark.Tree | lark.Token]) -> str: return " ".join(b)

  # Facts.
  def fact(self, f: list[lark.Tree | lark.Token]) -> tuple[Command, str]:
    return Command.FACT, "".join(f) + "."
  def pfact(self, f: list[lark.Tree | lark.Token]) -> tuple[Command, ProbFact]:
    return Command.PROB_FACT, ProbFact(*f)

  # Heads.
  def head(self, h: list[str]) -> str: return ", ".join(h)
  # Bodies.
  def body(self, b: list[str]) -> str: return ", ".join(b)

  # Rules.
  def rule(self, r: list[str]) -> tuple[Command, str]: return Command.RULE, f"{r[0]} :- {r[1]}."
  def prule(self, r: list[str]) -> tuple[Command, str, ProbFact]:
    o = f"{r[1]} :- {r[2]}"
    l, p = ProbRule(r[0], o)
    return Command.PROB_RULE, f"{l}. % {r[0]}::{o}.", p

  # Interpretations.
  def interp(self, i: list[str]) -> list[str]: return i

  # Queries.
  def query(self, q: list[list[str]]) -> tuple[str, Query]:
    return Command.QUERY, Query(q[0], q[1] if len(q) > 1 else [])

  # Probabilistic Logic Program.
  def plp(self, C: list[tuple[Command, str] | tuple[Command, ProbFact] | tuple[Command, str, ProbFact], tuple[str, Query]]) -> Program:
    # Logic Program.
    P  = []
    # Probabilistic Facts.
    PF = []
    # Queries.
    Q  = []
    for t, *c in C:
      if t == Command.FACT: P.append(c[0])
      elif t == Command.PROB_FACT: PF.append(c[0])
      elif t == Command.RULE: P.append(c[0])
      elif t == Command.PROB_RULE: P.append(c[0]); PF.append(c[1])
      elif t == Command.QUERY:
        Q.append(c[0])
      else: P.extend(c)
    return Program("\n".join(P), PF, Q)

"""Either parses `streams` as blocks of text containing the PLP when `from_str = True`, or
interprets `streams` as filenames to be read and parsed into a `Program`."""
def parse(*files: str, G: lark.Lark = None, from_str: bool = False) -> Program:
  return PLPTransformer().transform(read(*files, G = G, from_str = from_str))

# "Pre-ground an AST, replacing all expanding all probabilistic facts with intervals."
# def pre_ground(T: lark.Tree):
  # F = T.find_pred(is_pfact)
  # for f in F:
    # I = f.find_data("interval")
    # if len(I) == 0: continue
    # U = [expand_interval(i) for i in I]
