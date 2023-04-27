import pasp
import sys

ARGUMENTS = ["sem", "psem", "help"]
ARGUMENTS_SHORTCUTS = ["s", "p", "h"]
ARGUMENTS_HELP = {
  "sem": "Sets the semantics of the logic program.",
  "psem": "Sets the probabilistic semantics of the program.",
  "help": "Shows this help message."
}
ARGUMENTS_VALUES = {
  "sem": [
    ("stable", "Stable model semantics."),
    ("lstable", "L-stable model semantics."),
    ("partial", "Partial model semantics."),
    ("smproblog", "SMProbLog semantics."),
  ],
  "psem": [
    ("credal", "Credal semantics."),
    ("maxent", "Max-Entropy (uniform) semantics."),
  ],
  "help": None,
}
ARGUMENTS_HEADERS = [f"--{ARGUMENTS[i]}=<val> | -{ARGUMENTS_SHORTCUTS[i]} <val>" \
                     if ARGUMENTS_VALUES[a] is not None \
                     else f"--{ARGUMENTS[i]} | -{ARGUMENTS_SHORTCUTS[i]}" \
                     for i, a in enumerate(ARGUMENTS)]
ARGUMENTS_VALUES_HEADERS = {k: [ARGUMENTS_VALUES[k][i][0] for i in range(len(ARGUMENTS_VALUES[k]))] \
                            if ARGUMENTS_VALUES[k] is not None else None for k in ARGUMENTS}
ARGUMENTS_LJUST = len(max(ARGUMENTS_HEADERS, key=len))
ARGUMENTS_VALUES_LJUST = {k: len(max(ARGUMENTS_VALUES_HEADERS[k], key=len)) \
                          if ARGUMENTS_VALUES[k] is not None else None for k in ARGUMENTS}

EXAMPLES = [
"""
  % Runs the prisoners example with credal inference and stable semantics.
  pasp examples/prisoners.lp

  ℙ(e1 | u) = [0.290426, 0.379192]
  ℙ(e1 | not b, u) = [0.450125, 0.549875]
  ℙ(g | e1, u) = [0.000000, 1.000000]
  ℙ(d) = [0.000000, 1.000000]
  ℙ(e1 | g, u) = [0.000000, 0.549875]
  ℙ(e1 | ga, u) = [0.279971, 0.390743]
""",
"""
  % Runs the 3-coloring example with credal inference and L-stable semantics.
  pasp --sem=lstable examples/3coloring.lp

  ℙ(c(1,r)) = [0.000000, 1.000000]
  ℙ(e(1,2) | undef f) = [0.772727, 0.772727]
  ℙ(undef f) = [0.064453, 0.064453]
""",
"""
  % Runs the insomnia example with Max-Entropy inference and stable semantics.
  pasp --psem=maxent examples/insomnia.lp

  ℙ(insomnia) = [0.300000, 0.300000]
  ℙ(work) = [0.650000, 0.650000]
  ℙ(sleep) = [0.350000, 0.350000]
  ℙ(not sleep) = [0.650000, 0.650000]
  ℙ(not work) = [0.350000, 0.350000]
""",
]

def print_help():
  print("""pasp - Probabilistic Answer Set Programming
Usage: pasp [options] [files]

OPTIONS\n""")
  for i, a in enumerate(ARGUMENTS):
    print("  " + ARGUMENTS_HEADERS[i].ljust(ARGUMENTS_LJUST, ' ') + " : " + ARGUMENTS_HELP[a])
    if ARGUMENTS_VALUES[a] is not None:
      print("    Possible values:")
      for j in range(len(ARGUMENTS_VALUES[a])):
        print("      " + \
              ARGUMENTS_VALUES_HEADERS[a][j].ljust(ARGUMENTS_VALUES_LJUST[a], ' ') + " : " + \
              ARGUMENTS_VALUES[a][j][1])

  print("\nDefault values for options come first.")
  print("\nEXAMPLES")

  for e in EXAMPLES:
    print(e)

  print("\npasp is available at https://github.com/RenatoGeh/pasp.")
  print("Get help/report bugs via: https://github.com/RenatoGeh/issues.")

def try_arg(args: dict, a: str, pre: int, sep: str):
  if a.startswith(pre):
    T = a.split(sep)
    arg = T[0][len(pre):]
    if arg == "h" or arg == "help":
      print_help()
      sys.exit(0)
    is_shortcut = True
    try: arg = ARGUMENTS[ARGUMENTS_SHORTCUTS.index(arg)]
    except: is_shortcut = False
    if (arg not in ARGUMENTS) and (not is_shortcut):
      print(f"Unrecognized command: {T[0]}!")
      print_help()
      sys.exit(1)
    if len(T) != 2:
      print(f"Unable to parse argument-value: {a}!")
      print_help()
      sys.exit(1)
    val = T[1]
    if val not in ARGUMENTS_VALUES_HEADERS[arg]:
      print(f"Unrecognized option {val} for argument {arg}!")
      print_help()
      sys.exit(1)
    args[arg] = val
    return True
  return False

def parse_args() -> dict:
  args = {k: ARGUMENTS_VALUES[k][0][0] if ARGUMENTS_VALUES[k] is not None else None \
          for k in ARGUMENTS}
  files = []
  I = enumerate(sys.argv[1:], start = 1)
  for i, a in I:
    # Hack to make sure the second try_arg only occurs when -- does not work.
    if (not try_arg(args, a, "--", "=")):
      if try_arg(args, a + " " + sys.argv[i+1] if i+1 < len(sys.argv) else a, "-", " "):
        next(I)
      else:
        files.append(a)

  return args, files

def main():
  A, F = parse_args()
  if len(F) > 0:
    P = pasp.parse(*F, semantics = A["sem"])
    if "psemantics" not in P.directives: P.directives["psemantics"] = {"psemantics": A["psem"]}
    P()
  else:
    print("pasp version", pasp.__version__)
    inp = ""
    for l in sys.stdin: inp += l
    pasp.exact(pasp.parse(inp, from_str = True, semantics = A["sem"]), psemantics = A["psem"])
  return 0

if __name__ == "__main__": main()
