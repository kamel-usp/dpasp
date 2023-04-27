import sys, os

import pasp

if __name__ == "__main__":
  f = "earthquake" if len(sys.argv) < 2 else sys.argv[1]
  P = pasp.parse(f"examples/{f}.plp")
  print("=======\nProgram:\n=======")
  print(P)
  print("\n==============\nQuery results:\n==============")
  R = pasp.exact(P)
  # print("\n=================\nBC Query results:\n=================")
  # pasp.exact_bc(P)
  # print("\n=====================\nCredal Query results:\n=====================")
  # pasp.exact_sym(P)
