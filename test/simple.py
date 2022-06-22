import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)[:-4], "src"))

import credal

if __name__ == "__main__":
  f = "earthquake" if len(sys.argv) < 2 else sys.argv[1]
  P = credal.parse(f"examples/{f}.lp")
  print("=======\nProgram:\n=======")
  print(P)
  print("\n==============\nQuery results:\n==============")
  R = credal.exact(P)
