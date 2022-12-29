import unittest

from .utils import PaspTest, hoeffding
import pasp
import numpy as np

N_SAMPLES  = 500
N_ITERS    = 100
"""Epsilon for all test cases based on the number of samples. Give some tolerance, since it's an
approximation of an approximation."""
EPS = hoeffding(N_SAMPLES)*1.5

class TestLearning(PaspTest):
  def test_insomnia_ad(self):
    which = "examples/insomnia_ad.lp"
    W = [[0.3, 0.2, 0.5], [0.1, 0.6, 0.3], [0.9, 0.1, 0.0], [0.0, 0.5, 0.5]]
    P = pasp.parse(which)
    Q = pasp.parse(which)
    A = ["work(anna)", "work(bill)", "sleep(anna)", "sleep(bill)"]

    for w in W:
      P.AD[0].P = w
      S = pasp.sample(P, A, n = N_SAMPLES)
      Q.AD[0].P = [1/len(w) for _ in range(len(w))]
      pasp.learn(Q, S, A, niters = N_ITERS)

      for u, v in zip(P.AD[0].P, Q.AD[0].P):
        self.assertAlmostEqual(u, v, delta = EPS)

  def test_earthquake_ad(self):
    which = "examples/earthquake_ad.lp"
    P = pasp.parse(which)
    Q = pasp.parse(which)
    A = ["alarm", "calls(a)", "calls(b)"]
    S = pasp.sample(Q, A, n = N_SAMPLES)
    P.PF[0].p = 0.5; P.PF[0].learnable = True
    pasp.learn(P, S, A, niters = N_ITERS)

    self.assertAlmostEqual(P.PF[0].p, Q.PF[0].p, delta = EPS)

if __name__ == "__main__":
  unittest.main()
