import unittest

from .utils import PaspTest, hoeffding
import pasp
import numpy as np

N_SAMPLES  = 500
N_ITERS    = 10
"""Epsilon for all test cases based on the number of samples. Give some tolerance, since it's an
approximation of an approximation."""
EPS = hoeffding(N_SAMPLES)*1.5

class TestLearning(PaspTest):
  def test_insomnia_ad(self):
    which = "examples/insomnia_ad.plp"
    W = [[0.3, 0.2, 0.5], [0.1, 0.6, 0.3], [0.7, 0.1, 0.2], [0.1, 0.4, 0.5]]
    A = ["work(anna)", "work(bill)", "sleep(anna)", "sleep(bill)"]
    P = pasp.parse(which)
    Q = pasp.parse(which)

    for w in W:
      P.AD[0].P = w
      S = pasp.sample(P, A, n = N_SAMPLES)
      Q.AD[0].P = [1/len(w) for _ in range(len(w))]
      pasp.learn(Q, S, A, niters = 100, lr = 0.0001, alg = "lagrange")

      for u, v in zip(P.AD[0].P, Q.AD[0].P):
        self.assertAlmostEqual(u, v, delta = EPS)

  def test_earthquake_ad(self):
    which = "examples/earthquake_ad.plp"
    P = pasp.parse(which)
    Q = pasp.parse(which)
    A = ["alarm", "calls(a)", "calls(b)"]
    S = pasp.sample(Q, A, n = N_SAMPLES)
    P.PF[0].p = 0.5; P.PF[0].learnable = True
    P.PR[-1].p = 0.5; P.PR[-1].learnable = True
    P.PR[-2].p = 0.5; P.PR[-2].learnable = True
    pasp.learn(P, S, A, niters = N_ITERS)

    self.assertAlmostEqual(P.PF[0].p, Q.PF[0].p, delta = EPS)

  def test_neural_minimal(self):
    R = pasp.parse("examples/neural_minimal.plp")(quiet = True)
    self.assertTrue(np.allclose(R.flatten(), [0.8, 0.1], atol=1e-3))

  def test_neural_ad_minimal(self):
    R = pasp.parse("examples/neural_ad_minimal.plp")(quiet = True)
    E = [2/8, 5/8, 1/8, 1/4, 2/4, 1/4, 1/5, 2/5, 2/5, 1/3, 1/3, 1/3]
    self.assertTrue(np.allclose(R.flatten(), E, atol = 0.01))

  def test_neural_mult_minimal(self):
    R = pasp.parse("examples/neural_mult_minimal.plp")(quiet = True)
    self.assertTrue(np.allclose(R.flatten(), [0.25, 0.7, 0.1, 0.9, 0.5, 0.75], atol=1e-2))

  def test_neural_mult_ad_minimal(self):
    R = pasp.parse("examples/neural_mult_ad_minimal.plp")(quiet = True)
    E = [0.200, 0.400, 0.400,
         0.050, 0.150, 0.800,
         0.500, 0.300, 0.200,
         0.200, 0.600, 0.200,
         0.200, 0.275, 0.525,
         0.700, 0.150, 0.150,
         0.350, 0.250, 0.400,
         0.800, 0.200, 0.000,]
    self.assertTrue(np.allclose(R.flatten(), E, atol = 0.1))

  def test_neural_xor(self):
    R = pasp.parse("examples/neural_xor.plp")(quiet = True)
    self.assertTrue(np.allclose(R.flatten(), [.0, .7, .6, .0], atol = 0.01))

if __name__ == "__main__":
  unittest.main()
