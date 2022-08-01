import unittest
from .utils import PaspTest
import pasp

class TestExamples(PaspTest):
  def test_asia(self):
    P = pasp.parse("examples/asia.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(trip)
    self.assertApproxEqual(R[0], [0.01, 0.01], C[0])
    # ℙ(tuberculosis | trip)
    self.assertApproxEqual(R[1], [0.05, 0.05], C[1])
    # ℙ(cancer | smoking)
    self.assertApproxEqual(R[2], [0.1,  0.1 ], C[2])
    # ℙ(test | or)
    self.assertApproxEqual(R[3], [0.98, 0.98], C[3])
    # ℙ(smoking)
    self.assertApproxEqual(R[4], [0.5,  0.5 ], C[4])
    # ℙ(tuberculosis | not trip)
    self.assertApproxEqual(R[5], [0.01, 0.01], C[5])
    # ℙ(cancer | not smoking)
    self.assertApproxEqual(R[6], [0.01, 0.01], C[6])
    # ℙ(test | not or)
    self.assertApproxEqual(R[7], [0.05, 0.05], C[7])

  def test_earthquake(self):
    P = pasp.parse("examples/earthquake.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(alarm | burglary, earthquake)
    self.assertApproxEqual(R[0], [0.9,  0.9], C[0])
    # ℙ(alarm | not burglary, earthquake)
    self.assertApproxEqual(R[1], [0.1,  0.1], C[1])
    # ℙ(alarm | burglary, not earthquake)
    self.assertApproxEqual(R[2], [0.8,  0.8], C[2])
    # ℙ(alarm | not burglary, not earthquake)
    self.assertApproxEqual(R[3], [0.0,  0.0], C[3])

  def test_game(self):
    P = pasp.parse("examples/game.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(wins(b))
    self.assertApproxEqual(R[0], [0.7,  1.0], C[0])
    # ℙ(wins(c))
    self.assertApproxEqual(R[1], [0.3,  0.3], C[1])

  def test_insomnia(self):
    P = pasp.parse("examples/insomnia.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(insomnia)
    self.assertApproxEqual(R[0], [0.3,  0.3], C[0])
    # ℙ(work)
    self.assertApproxEqual(R[1], [0.3,  1.0], C[1])
    # ℙ(sleep)
    self.assertApproxEqual(R[2], [0.0,  0.7], C[2])
    # ℙ(not sleep)
    self.assertApproxEqual(R[3], [0.3,  1.0], C[3])
    # ℙ(not work)
    self.assertApproxEqual(R[4], [0.0,  0.7], C[4])

  def test_prisoners(self):
    P = pasp.parse("examples/prisoners.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    α = 19/40
    # ℙ(e1 | u)
    self.assertApproxEqual(R[0], [1.0/(1+2*(((1-α)/α)**2)), 1.0/(1+2*((α/(1-α))**2))], C[0])
    # ℙ(e1 | not b, u)
    self.assertApproxEqual(R[1], [1.0/(1+((1-α)/α)**2), 1.0/(1+((α/(1-α))**2))], C[1])
    # ℙ(g | e1, u)
    self.assertApproxEqual(R[2], [0.0, 1.0], C[2])
    # ℙ(d)
    self.assertApproxEqual(R[3], [0.0, 1.0], C[3])
    # ℙ(e1 | g, u)
    self.assertApproxEqual(R[4], [0.0, 1.0/(1+(α/(1-α))**2)], C[4])
    # ℙ(e1 | ga, u)
    self.assertApproxEqual(R[5], [1.0/(1+(((1-α)/α)**2)/α), 1.0/(1+((α/(1-α))**2)/(1-α))], C[5])

  def test_simple(self):
    P = pasp.parse("examples/simple.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(s(a))
    self.assertApproxEqual(R[0], [0.20, 0.20], C[0])
    # ℙ(s(b))
    self.assertApproxEqual(R[1], [0.30, 0.30], C[1])
    # ℙ(v)
    self.assertApproxEqual(R[2], [0.048, 0.048], C[2])

  def test_simpler(self):
    P = pasp.parse("examples/simpler.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(r)
    self.assertApproxEqual(R[0], [0.5, 0.5], C[0])
    # ℙ(v)
    self.assertApproxEqual(R[1], [0.25, 0.25], C[1])

  def test_smokers(self):
    P = pasp.parse("examples/smokers.lp")
    R, C = pasp.exact_py(P), pasp.exact(P)
    # ℙ(smokes(a))
    self.assertApproxEqual(R[0], [0.06, 0.06], C[0])
    # ℙ(smokes(b))
    self.assertApproxEqual(R[1], [0.2, 0.2], C[1])

if __name__ == "__main__":
  unittest.main()
