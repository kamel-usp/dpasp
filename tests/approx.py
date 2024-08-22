import unittest
from .utils import PaspTest
import numpy as np
import pasp

class TestApprox(PaspTest):
  def test_asia(self):
    P = pasp.parse("examples/asia.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, quiet=True, psemantics="maxent")
    self.assertApproxEqual(R, R_ex)

  def test_earthquake(self):
    P = pasp.parse("examples/earthquake.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, quiet=True, psemantics="maxent")
    self.assertApproxEqual(R, R_ex)

  def test_game(self):
    P = pasp.parse("examples/game.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_insomnia(self):
    P = pasp.parse("examples/insomnia.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_simple(self):
    P = pasp.parse("examples/simple.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_simpler(self):
    P = pasp.parse("examples/simpler.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_smokers(self):
    P = pasp.parse("examples/smokers.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_earthquake_ad(self):
    P = pasp.parse("examples/earthquake_ad.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

  def test_multinsomnia(self):
    P = pasp.parse("examples/multinsomnia.plp")
    R = pasp.approx.aseo(P, n_samples=4096, quiet=True)
    R_ex = pasp.exact(P, psemantics="maxent", quiet=True)
    self.assertApproxEqual(R, R_ex)

if __name__ == "__main__":
  unittest.main()
