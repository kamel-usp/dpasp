import unittest
from .utils import PaspTest
import pasp
import numpy as np

class TestCounting(PaspTest):
  @staticmethod
  def all_learnable(P: pasp.program.Program):
    for pf in P.PF: pf.learnable = True
    for ad in P.AD: ad.learnable = True

  @staticmethod
  def init_example(eg: str, semantics = "stable"):
    P = pasp.parse("examples/" + eg + ".plp", semantics = semantics)
    TestCounting.all_learnable(P)
    return P, pasp.count(P)

  def assert_strat(self, P: pasp.program.Program, C: tuple):
    k = 2**len(P.PF)
    for ad in P.AD: k *= len(ad.P)
    self.assertTrue(np.all(C[0] == k//2))
    self.assertTrue(np.array_equal([i for i in range(len(P.PF))], C[1]))
    if len(P.AD) == 0:
      self.assertIsNone(C[2])
      self.assertIsNone(C[3])
    else:
      for i, ad in enumerate(P.AD):
        self.assertTrue(np.all(C[2][i] == k//len(ad.P)))
        self.assertEqual(C[3][i], i)

  def test_asia(self):
    P, C = self.init_example("asia")
    self.assert_strat(P, C)

  def test_earthquake(self):
    P, C = self.init_example("earthquake")
    self.assert_strat(P, C)

  def test_simple(self):
    P, C = self.init_example("simple")
    self.assert_strat(P, C)

  def test_simpler(self):
    P, C = self.init_example("simpler")
    self.assert_strat(P, C)

  def test_smokers(self):
    P, C = self.init_example("smokers")
    self.assert_strat(P, C)

  def test_earthquake_ad(self):
    P, C = self.init_example("earthquake_ad")
    self.assert_strat(P, C)

  def test_multinsomnia(self):
    P, C = self.init_example("multinsomnia")
    self.assertTrue(np.array_equal(C[0], [[54, 27], [54, 27], [54, 27]]))
    self.assertTrue(np.array_equal(C[1], [0, 1, 2]))
    self.assertTrue(np.array_equal(C[2][0], [27, 27, 27]))
    self.assertTrue(np.array_equal(C[3], [0]))

  def test_barber(self):
    P, C = self.init_example("barber", semantics = "lstable")
    self.assertTrue(np.array_equal(C[0], [[1, 1]]))
    self.assertTrue(np.array_equal(C[1], [0]))
    self.assertIsNone(C[2])
    self.assertIsNone(C[3])

  def test_game(self):
    P, C = self.init_example("game")
    self.assertTrue(np.array_equal(C[0], [[1, 2]]))
    self.assertTrue(np.array_equal(C[1], [0]))
    self.assertIsNone(C[2])
    self.assertIsNone(C[3])

  def test_insomnia(self):
    P, C = self.init_example("insomnia")
    self.assertTrue(np.array_equal(C[0], [[2, 1]]))
    self.assertTrue(np.array_equal(C[1], [0]))
    self.assertIsNone(C[2])
    self.assertIsNone(C[3])

if __name__ == "__main__":
  unittest.main()
