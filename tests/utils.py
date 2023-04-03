import unittest
import math

CONFIDENCE = 0.99

def hoeffding(n: int, tol: float = 1e-2) -> float:
  """ Hoeffding Inequality.

  Takes the number of samples `n` and returns the probability error ϵ."""
  return math.sqrt(math.log(2/(1-CONFIDENCE))/(2*n))+tol

class PaspTest(unittest.TestCase):
  def assertApproxEqual(self, X: list, Y: list, Z: list = None):
    self.assertEqual(len(X), len(Y))
    if Z is not None: self.assertEqual(len(Y), len(Z))
    for x, y in zip(X, Y): self.assertAlmostEqual(x, y)
    if Z is not None:
      for y, z in zip(Y, Z): self.assertAlmostEqual(y, z)
