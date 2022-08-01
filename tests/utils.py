import unittest

class PaspTest(unittest.TestCase):
  def assertApproxEqual(self, X: list, Y: list, Z: list = None):
    self.assertEqual(len(X), len(Y))
    if Z is not None: self.assertEqual(len(Y), len(Z))
    for x, y in zip(X, Y): self.assertAlmostEqual(x, y)
    if Z is not None:
      for y, z in zip(Y, Z): self.assertAlmostEqual(y, z)

