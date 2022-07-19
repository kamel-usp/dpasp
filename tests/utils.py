import unittest

class PaspTest(unittest.TestCase):
  def assertApproxEqual(self, X: list, Y: list):
    self.assertEqual(len(X), len(Y))
    for x, y in zip(X, Y):
      self.assertAlmostEqual(x, y)

