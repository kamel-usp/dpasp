import unittest
from .utils import PaspTest
import numpy as np
import pasp

class TestExamples(PaspTest):
  def test_asia(self):
    P = pasp.parse("examples/asia.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(trip)
    self.assertApproxEqual(R[0,:], [0.01, 0.01])
    # ℙ(tuberculosis | trip)
    self.assertApproxEqual(R[1,:], [0.05, 0.05])
    # ℙ(cancer | smoking)
    self.assertApproxEqual(R[2,:], [0.1,  0.1 ])
    # ℙ(test | or)
    self.assertApproxEqual(R[3,:], [0.98, 0.98])
    # ℙ(smoking)
    self.assertApproxEqual(R[4,:], [0.5,  0.5 ])
    # ℙ(tuberculosis | not trip)
    self.assertApproxEqual(R[5,:], [0.01, 0.01])
    # ℙ(cancer | not smoking)
    self.assertApproxEqual(R[6,:], [0.01, 0.01])
    # ℙ(test | not or)
    self.assertApproxEqual(R[7,:], [0.05, 0.05])

  def test_earthquake(self):
    P = pasp.parse("examples/earthquake.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(alarm | burglary, earthquake)
    self.assertApproxEqual(R[0,:], [0.9,  0.9])
    # ℙ(alarm | not burglary, earthquake)
    self.assertApproxEqual(R[1,:], [0.1,  0.1])
    # ℙ(alarm | burglary, not earthquake)
    self.assertApproxEqual(R[2,:], [0.8,  0.8])
    # ℙ(alarm | not burglary, not earthquake)
    self.assertApproxEqual(R[3,:], [0.0,  0.0])

  def test_game(self):
    P = pasp.parse("examples/game.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(wins(b))
    self.assertApproxEqual(R[0,:], [0.7,  1.0])
    # ℙ(wins(c))
    self.assertApproxEqual(R[1,:], [0.3,  0.3])

  def test_insomnia(self):
    P = pasp.parse("examples/insomnia.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(insomnia)
    self.assertApproxEqual(R[0,:], [0.3,  0.3])
    # ℙ(work)
    self.assertApproxEqual(R[1,:], [0.3,  1.0])
    # ℙ(sleep)
    self.assertApproxEqual(R[2,:], [0.0,  0.7])
    # ℙ(not sleep)
    self.assertApproxEqual(R[3,:], [0.3,  1.0])
    # ℙ(not work)
    self.assertApproxEqual(R[4,:], [0.0,  0.7])

  def test_prisoners(self):
    P = pasp.parse("examples/prisoners.plp")
    R = pasp.exact(P, quiet = True)
    α = 19/40
    # ℙ(e1 | u)
    print(R[0,:], [1.0/(1+2*(((1-α)/α)**2)), 1.0/(1+2*((α/(1-α))**2))])
    self.assertApproxEqual(R[0,:], [1.0/(1+2*(((1-α)/α)**2)), 1.0/(1+2*((α/(1-α))**2))])
    # ℙ(e1 | not b, u)
    self.assertApproxEqual(R[1,:], [1.0/(1+((1-α)/α)**2), 1.0/(1+((α/(1-α))**2))])
    # ℙ(g | e1, u)
    self.assertApproxEqual(R[2,:], [0.0, 1.0])
    # ℙ(d)
    self.assertApproxEqual(R[3,:], [0.0, 1.0])
    # ℙ(e1 | g, u)
    self.assertApproxEqual(R[4,:], [0.0, 1.0/(1+(α/(1-α))**2)])
    # ℙ(e1 | ga, u)
    self.assertApproxEqual(R[5,:], [1.0/(1+(((1-α)/α)**2)/α), 1.0/(1+((α/(1-α))**2)/(1-α))])

  def test_simple(self):
    P = pasp.parse("examples/simple.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(s(a))
    self.assertApproxEqual(R[0,:], [0.20, 0.20])
    # ℙ(s(b))
    self.assertApproxEqual(R[1,:], [0.30, 0.30])
    # ℙ(v)
    self.assertApproxEqual(R[2,:], [0.048, 0.048])

  def test_simpler(self):
    P = pasp.parse("examples/simpler.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(r)
    self.assertApproxEqual(R[0,:], [0.5, 0.5])
    # ℙ(v)
    self.assertApproxEqual(R[1,:], [0.25, 0.25])

  def test_smokers(self):
    P = pasp.parse("examples/smokers.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(smokes(a))
    self.assertApproxEqual(R[0,:], [0.06, 0.06])
    # ℙ(smokes(b))
    self.assertApproxEqual(R[1,:], [0.2, 0.2])

  def test_earthquake_ad(self):
    P = pasp.parse("examples/earthquake_ad.plp")
    R = pasp.exact(P, quiet = True)
    # ℙ(alarm | burglary, earthquake(heavy))
    self.assertApproxEqual(R[0,:], [0.9, 0.9])
    # ℙ(alarm | not burglary, earthquake(mild))
    self.assertApproxEqual(R[1,:], [0.1, 0.1])
    # ℙ(alarm | burglary, not earthquake(mild))
    self.assertApproxEqual(R[2,:], [0.8058823529411767, 0.8058823529411767])
    # ℙ(alarm | not burglary, earthquake(none))
    self.assertApproxEqual(R[3,:], [0.0, 0.0])

  def test_multinsomnia(self):
    P = pasp.parse("examples/multinsomnia.plp")
    R = pasp.exact(P, quiet = True)
    O = [[0.300000, 0.300000], [0.500000, 0.500000], [0.700000, 0.700000], [0.300000, 1.000000],
         [0.500000, 1.000000], [0.700000, 1.000000], [0.000000, 0.700000], [0.000000, 0.500000],
         [0.000000, 0.300000], [0.030000, 0.200000], [0.042000, 0.200000], [0.067500, 0.450000],
         [0.157500, 0.450000], [0.073500, 0.350000], [0.122500, 0.350000]]
    for i, o in enumerate(O):
      self.assertApproxEqual(R[i,:], o)

  def test_fault_tree(self):
    P = pasp.parse("examples/fault_tree.plp")
    R = pasp.exact(P, quiet = True)
    # All marginal probabilities.
    T = np.array([[0.05     , 0.1       ], [0.05      , 0.1       ], [0.005    , 0.01      ],
                  [0.05475  , 0.109     ], [0.05475   , 0.109     ], [0.01198  , 0.019925  ],
                  [0.0074875, 0.0199    ], [0.01741262, 0.0346015 ], [0.0193778, 0.03942849],
                  [0.001    , 0.01      ], [0.02035842, 0.04903421]])
    self.assertTrue(np.allclose(R[0:-4], T))

  # This test might take a long time. Ignore for now.
  # def test_rain_util(self):
    # P = pasp.parse("examples/rain_utility.plp")
    # R = pasp.exact(P, quiet = True)
    # self.assertAlmostEqual(R[0,1], 0.5075)

class TestLStable(PaspTest):
  def test_barber(self):
    P = pasp.parse("examples/barber.plp", semantics = "lstable")
    R = pasp.exact(P, quiet = True)
    # ℙ(shaves(b, a))
    self.assertApproxEqual(R[0,:], [1.0, 1.0])
    # ℙ(not shaves(b, b))
    self.assertApproxEqual(R[1,:], [0.5, 0.5])
    # ℙ(undef shaves(b, b))
    self.assertApproxEqual(R[2,:], [0.5, 0.5])

  def test_3coloring(self):
    P = pasp.parse("examples/3coloring.plp", semantics = "lstable")
    R = pasp.exact(P, quiet = True)
    # ℙ(c(1, r))
    self.assertApproxEqual(R[0,:], [0.0, 1.0])
    # ℙ(e(1, 2) | undef f)
    self.assertApproxEqual(R[1,:], [0.7727272727272727, 0.7727272727272727])
    # ℙ(undef f)
    self.assertApproxEqual(R[2,:], [0.064453125, 0.064453125])

class TestPlog(PaspTest):
  def test_asia(self):
    P = pasp.parse("examples/asia.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(trip)
    self.assertAlmostEqual(R[0], 0.01)
    # ℙ(tuberculosis | trip)
    self.assertAlmostEqual(R[1], 0.05)
    # ℙ(cancer | smoking)
    self.assertAlmostEqual(R[2], 0.1)
    # ℙ(test | or)
    self.assertAlmostEqual(R[3], 0.98)
    # ℙ(smoking)
    self.assertAlmostEqual(R[4], 0.5)
    # ℙ(tuberculosis | not trip)
    self.assertAlmostEqual(R[5], 0.01)
    # ℙ(cancer | not smoking)
    self.assertAlmostEqual(R[6], 0.01)
    # ℙ(test | not or)
    self.assertAlmostEqual(R[7], 0.05)

  def test_earthquake(self):
    P = pasp.parse("examples/earthquake.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(alarm | burglary, earthquake)
    self.assertAlmostEqual(R[0], 0.9)
    # ℙ(alarm | not burglary, earthquake)
    self.assertAlmostEqual(R[1], 0.1)
    # ℙ(alarm | burglary, not earthquake)
    self.assertAlmostEqual(R[2], 0.8)
    # ℙ(alarm | not burglary, not earthquake)
    self.assertAlmostEqual(R[3], 0.0)

  def test_simple(self):
    P = pasp.parse("examples/simple.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(s(a))
    self.assertAlmostEqual(R[0], 0.20)
    # ℙ(s(b))
    self.assertAlmostEqual(R[1], 0.30)
    # ℙ(v)
    self.assertAlmostEqual(R[2], 0.048)

  def test_simpler(self):
    P = pasp.parse("examples/simpler.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(r)
    self.assertAlmostEqual(R[0], 0.5)
    # ℙ(v)
    self.assertAlmostEqual(R[1], 0.25)

  def test_smokers(self):
    P = pasp.parse("examples/smokers.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(smokes(a))
    self.assertAlmostEqual(R[0], 0.06)
    # ℙ(smokes(b))
    self.assertAlmostEqual(R[1], 0.2)

  def test_game(self):
    P = pasp.parse("examples/game.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(wins(b))
    self.assertAlmostEqual(R[0], 1.7/2)
    # ℙ(wins(c))
    self.assertAlmostEqual(R[1], 0.3)

  def test_insomnia(self):
    P = pasp.parse("examples/insomnia.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(insomnia)
    self.assertAlmostEqual(R[0], 0.3)
    # ℙ(work)
    self.assertAlmostEqual(R[1], 0.3+(1.0-0.3)/2)
    # ℙ(sleep)
    self.assertAlmostEqual(R[2], 0.7/2)
    # ℙ(not sleep)
    self.assertAlmostEqual(R[3], 0.3+(1.0-0.3)/2)
    # ℙ(not work)
    self.assertAlmostEqual(R[4], 0.7/2)

  def test_earthquake_ad(self):
    P = pasp.parse("examples/earthquake_ad.plp")
    R = pasp.exact(P, psemantics = "maxent", quiet = True).flatten()
    # ℙ(alarm | burglary, earthquake(heavy))
    self.assertAlmostEqual(R[0], 0.9)
    # ℙ(alarm | not burglary, earthquake(mild))
    self.assertAlmostEqual(R[1], 0.1)
    # ℙ(alarm | burglary, not earthquake(mild))
    self.assertAlmostEqual(R[2], 0.8058823529411767)
    # ℙ(alarm | not burglary, earthquake(none))
    self.assertAlmostEqual(R[3], 0.0)

if __name__ == "__main__":
  unittest.main()
