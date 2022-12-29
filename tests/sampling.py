import unittest

from .utils import PaspTest, hoeffding
import pasp
import numpy as np
import math

N_SAMPLES  = 2000
"Epsilon for all test cases based on the number of samples."
EPS = hoeffding(N_SAMPLES)

#print("Hoeffding tolerance:", EPS)

class SamplingCases(PaspTest):
  def test_insomnia_ad(self):
    W = [[0.3, 0.2, 0.5], [0.1, 0.6, 0.3], [0.9, 0.1, 0.0], [0.0, 0.5, 0.5]]
    P = pasp.parse("examples/insomnia_ad.lp")
    A = ["insomnia(anna)", "insomnia(bill)", "work(anna)", "work(bill)", "sleep(anna)",
         "sleep(bill)"]
    for w in W:
      P.AD[0].P = w
      R = pasp.exact(P, quiet = True, psemantics = "maxent")
      S = pasp.sample(P, A, n = N_SAMPLES)
      Q = np.sum(S, axis = 0) / N_SAMPLES

      for (p, _), q in zip(R, Q):
        self.assertAlmostEqual(p, q, delta = EPS)

  def test_earthquake(self):
    P = pasp.parse("examples/earthquake.lp")
    A = ["alarm", "burglary", "earthquake"]
    R = pasp.exact(P, quiet = True)
    S = pasp.sample(P, A, n = N_SAMPLES)

    # ℙ(alarm | burglary, earthquake)
    p = np.sum(S[:,0] & S[:,1] & S[:,2]) # ℙ(alarm, burglary, earthquake)
    q = np.sum(S[:,1] & S[:,2])          # ℙ(burglary, earthquake)
    self.assertAlmostEqual(p/q, R[0][0], delta = EPS)

    # ℙ(alarm | ¬burglary, earthquake)
    p = np.sum(S[:,0] & ~S[:,1] & S[:,2]) # ℙ(alarm, ¬burglary, earthquake)
    q = np.sum(~S[:,1] & S[:,2])          # ℙ(¬burglary, earthquake)
    self.assertAlmostEqual(p/q, R[1][0], delta = EPS)

    # ℙ(alarm | burglary, ¬earthquake)
    p = np.sum(S[:,0] & S[:,1] & ~S[:,2]) # ℙ(alarm, burglary, ¬earthquake)
    q = np.sum(S[:,1] & ~S[:,2])          # ℙ(burglary, ¬earthquake)
    self.assertAlmostEqual(p/q, R[2][0], delta = EPS)

    # ℙ(alarm | ¬burglary, ¬earthquake)
    p = np.sum(S[:,0] & ~S[:,1] & ~S[:,2]) # ℙ(alarm, ¬burglary, ¬earthquake)
    q = np.sum(~S[:,1] & ~S[:,2])          # ℙ(¬burglary, ¬earthquake)
    self.assertAlmostEqual(p/q, R[3][0], delta = EPS)

  def test_asia(self):
    P = pasp.parse("examples/asia.lp")
    A = ["trip", "smoking", "tuberculosis", "cancer", "or", "test"]
    R = pasp.exact(P, quiet = True)
    S = pasp.sample(P, A, n = N_SAMPLES)

    # These probabilities are too small for us to reliably approximate conditionals by sampling.
    # Since Hoeffding only holds for the distribution we are sampling from (the joint), we only
    # test against these.

    # ℙ(trip)
    self.assertAlmostEqual(np.sum(S[:,0])/N_SAMPLES, R[0][0], delta = EPS)

    # ℙ(smoking)
    self.assertAlmostEqual(np.sum(S[:,1])/N_SAMPLES, R[4][0], delta = EPS)

  def test_game(self):
    P = pasp.parse("examples/game.lp")
    A = ["wins(b)", "wins(c)"]
    R = pasp.exact(P, quiet = True, psemantics = "maxent")
    S = pasp.sample(P, A, n = N_SAMPLES)

    # ℙ(wins(b))
    self.assertAlmostEqual(np.sum(S[:,0])/N_SAMPLES, R[0][0], delta = EPS)

    # ℙ(wins(c))
    self.assertAlmostEqual(np.sum(S[:,1])/N_SAMPLES, R[1][0], delta = EPS)

  def test_insomnia(self):
    P = pasp.parse("examples/insomnia.lp")
    A = ["insomnia", "work", "sleep"]
    R = pasp.exact(P, quiet = True, psemantics = "maxent")
    S = pasp.sample(P, A, n = N_SAMPLES)

    # ℙ(insomnia)
    self.assertAlmostEqual(np.sum(S[:,0])/N_SAMPLES, R[0][0], delta = EPS)

    # ℙ(work)
    self.assertAlmostEqual(np.sum(S[:,1])/N_SAMPLES, R[1][0], delta = EPS)

    # ℙ(sleep)
    self.assertAlmostEqual(np.sum(S[:,2])/N_SAMPLES, R[2][0], delta = EPS)

  def test_multinsomnia(self):
    P = pasp.parse("examples/multinsomnia.lp")
    A = ["insomnia(anna)", "insomnia(bill)", "insomnia(charlie)",
         "work(anna)", "work(bill)", "work(charlie)",
         "sleep(anna)", "sleep(bill)", "sleep(charlie)",
         "calls(anna, bill)", "calls(anna, charlie)", "calls(bill, anna)", "calls(bill, charlie)",
         "calls(charlie, anna)", "calls(charlie, bill)"]
    R = pasp.exact(P, quiet = True, psemantics = "maxent")
    S = pasp.sample(P, A, n = N_SAMPLES)

    # ℙ(insomnia)
    for i, (r, _) in enumerate(R):
      self.assertAlmostEqual(np.sum(S[:,i])/N_SAMPLES, r, delta = EPS)

  def test_smokers(self):
    P = pasp.parse("examples/smokers.lp")
    A = ["smokes(a)", "smokes(b)"]
    R = pasp.exact(P, quiet = True, psemantics = "maxent")
    S = pasp.sample(P, A, n = N_SAMPLES)

    # ℙ(smokes(a))
    self.assertAlmostEqual(np.sum(S[:,0])/N_SAMPLES, R[0][0], delta = EPS)

    # ℙ(smokes(b))
    self.assertAlmostEqual(np.sum(S[:,1])/N_SAMPLES, R[1][0], delta = EPS)

  def test_earthquake_ad(self):
    P = pasp.parse("examples/earthquake_ad.lp")
    A = ["alarm", "burglary", "earthquake(mild)", "earthquake(none)"]
    R = pasp.exact(P, quiet = True, psemantics = "maxent")
    S = pasp.sample(P, A, n = N_SAMPLES)

    # The probability of earthquake(heavy) is too low for the approximation of
    #   ℙ(alarm | burglary, earthquake(heavy)).
    # We shall skip this one.

    # ℙ(alarm | ¬burglary, earthquake(mild))
    p = np.sum(S[:,0] & ~S[:,1] & S[:,2]) # ℙ(alarm, ¬burglary, earthquake(mild))
    q = np.sum(~S[:,1] & S[:,2])          # ℙ(¬burglary, earthquake(mild))
    self.assertAlmostEqual(p/q, R[1][0], delta = EPS)

    # ℙ(alarm | burglary, ¬earthquake(mild))
    p = np.sum(S[:,0] & S[:,1] & ~S[:,2]) # ℙ(alarm, burglary, ¬earthquake(mild))
    q = np.sum(S[:,1] & ~S[:,2])          # ℙ(burglary, ¬earthquake(mild))
    self.assertAlmostEqual(p/q, R[2][0], delta = EPS)

    # ℙ(alarm | ¬burglary, earthquake(none))
    p = np.sum(S[:,0] & ~S[:,1] & S[:,3]) # ℙ(alarm, ¬burglary, earthquake(none))
    q = np.sum(~S[:,1] & S[:,3])          # ℙ(¬burglary, earthquake(none))
    self.assertAlmostEqual(p/q, R[3][0], delta = EPS)

if __name__ == "__main__":
  unittest.main()
