from .program import Program

import numpy as np

def learn(P: Program, D: np.ndarray, A: np.ndarray = None, niters: int = 30, alg: str = "fixpoint",
          lr: float = 0.001, batch: int = 100, lstable_sat: bool = True):
  # Check if data dimensions all match.
  def assert_dims(N, e: int):
    if N.learnable and ((o := len(N.train)) != e):
      raise ValueError(f"Training data dimensions do not match!\n  Expected {n}, got {o}.")
  n = len(D)
  for N in P.NR: assert_dims(N, n)
  for N in P.NA: assert_dims(N, n)

  # Batch mode.
  if A is None:
    if type(D) is np.ndarray: data = D if np.issubdtype(D.dtype, bytes) else D.astype(bytes)
    else: data = D
    from learn import learn_batch as clearn_batch
    P.train()
    clearn_batch(P, data, niters = niters, alg = alg, eta = lr, batch = batch, lstable_sat = lstable_sat)
    P.eval()
    return

  # Non-batch mode.
  if type(A) is not np.ndarray: atoms = np.array(A, dtype = bytes)
  else: atoms = A if np.issubdtype(A.dtype, bytes) else A.astype(bytes)
  if type(D) is not np.ndarray: data = np.array(D, dtype = np.uint8)
  else: data = D if np.issubdtype(D.dtype, np.uint8) else D.astype(np.uint8)

  obs, obs_counts = np.unique(data, axis = 0, return_counts = True)
  from learn import learn as clearn
  P.train()
  clearn(P, obs, obs_counts, atoms, niters = niters, alg = alg, eta = lr, lstable_sat = lstable_sat)
  P.eval()
