import numpy as np

def learn(P, D: np.ndarray, A: np.ndarray = None, niters: int = 30, alg: str = "fixpoint",
          lr: float = 0.001, batch: int = None, smoothing: float = 1e-4, lstable_sat: bool = True,
          display: str = "loglikelihood"):
  # If batch is not given, set batch to the size of the dataset.
  if batch is None: batch = len(D)
  # Prepare training tensors.
  for N in P.NR: N.prepare_train(batch)
  for N in P.NA: N.prepare_train(batch)
  # Check if data dimensions all match.
  def assert_dims(N, e: int):
    if N.learnable and ((o := len(N.data[0].train)) != e):
      raise ValueError(f"Training data dimensions do not match!\n  Expected {n}, got {o}.")
  n = len(D)
  for N in P.NR: assert_dims(N, n)
  for N in P.NA: assert_dims(N, n)

  # Check if D is an np.ndarray or list.
  if not (issubclass(t := type(D), list) or issubclass(t, np.ndarray)):
    raise TypeError(f"Expected dataset of type list or numpy.ndarray, got {t}!")

  # Batch mode.
  if A is None:
    if type(D) is np.ndarray: data = D if np.issubdtype(D.dtype, bytes) else D.astype(bytes)
    else: data = D
    from learn import learn_batch as clearn_batch
    P.train()
    clearn_batch(P, data, niters = niters, alg = alg, lr = lr, batch = batch,
                 lstable_sat = lstable_sat, display = display, smoothing = smoothing)
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
  clearn(P, obs, obs_counts, atoms, niters = niters, alg = alg, lr = lr, lstable_sat = lstable_sat)
  P.eval()
