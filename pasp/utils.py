import sys
import os
import contextlib
import array
import timeit

import clingo

class DummyStream:
  def write(self, x): pass

_global_dummy_stream = DummyStream()

@contextlib.contextmanager
def suppress_out():
  _stdout = sys.stdout
  sys.stdout = _global_dummy_stream
  yield
  sys.stdout = _stdout

@contextlib.contextmanager
def suppress_ext_out():
  with open(os.devnull, "w") as devnull:
    _stdout = sys.stdout
    sys.stdout = devnull
    try:
      yield
    finally:
      sys.stdout = _stdout

@contextlib.contextmanager
def suppress_err():
  _stderr = sys.stderr
  sys.stderr = _global_dummy_stream
  yield
  sys.stderr = _stderr

@contextlib.contextmanager
def suppress_ext_err():
  with open(os.devnull, "w") as devnull:
    _stderr = sys.stderr
    sys.stderr = devnull
    try:
      yield
    finally:
      sys.stderr = _stderr

@contextlib.contextmanager
def timer_block():
  s = timeit.default_timer()
  try:
    yield
  finally:
    print("Time elapsed: ", timeit.default_timer() - s)

def start_timer() -> float: start_timer.s = timeit.default_timer(); return start_timer.s
def end_timer() -> float: return timeit.default_timer() - start_timer.s

def undef_atom_ignore(x, y):
  if x == clingo.MessageCode.AtomUndefined: return
  print(y, file = sys.stderr)

def new_array(n: int, t: type = 'b', v: int = 0): return array.array(t, [v])*n
def new_list(n: int, v = None): return [v for _ in range(n)]
