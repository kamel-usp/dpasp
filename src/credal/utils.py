import sys
import os
import contextlib

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
