import os
from setuptools import setup, Extension, find_packages, Command
import numpy as np

class TestCommand(Command):
  description = "Runs unit and functional tests for PASP."
  user_options = []
  test_modules = ["examples", "counting", "sampling", "learning", "approx"]

  def initialize_options(self):
    self.cwd = None

  def finalize_options(self):
    self.cwd = os.getcwd()

  def run(self):
    assert os.getcwd() == self.cwd, f"Must be in package root: {self.cwd}"
    cmd = "python -m unittest " + ' '.join(f"tests/{x}.py" for x in TestCommand.test_modules) + " -b"
    os.system(f"python setup.py build_ext --inplace && {cmd}")

def _attribute_environment_var(macros: list, key: str):
  if key not in os.environ: return
  for i, (k, v) in enumerate(macros):
    if k == key:
      macros[i] = (key, os.environ[key])
      break

def check_environment_vars(macros: list) -> dict:
  _attribute_environment_var(macros, "NUM_PROCS")
  return macros

# Debug concurrency problems by forcing sequential running.
# STD_MACROS = [("NUM_PROCS", str(1)), ("_GNU_SOURCE", None)]
STD_MACROS = check_environment_vars(
  [("NUM_PROCS", str(nproc-1 if (nproc := os.cpu_count()) > 1 else nproc)), ("_GNU_SOURCE", None)]
)
EXTRA_COMPILE_FLAGS = ["-Wno-unused-function", "-std=c11"]

exact    = Extension("exact",
                     libraries = ["m", "clingo", "pthread"],
                     depends = ["pasp/cprogram.c", "pasp/coptimize.c", "pasp/cinf.c",
                                "pasp/cutils.c", "pasp/carray.c", "pasp/cground.c",
                                "pasp/cexact.c", "pasp/cstorage.c", "progressbar/statusbar.c",
                                "pasp/cmap.c"],
                     sources = ["pasp/exact.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cground.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/coptimize.c",
                                "pasp/carray.c", "pasp/cprogram.c", "pasp/cexact.c",
                                "pasp/cstorage.c", "pasp/cmap.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = EXTRA_COMPILE_FLAGS,
                     define_macros = STD_MACROS)

ground   = Extension("ground",
                     libraries = ["clingo", "pthread"],
                     depends = ["pasp/cutils.c", "pasp/cprogram.c", "pasp/cground.c",
                                "pasp/carray.c", "bitvector/bitvector.c", "pasp/cinf.c",
                                "pasp/cmap.c", "thpool/thpool.c"],
                     sources = ["pasp/ground.c", "pasp/cground.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cprogram.c", "bitvector/bitvector.c",
                                "pasp/cinf.c", "pasp/cmap.c", "thpool/thpool.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = EXTRA_COMPILE_FLAGS,
                     define_macros = STD_MACROS)

learn    = Extension("learn",
                     libraries = ["clingo", "pthread", "ncurses"],
                     depends = ["pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c", "pasp/carray.c",
                                "pasp/cground.c", "pasp/cexact.c", "pasp/clearn.c", "pasp/cdata.c",
                                "pasp/cstorage.c", "pasp/cmap.c",
                                "progressbar/progressbar.c", "progressbar/statusbar.c"],
                     sources = ["pasp/learn.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cprogram.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/clearn.c",
                                "pasp/carray.c", "pasp/cdata.c", "pasp/cexact.c",
                                "pasp/cstorage.c", "pasp/cmap.c",
                                "pasp/coptimize.c", "pasp/cground.c", "progressbar/progressbar.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = EXTRA_COMPILE_FLAGS,
                     define_macros = STD_MACROS)

sample   = Extension("sample",
                     libraries = ["clingo", "pthread"],
                     depends = ["pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c", "pasp/carray.c",
                                "pasp/cground.c", "pasp/csample.c", "pasp/cmap.c"],
                     sources = ["pasp/sample.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cprogram.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/csample.c",
                                "pasp/carray.c", "pasp/cground.c", "pasp/cmap.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = EXTRA_COMPILE_FLAGS,
                     define_macros = STD_MACROS)

approx   = Extension("approx",
                     libraries = ["clingo", "pthread", "ncurses", "m"],
                     depends = ["pasp/approx.c", "pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cground.c", "pasp/capprox.c",
                                "pasp/cmodels.c", "pasp/cdata.c", "pasp/cstorage.c",
                                "pasp/ctree.c", "pasp/caseo.c", "pasp/ccounter.c", "pasp/cmap.c",
                                "bitvector/bitvector.c","thpool/thpool.c",
                                "progressbar/statusbar.c"],
                     sources = ["pasp/approx.c", "pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cground.c", "pasp/capprox.c",
                                "pasp/ccounter.c", "pasp/cmodels.c", "pasp/caseo.c",
                                "pasp/cdata.c", "pasp/ctree.c", "pasp/cstorage.c", "pasp/cmap.c",
                                "bitvector/bitvector.c", "thpool/thpool.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = EXTRA_COMPILE_FLAGS,
                     define_macros = STD_MACROS)

setup(
  packages = find_packages(where = ".", include = ["pasp*"]),
  include_package_data = True,
  ext_modules = [exact, ground, learn, sample, approx],
  cmdclass = {"test": TestCommand},
)
