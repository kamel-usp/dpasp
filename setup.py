import os
from setuptools import setup, Extension, find_packages, Command
import numpy as np

class TestCommand(Command):
  description = "Runs unit and functional tests for PASP."
  user_options = []

  def initialize_options(self):
    self.cwd = None

  def finalize_options(self):
    self.cwd = os.getcwd()

  def run(self):
    assert os.getcwd() == self.cwd, f"Must be in package root: {self.cwd}"
    os.system("python setup.py build_ext --inplace && python -m unittest tests/examples.py -b && " \
              "python -m unittest tests/counting.py -b")

exact    = Extension("exact",
                      libraries = ["m", "clingo", "pthread"],
                      depends = ["pasp/cprogram.c", "pasp/coptimize.c", "pasp/cinf.c",
                                 "pasp/cutils.c", "pasp/carray.c", "pasp/ground.c",
                                 "pasp/cexact.c"],
                      sources = ["pasp/exact.c", "thpool/thpool.c", "pasp/cinf.c",
                                 "bitvector/bitvector.c", "pasp/cutils.c", "pasp/coptimize.c",
                                 "pasp/carray.c", "pasp/cprogram.c", "pasp/cexact.c"],
                      include_dirs = [np.get_include()],
                      extra_compile_args = ["-Wno-unused-function"],
                      define_macros = [("NUM_PROCS", str(os.cpu_count())), ("_GNU_SOURCE", None)])
ground   = Extension("ground",
                      libraries = ["clingo"],
                      depends = ["pasp/cutils.c", "pasp/cprogram.c", "pasp/ground.h",
                                 "pasp/carray.c"],
                      sources = ["pasp/ground.c", "pasp/cutils.c", "pasp/carray.c",
                                 "pasp/cprogram.c"])

setup(
  packages = find_packages(where = ".", include = ["pasp*"]),
  include_package_data = True,
  ext_modules = [exact, ground],
  cmdclass = {"test": TestCommand},
)
