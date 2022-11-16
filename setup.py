import os
from setuptools import setup, Extension, find_packages, Command

class TestCommand(Command):
  description = "Runs unit and functional tests for PASP."
  user_options = []

  def initialize_options(self):
    self.cwd = None

  def finalize_options(self):
    self.cwd = os.getcwd()

  def run(self):
    assert os.getcwd() == self.cwd, f"Must be in package root: {self.cwd}"
    os.system("python setup.py build_ext --inplace && python -m unittest tests/examples.py -b")

cprogram  = Extension("cprogram",
                      libraries = ["clingo"],
                      depends = ["pasp/cutils.c", "pasp/cprogram.h", "pasp/carray.c"],
                      sources = ["pasp/cprogram.c", "pasp/cutils.c", "pasp/carray.c"])
cexact    = Extension("cexact",
                      libraries = ["m", "clingo", "pthread"],
                      depends = ["pasp/cprogram.c", "pasp/coptimize.c", "pasp/cinf.c",
                                 "pasp/cutils.c", "pasp/carray.c"],
                      sources = ["pasp/cexact.c", "thpool/thpool.c", "pasp/cinf.c",
                                 "bitvector/bitvector.c", "pasp/cutils.c", "pasp/coptimize.c",
                                 "pasp/carray.c"],
                      extra_compile_args = ["-Wno-unused-function"],
                      define_macros = [("NUM_PROCS", str(os.cpu_count())), ("_GNU_SOURCE", None)])
cground   = Extension("cground",
                      libraries = ["clingo"],
                      depends = ["pasp/cutils.c", "pasp/cprogram.c", "pasp/cground.h",
                                 "pasp/carray.c"],
                      sources = ["pasp/cground.c", "pasp/cutils.c", "pasp/carray.c"])

setup(
  packages = find_packages(where = ".", include = ["pasp*"]),
  include_package_data = True,
  ext_modules = [cprogram, cexact, cground],
  cmdclass = {"test": TestCommand},
)
