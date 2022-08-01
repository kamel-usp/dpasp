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

coptimize = Extension("coptimize",
                      depends = ["pasp/coptimize.h"],
                      sources = ["pasp/coptimize.c"])
cutils    = Extension("cutils",
                      libraries = ["clingo"],
                      depends = ["pasp/cutils.h"],
                      sources = ["pasp/cutils.c"])
cprogram  = Extension("cprogram",
                      libraries = ["clingo"],
                      depends = ["pasp/cutils.c", "pasp/cprogram.h"],
                      sources = ["pasp/cprogram.c"])
cexact    = Extension("cexact",
                      libraries = ["m", "clingo"],
                      depends = ["pasp/cprogram.c", "pasp/coptimize.c"],
                      sources = ["pasp/cexact.c"])
carray    = Extension("carray",
                      depends = ["pasp/carray.h"],
                      sources = ["pasp/carray.c"])

setup(
  packages = find_packages(where = ".", include = ["pasp*"]),
  include_package_data = True,
  ext_modules = [coptimize, cutils, cprogram, cexact, carray],
  cmdclass = {"test": TestCommand},
)
