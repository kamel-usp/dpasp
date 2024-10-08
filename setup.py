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

# Debug concurrency problems by forcing sequential running.
# STD_MACROS = [("NUM_PROCS", str(1)), ("_GNU_SOURCE", None)]
STD_MACROS = [("NUM_PROCS", str(nproc-1 if (nproc := os.cpu_count()) > 1 else nproc)), \
              ("_GNU_SOURCE", None)]

exact    = Extension("exact",
                     libraries = ["m", "clingo", "pthread"],
                     depends = ["pasp/cprogram.c", "pasp/coptimize.c", "pasp/cinf.c",
                                "pasp/cutils.c", "pasp/carray.c", "pasp/cground.c",
                                "pasp/cexact.c", "pasp/cstorage.c", "progressbar/statusbar.c"],
                     sources = ["pasp/exact.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cground.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/coptimize.c",
                                "pasp/carray.c", "pasp/cprogram.c", "pasp/cexact.c",
                                "pasp/cstorage.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = ["-Wno-unused-function"],
                     define_macros = STD_MACROS)

ground   = Extension("ground",
                     libraries = ["clingo", "pthread"],
                     depends = ["pasp/cutils.c", "pasp/cprogram.c", "pasp/cground.c",
                                "pasp/carray.c", "bitvector/bitvector.c", "pasp/cinf.c",
                                "thpool/thpool.c"],
                     sources = ["pasp/ground.c", "pasp/cground.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cprogram.c", "bitvector/bitvector.c",
                                "pasp/cinf.c", "thpool/thpool.c"],
                     include_dirs = [np.get_include()],
                     define_macros = STD_MACROS)

learn    = Extension("learn",
                     libraries = ["clingo", "pthread", "ncurses"],
                     depends = ["pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c", "pasp/carray.c",
                                "pasp/cground.c", "pasp/cexact.c", "pasp/clearn.c", "pasp/cdata.c",
                                "pasp/cstorage.c",
                                "progressbar/progressbar.c", "progressbar/statusbar.c"],
                     sources = ["pasp/learn.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cprogram.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/clearn.c",
                                "pasp/carray.c", "pasp/cdata.c", "pasp/cexact.c",
                                "pasp/cstorage.c",
                                "pasp/coptimize.c", "pasp/cground.c", "progressbar/progressbar.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = ["-Wno-unused-function"],
                     define_macros = STD_MACROS)

sample   = Extension("sample",
                     libraries = ["clingo", "pthread"],
                     depends = ["pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c", "pasp/carray.c",
                                "pasp/cground.c", "pasp/csample.c"],
                     sources = ["pasp/sample.c", "thpool/thpool.c", "pasp/cinf.c", "pasp/cprogram.c",
                                "bitvector/bitvector.c", "pasp/cutils.c", "pasp/csample.c",
                                "pasp/carray.c", "pasp/cground.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = ["-Wno-unused-function"],
                     define_macros = STD_MACROS)

approx   = Extension("approx",
                     libraries = ["clingo", "pthread", "ncurses", "m"],
                     depends = ["pasp/approx.c", "pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cground.c", "pasp/capprox.c",
                                "pasp/cmodels.c", "pasp/cdata.c", "pasp/cstorage.c",
                                "pasp/ctree.c", "pasp/caseo.c", "pasp/ccounter.c",
                                "bitvector/bitvector.c","thpool/thpool.c",
                                "progressbar/statusbar.c"],
                     sources = ["pasp/approx.c", "pasp/cprogram.c", "pasp/cinf.c", "pasp/cutils.c",
                                "pasp/carray.c", "pasp/cground.c", "pasp/capprox.c",
                                "pasp/ccounter.c", "pasp/cmodels.c", "pasp/caseo.c",
                                "pasp/cdata.c", "pasp/ctree.c", "pasp/cstorage.c",
                                "bitvector/bitvector.c", "thpool/thpool.c",
                                "progressbar/statusbar.c"],
                     include_dirs = [np.get_include()],
                     extra_compile_args = ["-Wno-unused-function"],
                     define_macros = STD_MACROS)

setup(
  packages = find_packages(where = ".", include = ["pasp*"]),
  include_package_data = True,
  ext_modules = [exact, ground, learn, sample, approx],
  cmdclass = {"test": TestCommand},
)
