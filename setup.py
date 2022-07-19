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
    os.system("python -m unittest tests/examples.py -b")

optimize = Extension("optimize", sources = ["pasp/optimize.c"])
setup(
  ext_modules = [optimize],
  cmdclass = {"test": TestCommand},
)
