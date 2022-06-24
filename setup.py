from setuptools import setup
from Cython.Build import cythonize

setup(
  name = "Probabilistic ASP",
  ext_modules = cythonize("src/pasp/__init__.py"),
  zip_safe = False,
)
