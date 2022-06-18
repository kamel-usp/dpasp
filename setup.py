from setuptools import setup
from Cython.Build import cythonize

setup(
  name = "Credal ASP",
  ext_modules = cythonize("src/credal/__init__.py"),
  zip_safe = False,
)
