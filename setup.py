from setuptools import setup, Extension, find_packages

optimize = Extension("optimize", sources = ["pasp/optimize.c"])

setup(ext_modules = [optimize])
