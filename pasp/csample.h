#ifndef _PASP_CSAMPLE
#define _PASP_CSAMPLE

#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include "cprogram.h"

bool naive_sample(program_t *P, size_t n, PyArrayObject *atoms, bool lstable_sat, PyObject **ret);

#endif
