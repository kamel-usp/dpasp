#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "csample.h"
#include "cground.h"
#include "cprogram.h"

static PyObject* sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t P = {0};
  PyObject *py_P, *py_atoms, *ret;
  PyArrayObject *atoms = NULL;
  bool ok = false, free_atoms = false;
  bool lstable_sat = true;
  size_t n = 1;
  static char *kwlist[] = { "", "", "n", "lstable_sat", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|nb", kwlist, &py_P, &py_atoms, &n, &lstable_sat))
    return NULL;

  if (!PyArray_Check(py_atoms)) {
    atoms = (PyArrayObject*) PyArray_FROM_OTF(py_atoms, NPY_STRING, NPY_ARRAY_IN_ARRAY);
    if (!atoms) {
      PyErr_SetString(PyExc_ValueError, "could not parse atoms as a numpy.ndarray!");
      goto cleanup;
    }
    free_atoms = true;
  } else atoms = (PyArrayObject*) py_atoms;

  if ((PyArray_NDIM(atoms) != 1) || (PyArray_SIZE(atoms) == 0)) {
    PyErr_SetString(PyExc_ValueError, "unexpected size dimension for atoms in sample!");
    goto cleanup;
  }

  if ((PyArray_TYPE(atoms) != NPY_STRING) && (PyArray_TYPE(atoms) != NPY_UNICODE)) {
    PyErr_SetString(PyExc_TypeError, "atoms must be a numpy.ndarray of type string or unicode!");
    goto cleanup;
  }

  if (!from_python_program(py_P, &P)) goto cleanup;
  if (needs_ground(&P)) {
    if (!ground_all(&P, NULL)) goto cleanup;
    if (P.stable) if (!ground_all(P.stable, NULL)) goto cleanup;
  }

  lstable_sat = lstable_sat && (P.sem == LSTABLE_SEMANTICS);
  if (!naive_sample(&P, n, atoms, lstable_sat, &ret)) goto cleanup;

  ok = true;
cleanup:
  if (free_atoms) Py_DECREF(atoms);
  free_program_contents(&P);
  return ok ? ret : NULL;
}

static PyMethodDef CsampleMethods[] = {
  {"sample", (PyCFunction) (void(*)(void)) sample, METH_VARARGS | METH_KEYWORDS,
    "Samples atoms from a program."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef samplemodule = {
  PyModuleDef_HEAD_INIT,
  "sample",
  "Sampling functions from the C side.",
  -1,
  CsampleMethods,
};

PyMODINIT_FUNC PyInit_sample(void) {
  PyObject *m;

  m = PyModule_Create(&samplemodule);
  if (!m) return NULL;
  import_array();

  return m;
}
