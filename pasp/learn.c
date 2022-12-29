#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "clearn.h"

#include "ground.h"
#include "cprogram.h"

#define ALG_FIXPOINT_S "fixpoint"
#define ALG_LAGRANGE_S "lagrange"
#define ALG_NEURASP_S  "neurasp"

#define ALG_FIXPOINT 0
#define ALG_LAGRANGE 1
#define ALG_NEURASP  2

static PyObject* learn(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t P = {0};
  PyObject *py_P, *py_obs, *py_obs_counts, *py_atoms;
  PyArrayObject *obs, *obs_counts, *atoms;
  bool ok = false;
  bool lstable_sat = true;
  size_t niters = 30;
  const char *alg_s = ALG_FIXPOINT_S;
  uint8_t alg = ALG_FIXPOINT;
  static char *kwlist[] = { "", "", "", "", "niters", "alg", "lstable_sat", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|nsb", kwlist, &py_P, &py_obs,
        &py_obs_counts, &py_atoms, &niters, &alg_s, &lstable_sat))
    return NULL;

  if (!PyArray_Check(py_obs) || !PyArray_Check(py_obs_counts) || !PyArray_Check(py_atoms)) {
    PyErr_SetString(PyExc_TypeError, "obs, obs_counts and atoms must be numpy.ndarray types!");
    return NULL;
  }

  obs = (PyArrayObject*) py_obs;
  obs_counts = (PyArrayObject*) py_obs_counts;
  atoms = (PyArrayObject*) py_atoms;

  {
    PyArray_Descr *obs_counts_descr = PyArray_DESCR(obs_counts);
    if (PyArray_TYPE(obs) != NPY_BOOL) {
      PyErr_SetString(PyExc_TypeError, "obs must be a numpy.ndarray of type bool!");
      return NULL;
    } if (obs_counts_descr->kind != 'i' && obs_counts_descr->kind != 'u') {
      PyErr_SetString(PyExc_TypeError, "obs_counts must be a numpy.ndarray of type int!");
      return NULL;
    } if (PyArray_TYPE(atoms) != NPY_STRING) {
      PyErr_SetString(PyExc_TypeError, "atoms must be a numpy.ndarray of type string (not unicode)!");
      return NULL;
    }
  }

  if (!strcmp(alg_s, ALG_FIXPOINT_S)) alg = ALG_FIXPOINT;
  else if (!strcmp(alg_s, ALG_LAGRANGE_S)) alg = ALG_LAGRANGE;
  else if (!strcmp(alg_s, ALG_NEURASP_S)) alg = ALG_NEURASP;
  else {
    PyErr_SetString(PyExc_ValueError, "alg must either be \"fixpoint\", \"lagrange\" or \"neurasp\"!");
    goto cleanup;
  }

  if (!from_python_program(py_P, &P)) return NULL;

  if (needs_ground(&P)) {
    if (!ground(&P)) goto cleanup;
    if (P.stable) if (!ground(P.stable)) goto cleanup;
  }

  switch(alg) {
    case ALG_FIXPOINT:
      if (!learn_fixpoint(&P, obs, obs_counts, atoms, niters, lstable_sat)) goto cleanup;
      break;
    case ALG_LAGRANGE:
      PyErr_SetString(PyExc_NotImplementedError, "Lagrange learning not yet implemented!");
      goto cleanup;
      break;
    case ALG_NEURASP:
      PyErr_SetString(PyExc_NotImplementedError, "NeurASP learning not yet implemented!");
      goto cleanup;
      break;
  }

  ok = true;
cleanup:
  free_program_contents(&P);
  return ok ? Py_None : NULL;
}

static PyMethodDef ClearnMethods[] = {
  {"learn", (PyCFunction) (void(*)(void)) learn, METH_VARARGS | METH_KEYWORDS,
    "Learns a program given data."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef learnmodule = {
  PyModuleDef_HEAD_INIT,
  "learn",
  "Learning functions from the C side.",
  -1,
  ClearnMethods,
};

PyMODINIT_FUNC PyInit_learn(void) {
  PyObject *m;

  m = PyModule_Create(&learnmodule);
  if (!m) return NULL;
  if (import_ground() < 0) return NULL;
  import_array();

  return m;
}