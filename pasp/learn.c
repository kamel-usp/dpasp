#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "clearn.h"
#include "cdata.h"

#include "cground.h"
#include "cprogram.h"

#define ALG_FIXPOINT_S "fixpoint"
#define ALG_LAGRANGE_S "lagrange"
#define ALG_NEURASP_S  "neurasp"

#define DISPLAY_NONE_S          "none"
#define DISPLAY_PROGRESS_S      "progress"
#define DISPLAY_LOGLIKELIHOOD_S "loglikelihood"

static bool prepare_dw(program_t *P) {
  PyObject *py_tensor_dw = NULL;
  PyArrayObject *py_dw = NULL;
  float *dw;

  for (size_t i = 0; i < P->NA_n; ++i) {
    py_tensor_dw = PyObject_GetAttrString(P->NA[i].self, "dw");
    if (!py_tensor_dw) {
      PyErr_SetString(PyExc_AttributeError, "could not access field dw of supposed NeuralRule object!");
      goto cleanup;
    }
    if (P->NA[i].learnable) {
      py_dw = (PyArrayObject*) PyObject_CallMethod(py_tensor_dw, "numpy", NULL);
      if (!py_dw) {
        PyErr_SetString(PyExc_AttributeError, "could not call method numpy in tensor NeuralRule.dw!");
        goto cleanup;
      }
      dw = (float*) PyArray_DATA(py_dw);
    } else dw = NULL;

    P->NA[i].dw = dw;

    Py_DECREF(py_tensor_dw);
    Py_DECREF(py_dw);
    py_tensor_dw = NULL;
    py_dw = NULL;
  }

  return true;
cleanup:
  Py_XDECREF(py_tensor_dw);
  Py_XDECREF(py_dw);
  return false;
}

static PyObject* learn(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t P = {0};
  PyObject *py_P, *py_obs, *py_obs_counts, *py_atoms;
  PyArrayObject *obs, *obs_counts, *atoms;
  bool ok = false;
  bool lstable_sat = true;
  size_t niters = 30;
  const char *alg_s = ALG_FIXPOINT_S, *display_s = DISPLAY_LOGLIKELIHOOD_S;
  uint8_t alg = ALG_LAGRANGE, display = DISPLAY_LOGLIKELIHOOD;
  double eta = 0.1;
  static char *kwlist[] = { "", "", "", "", "niters", "alg", "lr", "lstable_sat", "display", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|nsdbs", kwlist, &py_P, &py_obs,
        &py_obs_counts, &py_atoms, &niters, &alg_s, &eta, &lstable_sat, &display_s))
    return NULL;

  if (!PyArray_Check(py_obs) || !PyArray_Check(py_obs_counts) || !PyArray_Check(py_atoms)) {
    PyErr_SetString(PyExc_TypeError, "obs, obs_counts and atoms must be numpy.ndarray types!");
    return NULL;
  }

  obs = (PyArrayObject*) py_obs;
  obs_counts = (PyArrayObject*) py_obs_counts;
  atoms = (PyArrayObject*) py_atoms;

  npy_intp *obs_dims = PyArray_DIMS(obs);
  size_t atoms_n = (size_t) PyArray_SIZE(atoms);

  if ((PyArray_NDIM(obs) != 2) || (PyArray_NDIM(obs_counts) != 1) || (PyArray_NDIM(atoms) != 1) ||
      (atoms_n != (size_t) obs_dims[1]) || (PyArray_SIZE(obs_counts) != obs_dims[0]) ||
      (atoms_n == 0)) {
    PyErr_SetString(PyExc_ValueError, "unexpected size dimension for obs, obs_counts and/or atoms "
        "in learn!");
    goto cleanup;
  }

  {
    PyArray_Descr *obs_counts_descr = PyArray_DESCR(obs_counts);
    if (PyArray_TYPE(obs) != NPY_UINT8) {
      PyErr_SetString(PyExc_TypeError, "obs must be a numpy.ndarray of type uint8!");
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
    return NULL;
  }

  if (!strcmp(display_s, DISPLAY_NONE_S)) display = DISPLAY_NONE;
  else if (!strcmp(display_s, DISPLAY_PROGRESS_S)) display = DISPLAY_PROGRESS;
  else if (!strcmp(display_s, DISPLAY_LOGLIKELIHOOD_S)) display = DISPLAY_LOGLIKELIHOOD;
  else {
    PyErr_SetString(PyExc_ValueError, "display must either be \"none\", \"progress\" or \"loglikelihood\"!");
    return NULL;
  }

  if (!from_python_program(py_P, &P)) return NULL;
  if (!prepare_dw(&P)) goto cleanup;

  lstable_sat = lstable_sat && (P.sem == LSTABLE_SEMANTICS);
  switch(alg) {
    case ALG_FIXPOINT:
      if (!learn_fixpoint(&P, obs, obs_counts, atoms, niters, lstable_sat, display)) goto cleanup;
      break;
    case ALG_LAGRANGE:
      if (!learn_lagrange(&P, obs, obs_counts, atoms, niters, eta, lstable_sat, display)) goto cleanup;
      break;
    case ALG_NEURASP:
      if (!learn_neurasp(&P, obs, obs_counts, atoms, niters, eta, lstable_sat, display)) goto cleanup;
      break;
  }

  ok = true;
cleanup:
  free_program_contents(&P);
  return ok ? Py_None : NULL;
}

static PyObject* learn_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t P = {0};
  PyObject *py_P, *py_obs;
  PyArrayObject *obs;
  bool ok = false, free_obs = false;
  bool lstable_sat = true;
  size_t niters = 30, batch = 100;
  const char *alg_s = ALG_FIXPOINT_S, *display_s = DISPLAY_LOGLIKELIHOOD_S;
  uint8_t alg = ALG_FIXPOINT, display = DISPLAY_LOGLIKELIHOOD;
  double eta = 0.1, smooth = 1e-4;
  static char *kwlist[] = { "", "", "niters", "alg", "lr", "batch", "smoothing", "lstable_sat", "display", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|nsdndbs", kwlist, &py_P, &py_obs, &niters,
        &alg_s, &eta, &batch, &smooth, &lstable_sat, &display_s))
    return NULL;

  if (!PyArray_Check(py_obs)) {
    if (!ll2array(py_obs, &obs)) {
      PyErr_SetString(PyExc_TypeError, "obs must either be a numpy.ndarray object or a list of lists!");
      return NULL;
    }
    free_obs = true;
  } else obs = (PyArrayObject*) py_obs;

  if (PyArray_NDIM(obs) != 2) {
    PyErr_SetString(PyExc_ValueError, "unexpected size dimension for obs in learn!");
    goto cleanup;
  }

  if (PyArray_TYPE(obs) != NPY_STRING) {
    PyErr_SetString(PyExc_TypeError, "atoms must be a numpy.ndarray of type string (not unicode)!");
    return NULL;
  }

  if (!strcmp(alg_s, ALG_FIXPOINT_S)) alg = ALG_FIXPOINT;
  else if (!strcmp(alg_s, ALG_LAGRANGE_S)) alg = ALG_LAGRANGE;
  else if (!strcmp(alg_s, ALG_NEURASP_S)) alg = ALG_NEURASP;
  else {
    PyErr_SetString(PyExc_ValueError, "alg must either be \"fixpoint\", \"lagrange\" or \"neurasp\"!");
    return NULL;
  }

  if (!strcmp(display_s, DISPLAY_NONE_S)) display = DISPLAY_NONE;
  else if (!strcmp(display_s, DISPLAY_PROGRESS_S)) display = DISPLAY_PROGRESS;
  else if (!strcmp(display_s, DISPLAY_LOGLIKELIHOOD_S)) display = DISPLAY_LOGLIKELIHOOD;
  else {
    PyErr_SetString(PyExc_ValueError, "display must either be \"none\", \"progress\" or \"loglikelihood\"!");
    return NULL;
  }

  if (!from_python_program(py_P, &P)) return NULL;
  if (!prepare_dw(&P)) goto cleanup;

  lstable_sat = lstable_sat && (P.sem == LSTABLE_SEMANTICS);
  switch(alg) {
    case ALG_FIXPOINT:
      if (!learn_fixpoint_batch(&P, obs, niters, batch, smooth, lstable_sat, display)) goto cleanup;
      break;
    case ALG_LAGRANGE:
      if (!learn_lagrange_batch(&P, obs, niters, eta, batch, smooth, lstable_sat, display)) goto cleanup;
      break;
    case ALG_NEURASP:
      if (!learn_neurasp_batch(&P, obs, niters, eta, batch, smooth, lstable_sat, display)) goto cleanup;
      break;
  }

  ok = true;
cleanup:
  free_program_contents(&P);
  if (free_obs) Py_XDECREF(obs);
  return ok ? Py_None : NULL;
}

static PyMethodDef ClearnMethods[] = {
  {"learn", (PyCFunction) (void(*)(void)) learn, METH_VARARGS | METH_KEYWORDS,
    "Learns a program given data."},
  {"learn_batch", (PyCFunction) (void(*)(void)) learn_batch, METH_VARARGS | METH_KEYWORDS,
    "Learns a program given data in batch mode."},
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
  import_array();

  return m;
}
