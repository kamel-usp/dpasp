#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "cexact.h"

#include "cprogram.h"
#include "cground.h"
#include "cinf.h"

static PyObject* exact(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t p = {0};
  PyObject *py_P, *py_R = NULL;
  double *R = NULL;
  bool r = false, parallel = true, lstable_sat = true, quiet = false, status = true;
  const char *psem_arg = "credal";
  static char *kwlist[] = { "", "parallel", "lstable_sat", "psemantics", "quiet", "status", NULL };
  psemantics_t psem = CREDAL_SEMANTICS;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|bbsbb", kwlist, &py_P, &parallel, &lstable_sat,
        &psem_arg, &quiet, &status))
    return NULL;

  if (!strcmp(psem_arg, "maxent")) { psem = MAXENT_SEMANTICS; }
  else if (strcmp(psem_arg, "credal")) {
    PyErr_SetString(PyExc_ValueError, "psemantics must either be \"credal\" or \"maxent\"!");
    goto cleanup;
  }

  if (!from_python_program(py_P, &p)) return NULL;

  if (psem == MAXENT_SEMANTICS && p.CF_n > 0) {
    PyErr_SetString(PyExc_ValueError, "cannot have MaxEntropy semantics together with credal facts!");
    goto cleanup;
  }

  if (needs_ground(&p)) {
    if (!ground_all(&p, NULL)) goto cleanup;
    if (p.stable) if (!ground_all(p.stable, NULL)) goto cleanup;
  }

  lstable_sat = lstable_sat && (p.sem == LSTABLE_SEMANTICS);
  if (!exact_enum(&p, &R, lstable_sat, psem, quiet, status)) goto cleanup;

  /* Return result as a numpy array. */
  bool has_neural = p.NR_n + p.NA_n > 0;
  int nd;
  npy_intp dims[3];
  if (has_neural) {
    nd = 3;
    dims[0] = p.m_test;
    dims[1] = p.Q_n; dims[2] = psem == MAXENT_SEMANTICS ? 1 : 2;
  } else {
    nd = 2;
    dims[0] = p.Q_n;
    dims[1] = psem == MAXENT_SEMANTICS ? 1 : 2;
  }
  py_R = PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, R);
  if (!py_R) goto cleanup;
  PyArray_ENABLEFLAGS((PyArrayObject*) py_R, NPY_ARRAY_OWNDATA);

  r = true;
  goto cleanup;
cleanup:
  free_program_contents(&p);
  return r ? py_R : NULL;
}

static inline PyObject* count(PyObject *self, PyObject *args, PyObject *kwargs) {
  program_t P = {0};
  PyObject *py_P = NULL;
  count_storage_t C = {0};
  bool ok = false;
  bool lstable_sat = true;
  static char *kwlist[] = { "", "lstable_sat", NULL };
  PyObject *py_F, *py_I_F, *py_A, *py_I_A = py_A = py_I_F = py_F = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|b", kwlist, &py_P, &lstable_sat)) return NULL;
  if (!from_python_program(py_P, &P)) return NULL;

  lstable_sat = lstable_sat && (P.sem == LSTABLE_SEMANTICS);
  if (!count_models(&P, lstable_sat, &C)) goto cleanup;

  npy_intp dims[2] = {C.n, 2};
  if (C.n > 0) {
    py_F = PyArray_SimpleNewFromData(2, dims, NPY_UINT16, C.F);
    if (!py_F) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_F, NPY_ARRAY_OWNDATA);
    py_I_F = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C.I_F);
    if (!py_I_F) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_I_F, NPY_ARRAY_OWNDATA);
  }
  if (C.m > 0) {
    py_A = PyTuple_New(C.m);
    if (!py_A) goto cleanup;
    for (size_t i = 0; i < C.m; ++i) {
      dims[0] = P.AD[C.I_A[i]].n;
      PyObject *py_A_i = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C.A[i]);
      if (!py_A_i) goto cleanup;
      PyArray_ENABLEFLAGS((PyArrayObject*) py_A_i, NPY_ARRAY_OWNDATA);
      PyTuple_SET_ITEM(py_A, i, py_A_i);
    }
    dims[0] = C.m;
    py_I_A = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C.I_A);
    if (!py_I_A) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_I_A, NPY_ARRAY_OWNDATA);
  }

  ok = true;
cleanup:
  free_program_contents(&P);
  if (!ok) {
    free_count_storage_contents(&C, true);
    Py_XDECREF(py_F); Py_XDECREF(py_I_F);
    Py_XDECREF(py_A); Py_XDECREF(py_I_A);
    return Py_None;
  }
  return Py_BuildValue("OOOO", py_F ? py_F : Py_None, py_I_F ? py_I_F : Py_None,
      py_A ? py_A : Py_None, py_I_A ? py_I_A : Py_None);
}

static PyMethodDef CexactMethods[] = {
  {"exact", (PyCFunction)(void(*)(void)) exact, METH_VARARGS | METH_KEYWORDS,
    "Runs exact inference in order to answer the queries in `P`."},
  {"count", (PyCFunction)(void(*)(void)) count, METH_VARARGS | METH_KEYWORDS,
    "Counts the number of models for each possible learnable fact or annotated disjunction."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef exactmodule = {
  PyModuleDef_HEAD_INIT,
  "exact",
  "Exact inference functions from the C side.",
  -1,
  CexactMethods,
};

PyMODINIT_FUNC PyInit_exact(void) {
  PyObject *m;

  m = PyModule_Create(&exactmodule);
  if (!m) return NULL;
  import_array();

  return m;
}
