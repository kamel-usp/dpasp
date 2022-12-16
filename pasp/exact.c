#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "cexact.h"

#include "cprogram.h"
#include "ground.h"
#include "cinf.h"

#define EXACT_ENUM 0

static PyObject* exact_opt(PyObject *self, PyObject *args, PyObject *kwargs, int choice) {
  program_t p = {0};
  PyObject *py_P, *py_R = NULL;
  double (*R)[2] = NULL;
  size_t i;
  bool r = false, parallel = true, lstable_sat = true;
  const char *psem_arg = "credal";
  static char *kwlist[] = { "", "parallel", "lstable_sat", "psemantics", NULL };
  psemantics_t psem = CREDAL_SEMANTICS;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|bbs", kwlist, &py_P, &parallel, &lstable_sat,
        &psem_arg))
    return NULL;

  if (!strcmp(psem_arg, "maxent")) { psem = MAXENT_SEMANTICS; }
  else if (strcmp(psem_arg, "credal")) {
    PyErr_SetString(PyExc_ValueError, "psemantics must either be \"credal\" or \"maxent\"!");
    goto cleanup;
  }

  if (!from_python_program(py_P, &p)) return NULL;

  R = (double (*)[2]) malloc(p.Q_n*sizeof(*R));
  if (!R) goto cleanup;

  if (needs_ground(&p)) {
    if (!ground(&p)) goto cleanup;
    if (p.stable) if (!ground(p.stable)) goto cleanup;
  }

  if (!exact_enum(&p, R, lstable_sat, psem)) goto badval;

  py_R = PyTuple_New(p.Q_n);
  if (!py_R) {
    PyErr_SetString(PyExc_MemoryError, "could not create new py_R tuple!");
    goto cleanup;
  } for (i = 0; i < p.Q_n; ++i) {
    PyObject *py_R_i = PyTuple_New(2);
    if (!py_R_i) {
      PyErr_SetString(PyExc_MemoryError, "could not create new py_R_i tuple!");
      goto cleanup;
    }
    PyTuple_SET_ITEM(py_R_i, 0, PyFloat_FromDouble(R[i][0]));
    PyTuple_SET_ITEM(py_R_i, 1, PyFloat_FromDouble(R[i][1]));
    PyTuple_SET_ITEM(py_R, i, py_R_i);
  }
  r = true;
  goto cleanup;
badval:
  PyErr_SetString(PyExc_Exception, "clingo or unknown error!");
cleanup:
  free_program_contents(&p);
  free(R);
  if (!r) Py_XDECREF(py_R);
  return r ? py_R : NULL;
}

static inline PyObject* exact(PyObject *self, PyObject *args, PyObject *kwargs) {
  return exact_opt(self, args, kwargs, EXACT_ENUM);
}

static inline PyObject* count(PyObject *self, PyObject *args) {
  program_t P = {0};
  PyObject *py_P = NULL;
  count_storage_t *C = NULL;
  bool ok = false;
  PyObject *py_F, *py_I_F, *py_A, *py_I_A = py_A = py_I_F = py_F = NULL;

  if (!PyArg_ParseTuple(args, "O", &py_P)) return NULL;
  if (!from_python_program(py_P, &P)) return NULL;
  C = count_models(&P, true);

  if (!C) goto cleanup;

  npy_intp dims[2] = {C->n, 2};
  if (C->n > 0) {
    py_F = PyArray_SimpleNewFromData(2, dims, NPY_UINT16, C->F);
    if (!py_F) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_F, NPY_ARRAY_OWNDATA);
    py_I_F = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C->I_F);
    if (!py_I_F) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_I_F, NPY_ARRAY_OWNDATA);
  }
  if (C->m > 0) {
    py_A = PyTuple_New(C->m);
    if (!py_A) goto cleanup;
    for (size_t i = 0; i < C->m; ++i) {
      dims[0] = P.AD[C->I_A[i]].n;
      PyObject *py_A_i = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C->A[i]);
      if (!py_A_i) goto cleanup;
      PyArray_ENABLEFLAGS((PyArrayObject*) py_A_i, NPY_ARRAY_OWNDATA);
      PyTuple_SET_ITEM(py_A, i, py_A_i);
    }
    dims[0] = C->m;
    py_I_A = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, C->I_A);
    if (!py_I_A) goto cleanup;
    PyArray_ENABLEFLAGS((PyArrayObject*) py_I_A, NPY_ARRAY_OWNDATA);
  }

  ok = true;
cleanup:
  free_program_contents(&P);
  if (!ok) {
    if (C) free_count_storage(C);
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
  {"count", (PyCFunction)(void(*)(void)) count, METH_VARARGS,
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
  if (import_ground() < 0) return NULL;
  import_array();

  return m;
}
