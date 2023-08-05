#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "capprox.h"
#include "caseo.h"
#include "cground.h"

static PyObject* _aseo(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *py_P, *py_R = py_P = NULL;
  program_t P = {0};
  double *R = NULL;
  models_t *M = NULL;
  size_t n_samples, scale = 100;
  bool quiet = false, status = true;
  static char *kwlist[] = { "program", "n_samples", "scale", "quiet", "status", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "On|nbb", kwlist, &py_P, &n_samples, &scale,
        &quiet, &status))
    return NULL;

  if (!from_python_program(py_P, &P)) return NULL;
  if (needs_ground(&P)) {
    if (!ground_all(&P, NULL)) goto cleanup;
    if (P.stable) if (!ground_all(P.stable, NULL)) goto cleanup;
  }

  M = aseo(&P, n_samples, MAXENT_SEMANTICS, NULL, (int) scale, approx_rec_query_maxent);
  if (!M) goto cleanup;
  if (!approx_query_maxent(&P, M, &R)) goto cleanup;

  npy_intp dims[3] = {P.Q_n, 1, 0};
  py_R = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, R);
  if (!py_R) goto cleanup;
  PyArray_ENABLEFLAGS((PyArrayObject*) py_R, NPY_ARRAY_OWNDATA);

cleanup:
  models_free(M);
  free_program_contents(&P);
  return py_R;
}

static PyMethodDef CapproxMethods[] = {
  {"aseo", (PyCFunction)(void(*)(void)) _aseo, METH_VARARGS | METH_KEYWORDS,
    "Runs approximate inference using answer set enumeration by optimality."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef approxmodule = {
  PyModuleDef_HEAD_INIT,
  "approx",
  "Approximate inference functions from the C side.",
  -1,
  CapproxMethods,
};

PyMODINIT_FUNC PyInit_approx(void) {
  PyObject *m;

  m = PyModule_Create(&approxmodule);
  if (!m) return NULL;
  import_array();

  return m;
}
