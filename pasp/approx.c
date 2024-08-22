#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "capprox.h"
#include "caseo.h"
#include "cground.h"

static bool _aseo_maxent(program_t *P, size_t n_samples, size_t scale, double **R, bool quiet,
    bool status) {
  bool ok = false;
  double *a, *b, *r = a = b = NULL;

  /* Reuse weighted literals for ASEO. */
  size_t n = num_prob_params(P);
  clingo_weighted_literal_t *W = (clingo_weighted_literal_t*) malloc(n*sizeof(clingo_weighted_literal_t));
  clingo_weighted_literal_t *U = (clingo_weighted_literal_t*) malloc(n*sizeof(clingo_weighted_literal_t));
  if (!(W && U)) goto nomem;

  /* Initialize probabilities in neural components. */
  for (size_t i = 0; i < P->NR_n; ++i)
    if (!update_pr_neural_rule(&P->NR[i])) goto cleanup;
  for (size_t i = 0; i < P->NA_n; ++i)
    if (!update_pr_neural_annot_disj(&P->NA[i])) goto cleanup;

  /* Run first ASEO separately to get the number of models. */
  models_t M = {0};
  if (!aseo_reuse(P, n_samples, MAXENT_SEMANTICS, NULL, (int) scale, 0, W, U, &M,
        approx_rec_query_maxent, false)) {
    models_free_contents(&M);
    goto cleanup;
  }
  size_t n_M = M.n;

  /* The resulting (flattened) array has dimension n*k x 1, where n is the number of queries and k
   * is the number of examples in the neural dataset. */
  size_t m_neural = (P->NA_n + P->NR_n > 0) ? P->m_test : 1;
  r = (double*) malloc(n_M*m_neural*sizeof(double));
  if (!r) goto cleanup;
  *R = r;
  /* Arrays a and b are the cumulated probabilities for each query. */
  a = (double*) calloc(n_M, sizeof(double));
  if (!a) goto cleanup;
  b = (double*) calloc(n_M, sizeof(double));
  if (!b) goto cleanup;

  approx_query_maxent_ab(P, &M, a, b);
  approx_query_maxent_r(P, &M, r, a, b);
  models_free_contents(&M);
  if (!quiet) {
    for (size_t i = 0; i < P->Q_n; ++i) {
      print_query(P->Q+i);
      wprintf(L" = %f\n", r[i]);
    }
    fputws(L"---\n", stdout);
  }
  r += n_M;

  for (size_t i = 1; i < m_neural; ++i) {
    if (!aseo_reuse(P, n_samples, MAXENT_SEMANTICS, NULL, (int) scale, i, W, U, &M,
          approx_rec_query_maxent, status)) {
      models_free_contents(&M);
      goto cleanup;
    }
    memset(a, 0, n_M*sizeof(double));
    memset(b, 0, n_M*sizeof(double));
    approx_query_maxent_ab(P, &M, a, b);
    approx_query_maxent_r(P, &M, r, a, b);
    models_free_contents(&M);
    if (!quiet) {
      for (size_t i = 0; i < P->Q_n; ++i) {
        print_query(P->Q+i);
        wprintf(L" = %f\n", r[i]);
      }
      fputws(L"---\n", stdout);
    }
    r += n_M;
  }

  ok = true;
  goto cleanup;
nomem:
  PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for ASEO!");
cleanup:
  free(W); free(U);
  free(a); free(b);
  return ok;
}

static PyObject* _aseo(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *py_P, *py_R = py_P = NULL;
  program_t P = {0};
  double *R = NULL;
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

  if (!_aseo_maxent(&P, n_samples, scale, &R, quiet, status)) goto cleanup;

  bool has_neural = P.NA_n+P.NR_n > 0;
  npy_intp dims[3] = {P.Q_n, 1, 1};
  if (has_neural) { dims[0] = P.m_test; dims[1] = P.Q_n; dims[2] = 1; }
  py_R = PyArray_SimpleNewFromData(has_neural ? 3 : 2, dims, NPY_DOUBLE, R);
  if (!py_R) goto cleanup;
  PyArray_ENABLEFLAGS((PyArrayObject*) py_R, NPY_ARRAY_OWNDATA);

cleanup:
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
