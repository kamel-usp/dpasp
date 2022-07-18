#ifdef PASP_DEBUG

#include <stdio.h>
#include <stdlib.h>

#else

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#endif

#define bool char

/* The polynomial to evaluate, where X are the variables, S are the signs of each factor, C are the
 * coefficients, n are the number of terms and m are the number of variables. For example, the
 * following polynomial
 *
 *   f(x, y, z) = 0.2*(1-x)*(1-y)*z+0.4*(1-x)*y*(1-z)+0.3*x*(1-y)*(1-z)+0.5*x*y*(1-z)
 *
 * under the evaluation f(0.2, 0.5, 0.7) would be represented as
 *
 *   X = {0.2, 0.5, 0.7}
 *   S = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0}
 *   C = {0.2, 0.4, 0.3, 0.5}
 *   n = 4
 *   m = 3
 *
 * Note that |X| = m, |S| = n*m, and |C| = n.
 */
static double f(double *X, bool *S, double *C, int n, int m) {
  int i, j, u;
  double s, y;
  for (i = s = 0; i < n; ++i) {
    y = 1;
    for (j = 0; j < m; ++j) {
      u = m*i+j;
      y *= S[u] ? X[j] : 1-X[j];
    }
    s += C[i]*y;
  }
  return s;
}

#define OPTIMIZE_BFCA     0
#define OPTIMIZE_BF       1
#define OPTIMIZE_BFCA_SMP 2
#define OPTIMIZE_BF_SMP   3

#define OPTIMIZE_IS_BF(x)      (x) & 1
#define OPTIMIZE_IS_BFCA(x)    !(OPTIMIZE_IS_BF(x))
#define OPTIMIZE_IS_SMP(x)     (x) & 2
#define OPTIMIZE_IS_NOT_SMP(x) !(OPTIMIZE_IS_SMP(x))

#define BFCA_MAXIMIZE -1
#define BFCA_MINIMIZE 1

#define bfca_min(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, tries, smp)
#define bfca_max(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, tries, smp)

/* Brute-force coordinate descent.
 *
 * Array X are the coordinates to optimize, S_i, C_i, n_i, m - where i ∈ {a, b} are the polynomials
 * that compose the objective function g(X)=a(X)/(a(X)+b(X)) to be optimized, L and U are the lower
 * and upper probabilities respectively of the credal facts. Integer maxmin ∈ {-1, 1} and defines
 * whether to minimize or maximize the objective function (prefer BFCA_MINIMIZE and BFCA_MAXIMIZE
 * instead). Parameter tries tells the algorithm how many initialization resets are to be tried
 * for finding possibly global optima, and smp determines (if true) that the function should
 * override the objective function with g(X)=a(X).
 */
static double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    int n_a, int n_b, int m, int maxmin, int tries, bool smp) {
  double est, lest, best = 1;
  double a_l, a_u, b_l, b_u, l, u;
  int i, t, r;
  for (t = 0; t < tries; ++t) {
    r = rand() % (1 << m);
    for (i = 0; i < m; ++i) X[i] = (r >> i) % 2 ? L[i] : U[i];
    est = 0;
    lest = -1;
    while (est > lest) {
      for (i = 0; i < m; ++i) {
        if (smp) {
          X[i] = L[i];
          l = maxmin*f(X, S_a, C_a, n_a, m);
          X[i] = U[i];
          u = maxmin*f(X, S_a, C_a, n_a, m);
        } else {
          X[i] = L[i];
          a_l = f(X, S_a, C_a, n_a, m);
          b_l = f(X, S_b, C_b, n_b, m);
          X[i] = U[i];
          a_u = f(X, S_a, C_a, n_a, m);
          b_u = f(X, S_b, C_b, n_b, m);
          l = a_l+b_l;
          if (l != 0) l = maxmin*(a_l/l);
          u = a_u+b_u;
          if (u != 0) u = maxmin*(a_u/u);
        }
        lest = est;
        if (l < u) {
          X[i] = L[i];
          est = l;
        } else {
          X[i] = U[i];
          est = u;
        }
      }
    }
    if (best > est) best = est;
  }
  return maxmin*best;
}

/* Brute-force.
 *
 * Array X are the coordinates to optimize, S_i, C_i, n_i, m - where i ∈ {a, b} are the polynomials
 * that compose the objective function g(X)=a(X)/(a(X)+b(X)) to be optimized, L and U are the lower
 * and upper probabilities respectively of the credal facts. Parameter low and up are pointers to
 * where the function should store the minimized and maximized values. This function is constrained
 * over 1 ≤ m ≤ 30 (any call above 30 would end up taking too long anyway). Parameter smp
 * determines (if true) that the function should override the objective function with g(X)=a(X)
 */
static void bf(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    int n_a, int n_b, int m, double *low, double *up, bool smp) {
  int i, j;
  long k;
  double a, b, y;

  *low = 1.0; *up = 0.0;
  k = 1 << m;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < m; ++j) X[j] = (i >> j) % 2 ? L[j] : U[j];
    if (smp) {
      a = f(X, S_a, C_a, n_a, m);
      b = f(X, S_b, C_b, n_b, m);
      if (*low > a) *low = a;
      if (*up < b) *up = b;
    } else {
      a = f(X, S_a, C_a, n_a, m);
      b = f(X, S_b, C_b, n_b, m);
      y = a+b;
      if (y != 0) y = a/y;
      if (*low > y) *low = y;
      if (*up < y) *up = y;
    }
  }
}

#ifndef PASP_DEBUG

static PyObject* optimize_opt(PyObject *self, PyObject *args, int choice) {
  double min, max;
  PyObject *py_S_a, *py_S_b, *py_C_a, *py_C_b, *py_L, *py_U;
  bool *S_a, *S_b, smp;
  double *X, *C_a, *C_b, *L, *U;
  int n_a, n_b, m, t, i, j;

  smp = OPTIMIZE_IS_SMP(choice);

  /* Parse arguments. */
  if (OPTIMIZE_IS_BFCA(choice)) {
    if (!PyArg_ParseTuple(args, "OOOOOOi", &py_S_a, &py_S_b, &py_C_a, &py_C_b, &py_L, &py_U, &t))
      return NULL;
  } else {
    if (!PyArg_ParseTuple(args, "OOOOOO", &py_S_a, &py_S_b, &py_C_a, &py_C_b, &py_L, &py_U))
      return NULL;
  }

  /* Convert S_a, S_b, C_a, C_b and B into PySequence_Fast types. */
  py_S_a = PySequence_Fast(py_S_a, "argument S_a must either be a list or a tuple!");
  if (!py_S_a) return NULL;
  py_C_a = PySequence_Fast(py_C_a, "argument C_a must either be a list or a tuple!");
  if (!py_C_a) { Py_DECREF(py_S_a); return NULL; }
  py_L = PySequence_Fast(py_L, "argument L must either be a list or a tuple!");
  if (!py_L) { Py_DECREF(py_S_a); Py_DECREF(py_C_a); return NULL; }
  py_U = PySequence_Fast(py_U, "argument U must either be a list or a tuple!");
  if (!py_U) { Py_DECREF(py_S_a); Py_DECREF(py_C_a); Py_DECREF(py_L); return NULL; }
  py_S_b = PySequence_Fast(py_S_b, "argument S_b must either be a list or a tuple!");
  if (!py_S_b) { Py_DECREF(py_S_a); Py_DECREF(py_C_a); Py_DECREF(py_L); Py_DECREF(py_U); return NULL; }
  py_C_b = PySequence_Fast(py_C_b, "argument C_b must either be a list or a tuple!");
  if (!py_C_b) { Py_DECREF(py_S_a); Py_DECREF(py_C_a); Py_DECREF(py_L); Py_DECREF(py_U); Py_DECREF(py_S_b); return NULL; }

#define OPTIMIZE_CLEAR_ALL Py_DECREF(py_S_a); Py_DECREF(py_C_a); Py_DECREF(py_L); Py_DECREF(py_U); Py_DECREF(py_S_a); Py_DECREF(py_C_b)

  /* Get lengths and verify if their correctness. */
  m = PySequence_Fast_GET_SIZE(py_L);
  if (PySequence_Fast_GET_SIZE(py_U) != m) {
    OPTIMIZE_CLEAR_ALL;
    PyErr_SetString(PyExc_ValueError, "it must be true that |L| = |U|!");
    return NULL;
  }
  n_a = PySequence_Fast_GET_SIZE(py_C_a);
  if (PySequence_Fast_GET_SIZE(py_S_a) != n_a * m) {
    OPTIMIZE_CLEAR_ALL;
    PyErr_SetString(PyExc_ValueError, "it must be true that |S_a| = |C_a| * |L|!");
    return NULL;
  }
  n_b = PySequence_Fast_GET_SIZE(py_C_b);
  if (PySequence_Fast_GET_SIZE(py_S_b) != n_b * m) {
    OPTIMIZE_CLEAR_ALL;
    PyErr_SetString(PyExc_ValueError, "it must be true that |S_b| = |C_b| * |L|!");
    return NULL;
  }

  /* Allocate C arrays. */
  L = (double*) malloc(m*sizeof(double));
  if (!L) { OPTIMIZE_CLEAR_ALL; return PyErr_NoMemory(); }
  U = (double*) malloc(m*sizeof(double));
  if (!U) { OPTIMIZE_CLEAR_ALL; free(L); return PyErr_NoMemory(); }
  C_a = (double*) malloc(n_a*sizeof(double));
  if (!C_a) { OPTIMIZE_CLEAR_ALL; free(L); free(U); return PyErr_NoMemory(); }
  S_a = (bool*) malloc(n_a*m*sizeof(bool));
  if (!S_a) { OPTIMIZE_CLEAR_ALL; free(L); free(U); free(C_a); return PyErr_NoMemory(); }
  X = (double*) malloc(m*sizeof(double));
  if (!X) { OPTIMIZE_CLEAR_ALL; free(L); free(U); free(C_a); free(S_a); return PyErr_NoMemory(); }
  C_b = (double*) malloc(n_b*sizeof(double));
  if (!C_b) { OPTIMIZE_CLEAR_ALL; free(L); free(U); free(C_a); free(S_a); free(X); return PyErr_NoMemory(); }
  S_b = (bool*) malloc(n_b*m*sizeof(bool));
  if (!S_b) { OPTIMIZE_CLEAR_ALL; free(L); free(U); free(C_a); free(C_b); free(S_a); free(X); return PyErr_NoMemory(); }

#define OPTIMIZE_FREE_ALL free(L); free(U); free(C_a); free(S_a); free(X); free(S_b); free(C_b)

#define catch_val_error(var, msg) if ((var) == -1 && !PyErr_Occurred()) { \
      OPTIMIZE_CLEAR_ALL; OPTIMIZE_FREE_ALL; PyErr_SetString(PyExc_ValueError, msg); return NULL; \
    }

  /* Get lower and upper probabilities from bounds B. */
  for (i = 0; i < m; ++i) {
    L[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_L, i));
    catch_val_error(L[i], "argument L must be a list (or tuple) of floats!");
    U[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_U, i));
    catch_val_error(U[i], "argument U must be a list (or tuple) of floats!");
  }

  /* Store values from C_a, S_a. */
  for (i = 0; i < n_a; ++i) {
    C_a[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_C_a, i));
    catch_val_error(C_a[i], "argument C_a must be a list of floats!");
    for (j = 0; j < m; ++j) {
      int u = i*m+j;
      S_a[u] = (bool) PyLong_AsLong(PySequence_Fast_GET_ITEM(py_S_a, u));
      catch_val_error(S_a[u], "argument S_a must be a list of bools!");
    }
  }

  /* Store values from C_b, S_b. */
  for (i = 0; i < n_b; ++i) {
    C_b[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_C_b, i));
    catch_val_error(C_b[i], "argument C_b must be a list of floats!");
    for (j = 0; j < m; ++j) {
      int u = i*m+j;
      S_b[u] = (bool) PyLong_AsLong(PySequence_Fast_GET_ITEM(py_S_b, u));
      catch_val_error(S_b[u], "argument S_b must be a list of bools!");
    }
  }

  if (OPTIMIZE_IS_BFCA(choice)) {
    min = bfca_min(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, t, smp);
    max = smp ? bfca_max(X, S_b, S_a, C_b, C_a, L, U, n_b, n_a, m, t, smp) :
                bfca_max(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, t, smp);
  } else bf(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, &min, &max, smp);

  OPTIMIZE_CLEAR_ALL; OPTIMIZE_FREE_ALL;

  return Py_BuildValue("(dd)", min, max);

#undef OPTIMIZE_CLEAR_ALL
#undef OPTIMIZE_FREE_ALL
#undef catch_val_error
}

static inline PyObject* optimize_bfca(PyObject *self, PyObject *args) {
  return optimize_opt(self, args, OPTIMIZE_BFCA);
}
static inline PyObject* optimize_bfca_smp(PyObject *self, PyObject *args) {
  return optimize_opt(self, args, OPTIMIZE_BFCA_SMP);
}
static inline PyObject* optimize_bf(PyObject *self, PyObject *args) {
  return optimize_opt(self, args, OPTIMIZE_BF);
}
static inline PyObject* optimize_bf_smp(PyObject *self, PyObject *args) {
  return optimize_opt(self, args, OPTIMIZE_BF_SMP);
}

static PyMethodDef OptimizeMethods[] = {
  {"bfca", optimize_bfca, METH_VARARGS, "Finds local optima through coordinate ascent."},
  {"bf", optimize_bf, METH_VARARGS, "Finds global optima through brute-force."},
  {"bfca_smp", optimize_bfca_smp, METH_VARARGS, "Finds local optima through coordinate ascent."},
  {"bf_smp", optimize_bf_smp, METH_VARARGS, "Finds global optima through brute-force."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef optimizemodule = {
  PyModuleDef_HEAD_INIT,
  "optimize",
  "Optimize credal polynomial functions.",
  -1,
  OptimizeMethods,
};

PyMODINIT_FUNC PyInit_optimize(void) { return PyModule_Create(&optimizemodule); }

#else

int main() {
  double C_a[] = {0.2, 0.4, 0.3, 0.5};
  double C_b[] = {0.7, 0.1, 0.2, 0.8, 0.3};
  bool S_a[] = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0};
  bool S_b[] = {1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  double X[] = {0, 0, 0};
  double L[] = {0.3, 0.5, 0.0};
  double U[] = {0.7, 0.5, 1.0};
  int n_a = 4, n_b = 5, m = 3;
  double min, max;
  puts("|| Coordinate-ascent ||");
  printf("Minimized value: %f\n", bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, 3));
  printf("Maximized value: %f\n", bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, 3));
  puts("\n|| Brute-force ||");
  bf(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, &min, &max);
  printf("Minimized value: %f\nMaximized value: %f\n", min, max);
  return 0;
}

#endif
