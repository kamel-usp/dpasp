#include "cdata.h"

#include "cutils.h"

#define MAX_ATOM_SIZE 256
#define POS_RULE ":- not %s."
#define NEG_RULE ":- not not %s."
#define MIS_RULE " "

bool obs_to_char(PyArrayObject *obs, size_t j, PyArrayObject *atoms, array_char_t *O) {
  size_t n = (size_t) PyArray_SIZE(atoms);

  for (size_t i = 0; i < n; ++i) {
    uint8_t *o = (uint8_t*) PyArray_GETPTR2(obs, j, i);
    char *a = (char*) PyArray_GETPTR1(atoms, i);
    char rule[MAX_ATOM_SIZE];
    int l = snprintf(rule, MAX_ATOM_SIZE, !(*o) ? NEG_RULE : ((*o == 1) ? POS_RULE : MIS_RULE), a);
    if (l < 0 || l > MAX_ATOM_SIZE) {
      PyErr_SetString(PyExc_ValueError, "could not interpolate atom in observation rule!");
      goto cleanup;
    }
    array_char_writeln(O, rule, 0);
  }

  return true;
cleanup:
  return false;
}

void* observations_atoms(observations_t *O) {
  if (O->dense) return O->V;
  return O->A;
}

bool init_observations(observations_t *O, PyArrayObject *obs, PyArrayObject *atoms) {
  /* Initialize numpy. */
  import_array();
  size_t n = (size_t) PyArray_DIM(obs, 0);
  size_t m = (size_t) PyArray_SIZE(atoms);
  size_t len = (size_t) PyArray_ITEMSIZE(atoms);
  clingo_symbol_t *A = NULL;
  uint8_t **S = NULL;

  A = (clingo_symbol_t*) malloc(m*sizeof(clingo_symbol_t));
  if (!A) goto cleanup;
  for (size_t i = 0; i < m; ++i) {
    char a[MAX_ATOM_SIZE] = {0};
    char *d = (char*) PyArray_GETPTR1(atoms, i);
    /* Replace ';' with ','. */
    size_t j;
    for (j = 0; d[j] && j < len; ++j) a[j] = (d[j] == ';')*',' + (d[j] != ';')*d[j];
    a[j] = '\0';
    if (!clingo_parse_term(a, NULL, NULL, 20, &A[i])) goto clingo_err;
  }

  S = (uint8_t**) malloc(n*sizeof(uint8_t*));
  if (!S) goto cleanup;
  for (size_t i = 0; i < n; ++i) {
    S[i] = (uint8_t*) malloc(m*sizeof(uint8_t));
    if (!S[i]) {
      for (size_t j = 0; j < i; ++j) free(S[j]);
      goto cleanup;
    }
    for (size_t j = 0; j < m; ++j) {
      uint8_t *o = (uint8_t*) PyArray_GETPTR2(obs, i, j);
      S[i][j] = (uint8_t) *o;
    }
  }

  O->A = A; O->S = S;
  O->n = n; O->m = m;
  O->dense = false;

  return true;
clingo_err:
  PyErr_SetString(PyExc_ValueError, "atoms given as observations are not in clingo syntax!");
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
cleanup:
  free(A);
  free(S);
  return false;
}

void free_observations_contents(observations_t *O) {
  free(O->A);
  for (size_t i = 0; i < O->n; ++i) free(O->S[i]);
  free(O->S);
}

void free_observations(observations_t *O) { free_observations_contents(O); free(O); }

bool init_dense_observations(observations_t *O, PyArrayObject *obs, size_t batch) {
  /* Initialize numpy. */
  import_array();
  size_t n = PyArray_DIM(obs, 0);
  size_t m = PyArray_DIM(obs, 1);
  if (batch > n || batch <= 0) batch = n;
  batch = n > batch ? batch : (size_t) n;
  size_t len = (size_t) PyArray_ITEMSIZE(obs);
  clingo_symbol_t **V = NULL;
  uint8_t **S = NULL;

  V = (clingo_symbol_t**) malloc(batch*sizeof(clingo_symbol_t*));
  if (!V) goto nomem;
  S = (uint8_t**) malloc(batch*sizeof(uint8_t*));
  if (!S) goto nomem;
  for (size_t i = 0; i < batch; ++i) {
    V[i] = (clingo_symbol_t*) calloc(m, sizeof(clingo_symbol_t));
    if (!V[i]) {
      for (size_t j = 0; j < i; ++j) { free(V[j]); free(S[j]); }
      goto nomem;
    }
    S[i] = (uint8_t*) malloc(m*sizeof(uint8_t));
    if (!S[i]) {
      for (size_t j = 0; j < i; ++j) { free(V[j]); free(S[j]); }
      free(V[i]);
      goto nomem;
    }
  }
  char a[MAX_ATOM_SIZE] = {0};
  for (size_t i = 0; i < batch; ++i)
    for (size_t j = 0; j < m; ++j) {
      char *d = PyArray_GETPTR2(obs, i, j);
      if (!d[0]) break;
      bool o = d[0] == '~';
      size_t slen = len-o, l;
      d += o;
      /* Replace ';' with ','. */
      for (l = 0; d[l] && l < slen; ++l) a[l] = (d[l] == ';')*',' + (d[l] != ';')*d[l];
      a[l] = '\0';
      if (!clingo_parse_term(a, NULL, NULL, 20, &V[i][j])) goto clingo_err;
      S[i][j] = !o;
    }

  O->V = V; O->S = S;
  O->n = batch; O->m = m;
  O->batch = batch; O->i = 0;
  O->dense = true;

  return true;
nomem:
  free(V); free(S);
  O->V = NULL; O->S = NULL;
  O->batch = 0;
  goto error;
clingo_err:
  PyErr_SetString(PyExc_ValueError, "atoms given as observations are not in clingo syntax!");
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
error:
  return false;
}

bool next_dense_observations(observations_t *O, PyArrayObject *obs) {
  size_t n = PyArray_DIM(obs, 0);
  size_t len = (size_t) PyArray_ITEMSIZE(obs);

  O->i += O->n;

  bool reset = O->i >= n;
  if (reset) O->i = 0;
  size_t next = O->batch + O->i;
  if (next > n) next = n;
  size_t b = next - O->i;

  char a[MAX_ATOM_SIZE] = {0};
  for (size_t i = 0; i < b; ++i)
    for (size_t j = 0; j < O->m; ++j) {
      char *d = PyArray_GETPTR2(obs, O->i+i, j);
      if (!d[0]) {
        memset(O->V[i] + j, 0, (O->m-j)*sizeof(clingo_symbol_t));
        break;
      } else {
        bool o = d[0] == '~';
        size_t slen = len-o, l;
        d += o;
        /* Replace ';' with ','. */
        for (l = 0; d[l] && l < slen; ++l) a[l] = (d[l] == ';')*',' + (d[l] != ';')*d[l];
        a[l] = '\0';
        if (!clingo_parse_term(a, NULL, NULL, 20, &O->V[i][j])) goto clingo_err;
        O->S[i][j] = !o;
      }
    }
  O->n = b;

  return true;
clingo_err:
  PyErr_SetString(PyExc_ValueError, "atoms given as observations are not in clingo syntax!");
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  return false;
}

void free_dense_observations_contents(observations_t *O) {
  for (size_t i = 0; i < O->batch; ++i) { free(O->V[i]); free(O->S[i]); }
  free(O->V); free(O->S);
}

void free_dense_observations(observations_t *O) { free_dense_observations_contents(O); free(O); }

bool ll2array(PyObject *ll, PyArrayObject **obs) {
  if (!PyList_Check(ll)) return false;
  char *data = NULL;
  size_t n = PyList_GET_SIZE(ll);
  size_t m = 0;
  Py_ssize_t c = 0;

  for (size_t i = 0; i < n; ++i) {
    PyObject *l = PyList_GET_ITEM(ll, i);
    if (!PyList_Check(l)) return false;
    size_t k = PyList_GET_SIZE(l);
    for (size_t j = 0; j < k; ++j) {
      Py_ssize_t s;
      PyObject *e = PyList_GET_ITEM(l, j);
      if (!PyUnicode_AsUTF8AndSize(e, &s)) return false;
      c = (s > c)*s + (s <= c)*c;
    }
    m = (k > m)*k + (k <= m)*m;
  }

  data = calloc(n*m*c, sizeof(char));
  if (!data) return false;

  for (size_t i = 0; i < n; ++i) {
    PyObject *l = PyList_GET_ITEM(ll, i);
    size_t k = PyList_GET_SIZE(l);
    for (size_t j = 0; j < k; ++j) {
      Py_ssize_t s;
      PyObject *e = PyList_GET_ITEM(l, j);
      const char *str = PyUnicode_AsUTF8AndSize(e, &s);
      if (!s) goto cleanup;
      memcpy(data + i*m*c + j*c, str, s);
    }
  }

  /* Initialize numpy. */
  import_array();
  npy_intp dims[2] = {n, m};
  *obs = (PyArrayObject*) PyArray_New(&PyArray_Type, 2, dims, NPY_STRING, NULL, data, c,
      NPY_ARRAY_CARRAY_RO, NULL);
  if (!*obs) goto cleanup;
  PyArray_ENABLEFLAGS(*obs, NPY_ARRAY_OWNDATA);

  return true;
cleanup:
  free(data);
  return false;
}
