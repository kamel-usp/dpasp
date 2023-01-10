#include "cdata.h"

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
    memcpy(a, (char*) PyArray_GETPTR1(atoms, i), len);
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

  return true;
clingo_err:
  PyErr_SetString(PyExc_ValueError, "atoms given as observations are not in clingo syntax!");
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
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
