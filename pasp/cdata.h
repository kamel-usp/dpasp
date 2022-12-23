#ifndef _PASP_CDATA
#define _PASP_CDATA

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <clingo.h>

#include <stdbool.h>
#include "carray.h"

/* Write constraint restrictions into string O consistent with the observation in obs and atoms.
 * Assumes obs and atoms dimensions and sizes are correct. */
bool obs_to_char(PyArrayObject *obs, size_t j, PyArrayObject *atoms, array_char_t *O);

typedef struct {
  /* Atoms in observations. */
  clingo_symbol_t *A;
  /* Signs of observations. */
  bool **S;
  /* Number of observations. */
  size_t n;
  /* Number of atoms within each observation. */
  size_t m;
} observations_t;

bool init_observations(observations_t *O, PyArrayObject *obs, PyArrayObject *atoms);
void free_observations_contents(observations_t *O);
void free_observations(observations_t *O);

#endif
