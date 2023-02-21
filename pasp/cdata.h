#ifndef _PASP_CDATA
#define _PASP_CDATA

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <clingo.h>

#include <stdbool.h>
#include "carray.h"

#define OBSERVATION_NEG 0
#define OBSERVATION_POS 1
#define OBSERVATION_MIS 2

/* Write constraint restrictions into string O consistent with the observation in obs and atoms.
 * Assumes obs and atoms dimensions and sizes are correct. */
bool obs_to_char(PyArrayObject *obs, size_t j, PyArrayObject *atoms, array_char_t *O);

typedef struct {
  /* Atoms in observations. */
  union {
    clingo_symbol_t *A;
    clingo_symbol_t **V;
  };
  /* Signs of observations. */
  uint8_t **S;
  /* Number of observations. */
  size_t n;
  /* Number of atoms within each observation. */
  size_t m;
  /* Whether in dense or sparse representation. */
  bool dense;
  /* Batch size. Undefined if !dense. */
  size_t batch;
  /* Start index. Undefined if !dense. */
  size_t i;
} observations_t;

void* observations_atoms(observations_t *O);

bool init_observations(observations_t *O, PyArrayObject *obs, PyArrayObject *atoms);
void free_observations_contents(observations_t *O);
void free_observations(observations_t *O);

bool init_dense_observations(observations_t *O, PyArrayObject *obs, size_t batch);
bool next_dense_observations(observations_t *O, PyArrayObject *obs);
void free_dense_observations_contents(observations_t *O);
void free_dense_observations(observations_t *O);

bool ll2array(PyObject *ll, PyArrayObject **obs);

#endif
