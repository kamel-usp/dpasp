#ifndef _PASP_CLEARN
#define _PASP_CLEARN

#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include "cexact.h"

typedef struct {
  /* Number of learnable probabilistic facts. */
  size_t n;
  /* Number of learnable annotated disjunctions. */
  size_t m;
  /* Probabilities for each learnable PF. */
  double (*F)[2];
  /* Probabilities for each learnable AD. */
  double **A;
  /* Indices of learnable PFs within the global PF array. */
  uint16_t *I_F;
  /* Indices of learnable ADs within the global AD array. */
  uint16_t *I_A;
} parameters_t;

void free_parameters_contents(parameters_t *W);
void free_parameters(parameters_t *W);

typedef struct {
  /* Number of learnable probabilistic facts. */
  size_t n;
  /* Number of learnable annotated disjunctions facts. */
  size_t m;
  /* Indices of learnable PFs within the global PF array. */
  uint16_t *F;
  /* Indices of learnable ADs within the global AD array. */
  uint16_t *A;
} indices_t;

bool init_indices(indices_t *I, program_t *P);
void free_indices_contents(indices_t *I);

bool learn_fixpoint(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, bool lstable_sat);
bool learn_lagrange(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat);
bool learn_neurasp(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat);

bool update_program_parameters(program_t *P, indices_t *I);

#endif
