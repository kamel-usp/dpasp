#ifndef _PASP_CLEARN
#define _PASP_CLEARN

#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include "cexact.h"

#define ALG_FIXPOINT 0
#define ALG_LAGRANGE 1
#define ALG_NEURASP  2

#define DISPLAY_NONE          0
#define DISPLAY_PROGRESS      1
#define DISPLAY_LOGLIKELIHOOD 2

bool learn_fixpoint(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, bool lstable_sat, uint8_t display);
bool learn_lagrange(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat, uint8_t display);
bool learn_neurasp(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat, uint8_t display);

bool learn_fixpoint_batch(program_t *P, PyArrayObject *obs, size_t niters, size_t batch,
    double smooth, bool lstable_sat, uint8_t display);
bool learn_lagrange_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta, size_t batch,
    double smooth, bool lstable_sat, uint8_t display);
bool learn_neurasp_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta, size_t batch,
    double smooth, bool lstable_sat, uint8_t display);

bool update_program_parameters(program_t *P, prob_storage_t *Q);

#endif
