#ifndef _PASP_CGROUND
#define _PASP_CGROUND

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <clingo.h>
#include <stdbool.h>

#include "carray.h"
#include "cprogram.h"
#include "cexact.h"
#include "cinf.h"

bool partial_update_program(program_t *P, array_char_t *gr_P, array_prob_fact_t_t *gr_PF);
bool ground_all(program_t *P, prob_storage_t *Q);
bool needs_ground(program_t *P);

#endif
