#ifndef _PASP_CAPPROX
#define _PASP_CAPPROX

#include <clingo.h>

#include "cprogram.h"
#include "cmodels.h"
#include "cdata.h"
#include "cstorage.h"

bool approx_rec_obs_maxent(const clingo_model_t *cM, program_t *P, models_t *M, size_t model_idx,
    observations_t *O, clingo_control_t *C);
bool approx_rec_query_maxent(const clingo_model_t *cM, program_t *P, models_t *M, size_t model_idx,
    observations_t *O, clingo_control_t *C);

bool approx_obs_maxent(program_t *P, models_t *M, observations_t *O, prob_storage_t *Q, bool derive);
bool approx_query_maxent(program_t *P, models_t *M, double **R);

void approx_query_maxent_ab(program_t *P, models_t *M, double *a, double *b);
void approx_query_maxent_r(program_t *P, models_t *M, double *r, double *a, double *b);

#endif
