#ifndef _PASP_CEXACT
#define _PASP_CEXACT

#include "cdata.h"
#include "cprogram.h"
#include "cinf.h"
#include "cstorage.h"

/* Compute (exactly) query probabilities by exhaustively enumerating all models. */
bool exact_enum(program_t *P, double **R, bool lstable_sat, psemantics_t psem, bool quiet, bool status);
/* Count number of models for each learnable probabilistic fact or annotated disjunction. */
bool count_models(program_t *P, bool lstable_sat, count_storage_t *C);

/* Compute the probability of an observation O (a '\0' terminating const char*), returning the
 * probability ℙ(θ, O) and ℙ(O), where θ covers learnable PFs and ADs. The probabilities are not
 * normalized - e.g. if using the maxent semantic, then these have to be divided by the number of
 * models (i.e. the output of count_models). */
bool prob_obs(program_t *P, observations_t *obs, bool lstables_sat, prob_storage_t *ret, bool derive);
/* Same as prob_obs, but reuse the prob_storage_t's in Q. It's memory safe to assign ret to &Q[0]
 * or NULL; the latter used if the user prefers to access data directly from Q. */
bool prob_obs_reuse(program_t *P, observations_t *obs, bool lstable_sat, prob_storage_t *ret,
    prob_storage_t Q[NUM_PROCS], bool derive, size_t num_procs);

#endif
