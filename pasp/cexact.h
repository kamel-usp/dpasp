#ifndef _PASP_CEXACT
#define _PASP_CEXACT

#include "cprogram.h"
#include "cinf.h"

#include "../thpool/thpool.h"

typedef struct {
  /* Number of learnable probabilistic facts. */
  size_t n;
  /* Number of learnable annotated disjunctions. */
  size_t m;
  /* Number of models for each learnable probabilistic fact. */
  uint16_t (*F)[2];
  /* Indices of learnable PFs within the global PF array. */
  uint16_t *I_F;
  /* Number of models for each value of each learnable annotated disjunction. */
  uint16_t **A;
  /* Indices of learnable ADs within the global AD array. */
  uint16_t *I_A;
} count_storage_t;

void free_count_storage_contents(count_storage_t *C, bool free_shared);
void free_count_storage(count_storage_t *C);

/* Compute (exactly) query probabilities by exhaustively enumerating all models. */
bool exact_enum(program_t *P, double (*R)[2], bool lstable_sat, psemantics_t psem, bool quiet);
/* Count number of models for each learnable probabilistic fact or annotated disjunction. */
count_storage_t* count_models(program_t *P, bool lstable_sat);

#endif
