#ifndef _PASP_CMODELS
#define _PASP_CMODELS

#include "ctree.h"
#include "cinf.h"
#include "ccounter.h"
#include "../bitvector/bitvector.h"

/** A models struct for dealing with approximate inference. */
typedef struct {
  /** Model count and probability as represented by an array of ctree_t leaves. */
  ctree_t **L;
  /** Upper bound of models. */
  size_t M;
  /** Number of observations/queries. */
  size_t n;
  /** Actual number of models. */
  size_t m;
  /** Counters for approximate inference. */
  counter_t C;
  /** Root of ctree_t. */
  ctree_t root;
} models_t;

/** Initialize models, where m is the upper bound of models in M, n is the number of queries or
 * observations and obs indicates whether to use the observation or query format. */
bool models_init(models_t *M, size_t m, size_t n, bool obs);
/** Create models, where m is the number of models in M, n is the number of queries or observations
 * and psem representing the probabilistic semantics being used. */
models_t* models_create(size_t m, size_t n, bool obs);

/** Frees models. */
void models_free(models_t *M);
/** Frees the contents of models. */
void models_free_contents(models_t *M);

/** Counts the number of true's for the i-th observation/query up to model j. */
#define models_sum_up_to(M, i, j) counter_sum_up_to(&(M)->C, i, j)
/** Counts the number of true's for the i-th observation/query. */
#define models_sum(M, i) counter_sum(&(M)->C, i)

/** Sets to true the i-th observation/query bit of model j. */
#define models_TRUE(M, i, j) counter_SET(&(M)->C, i, j, true)
/** Sets to false the i-th observation/query bit of model j. */
#define models_FALSE(M, i, j) counter_SET(&(M)->C, i, j, false)

/** Gets the bit value of the i-th observation/query and model j. */
#define models_GET(M, i, j) counter_GET(&(M)->C, i, j)

/** Returns the probability of the i-th observation/query for model j normalized by the number of
 * total observed models in that total choice. */
#define models_prob(M, i, j) counter_GET(&(M)->C, i, j)*(M)->L[j]->pr/(M)->L[j]->n

#endif
