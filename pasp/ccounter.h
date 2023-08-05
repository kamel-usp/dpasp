#ifndef _PASP_CCOUNTER
#define _PASP_CCOUNTER

#include "../bitvector/bitvector.h"

/** Counts the number of true and false occurrences for each observation/queries and models. */
typedef struct {
  /** Bitvector indicating whether observation/query holds for each model. */
  bitvec_t *c;
  /** Number of observations/queries. */
  size_t n;
} counter_t;

/** Creates a new counter_t for n observations/queries over k models. */
counter_t* counter_create(size_t n, size_t k);
/** Initializes C to a new counter_t for n observations/queries over k models. */
bool counter_init(counter_t *C, size_t n, size_t k);

/** Frees the content of counter. */
void counter_free_contents(counter_t *C);
/** Frees counter. */
void counter_free(counter_t *C);

/** Sets the i-th observation/query for the j-th model to v. */
#define counter_set(C, i, j, v) bitvec_set(&(C)->c[i], j, v)
/** Sets the i-th observation/query for the j-th model to v with no error checking. */
#define counter_SET(C, i, j, v) bitvec_SET(&(C)->c[i], j, v)

/** Sets v to the i-th observation/query for the j-th model to v. */
#define counter_get(C, i, j, v) bitvec_get(&(C)->c[i], j, v)
/** Returns the i-th observation/query for the j-th model to v with no error checking. */
#define counter_GET(C, i, j) bitvec_GET(&(C)->c[i], j)

/** Counts the number of true values for each observation/query up to the j-th model. */
#define counter_sum_up_to(C, i, j) bitvec_sum_up_to(&(C)->c[i], j)
/** Counts the number of true values for each observation/query. */
#define counter_sum(C, i) bitvec_sum(&(C)->c[i], j)

#endif
