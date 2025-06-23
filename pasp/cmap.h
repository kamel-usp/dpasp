#ifndef _PASP_CMAP
#define _PASP_CMAP

#include "cprogram.h"
#include "carray.h"

typedef struct {
  /* MAP Query. */
  query_t *Q;
  /* Probabilities for all instantiations of the optimization variables. */
  double *P;
  /* Normalizing constant when conditioning. */
  double z;
  /* Number of probs. n=2^m, where m=|X| and X is the set of optimization variables. */
  uint64_t n;
  /* Counting dynamic array. */
  array_uint64_t C;
} map_mapping_t;

bool init_map_mapping(map_mapping_t *M, query_t *Q);
bool count_map_mapping(map_mapping_t *M, const clingo_model_t *model);
void accumulate_map_mapping(map_mapping_t *M, double u, size_t z_counts);
uint64_t argmax_map_mapping(map_mapping_t *M, double *p);
void add_map_mapping(map_mapping_t *dst, map_mapping_t *src);
bool print_vals_map_mapping(map_mapping_t *M, uint64_t X);
void reset_map_mapping(map_mapping_t *M);
void free_contents_map_mapping(map_mapping_t *M);

#endif
