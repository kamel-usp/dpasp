#ifndef _PASP_CMAP
#define _PASP_CMAP

#include "cprogram.h"
#include "carray.h"

typedef struct {
  query_t *Q;
  double *P;
  uint64_t n;
  array_uint64_t C;
} map_mapping_t;

bool init_map_mapping(map_mapping_t *M, query_t *Q);
bool count_map_mapping(map_mapping_t *M, const clingo_model_t *model);
void accumulate_map_mapping(map_mapping_t *M, double u);
uint64_t argmax_map_mapping(map_mapping_t *M, double *p);
bool print_vals_map_mapping(map_mapping_t *M, uint64_t X);
void reset_map_mapping(map_mapping_t *M);
void free_contents_map_mapping(map_mapping_t *M);

#endif
