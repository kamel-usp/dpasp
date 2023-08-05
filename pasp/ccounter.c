#include "ccounter.h"

counter_t* counter_create(size_t n, size_t k) {
  counter_t *C = (counter_t*) malloc(sizeof(counter_t));
  if (!C) return NULL;
  if (!counter_init(C, n, k)) { free(C); return NULL; }
  return C;
}

bool counter_init(counter_t *C, size_t n, size_t k) {
  C->c = (bitvec_t*) malloc(n*sizeof(bitvec_t));
  if (!C->c) return NULL;
  C->n = n;
  for (size_t i = 0; i < n; ++i)
    if (!bitvec_init(&C->c[i], k)) {
      for (size_t j = 0; j < i; ++j) bitvec_free_contents(&C->c[j]);
      return NULL;
    }
  return C;
}

void counter_free_contents(counter_t *C) {
  for (size_t i = 0; i < C->n; ++i) bitvec_free_contents(&C->c[i]);
}
void counter_free(counter_t *C) { counter_free_contents(C); free(C); }
