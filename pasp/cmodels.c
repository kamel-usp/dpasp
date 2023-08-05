#include "cmodels.h"

#include <stdio.h>

bool models_init(models_t *M, size_t m, size_t n, bool obs) {
  if (!counter_init(&M->C, obs ? n : (n << 1), m)) return false;
  M->L = (ctree_t**) malloc(m*sizeof(ctree_t*));
  ctree_init(&M->root);
  if (!M->L) return false;
  M->n = n; M->M = m;
  return true;
}

models_t* models_create(size_t m, size_t n, bool obs) {
  models_t *M = (models_t*) malloc(sizeof(models_t));
  if (!M) return NULL;
  if (!models_init(M, m, n, obs)) { free(M); return NULL; }
  return M;
}

void models_free_contents(models_t *M) {
  if (!M) return;
  counter_free_contents(&M->C);
  ctree_free_contents(&M->root);
  free(M->L);
}
void models_free(models_t *M) { models_free_contents(M); free(M); }
