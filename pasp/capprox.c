#include "capprox.h"

#include "cinf.h"

bool approx_rec_obs_maxent(const clingo_model_t *cM, program_t *P, models_t *M, size_t model_idx,
    observations_t *O, clingo_control_t *C) {
  bool ok = false;
  for (size_t i = 0; i < O->n; ++i) {
    for (size_t j = 0; j < O->m; ++j) {
      bool contains_atom;
      if (O->dense) {
        if (!O->V[i][j]) break;
        if (!clingo_model_contains(cM, O->V[i][j], &contains_atom)) goto cleanup;
      } else {
        if (O->S[i][j] == OBSERVATION_MIS) continue;
        if (!clingo_model_contains(cM, O->A[j], &contains_atom)) goto cleanup;
      }
      if (contains_atom != O->S[i][j]) goto next_obs;
    }
    /* Consistent, therefore record as such. */
    models_TRUE(M, i, model_idx);
next_obs: ;
  }
  ok = true;
cleanup:
  return ok;
}

bool approx_rec_query_maxent(const clingo_model_t *cM, program_t *P, models_t *M, size_t model_idx,
    observations_t *O, clingo_control_t *C) {
  bool ok = false, is_partial = P->sem != STABLE_SEMANTICS;

  for (size_t i = 0; i < P->Q_n; ++i) {
    query_t *q = P->Q+i;
    bool all_e = true, all_q = true, c;
    for (size_t j = 0; j < q->E_n; ++j) {
      if (!model_contains(cM, q, j, &c, MODEL_CONTAINS_EVI, is_partial)) goto cleanup;
      if (!c) { all_e = false; break; }
    }
    if (!all_e) continue;
    for (size_t j = 0; j < q->Q_n; ++j) {
      if (!model_contains(cM, q, j, &c, MODEL_CONTAINS_QUERY, is_partial)) goto cleanup;
      if (!c) { all_q = false; break; }
    }
    models_TRUE(M, i+M->n, model_idx); /* count_e is set to the second half of M->Q. */
    if (all_q) models_TRUE(M, i, model_idx);  /* count_q_e is set to the first half of M->Q. */
  }

  ok = true;
cleanup:
  return ok;
}

bool approx_query_credal(const clingo_model_t *cM, program_t *P, models_t *M, observations_t *O,
    clingo_control_t *C) {
  /* Approximate credal inference not yet implemented. */
  return false;
}

bool approx_query_maxent(program_t *P, models_t *M, double **R) {
  bool ok = false;
  double *a, *b, *r = b = a = NULL;
  size_t n = M->n;

  /* The resulting (flattened) array has dimension n*k x 1, where n is the number of queries and k
   * is the number of examples in the neural dataset. */
  r = (double*) malloc(M->n*sizeof(double));
  if (!r) goto cleanup;
  /* Arrays a and b are the cumulated probabilities for each query. */
  a = (double*) calloc(M->n, sizeof(double));
  if (!a) goto cleanup;
  b = (double*) calloc(M->n, sizeof(double));
  if (!b) goto cleanup;

  /*printf("n = %lu, m = %lu\n", n, M->m);*/
  for (size_t i = 0; i < n; ++i) {
    size_t l = i+n;
    for (size_t j = 0; j < M->m; ++j) {
      a[i] += models_prob(M, i, j);
      /*printf("a[%lu] += %d * %f / %lu = %f = %f\n", i, counter_GET(&M->C, i, j), M->L[j]->pr, M->L[j]->n,*/
          /*counter_GET(&M->C, i, j)*M->L[j]->pr/M->L[j]->n, a[i]);*/
      b[i] += models_prob(M, l, j);
      /*printf("b[%lu] += %d * %f / %lu = %f %f\n", i, counter_GET(&M->C, l, j), M->L[j]->pr, M->L[j]->n,*/
          /*counter_GET(&M->C, l, j)*M->L[j]->pr/M->L[j]->n, b[i]);*/
    }
    r[i] = a[i]/b[i];
    /*printf("  a[%lu] * b[%lu] = %f\n", i, i, r[i]);*/
  }

  *R = r;

  ok = true;
cleanup:
  free(a); free(b);
  if (!ok) free(r);
  return ok;
}
