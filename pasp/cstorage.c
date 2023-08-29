#include "cstorage.h"

void free_count_storage_contents(count_storage_t *C, bool free_shared) {
  free(C->F);
  free(C->A);
  if (free_shared) {
    free(C->I_F);
    free(C->I_A);
  }
}
void free_count_storage(count_storage_t *C) { free_count_storage_contents(C, true); free(C); }

bool prob_storage_learnable(prob_storage_t *S) { return S->n || S->m || S->pr || S->nr || S->na; }

size_t init_prob_storage_seq(prob_storage_t Q[NUM_PROCS], program_t *P, observations_t *O) {
  size_t total_choice_n = get_num_facts(P);
  size_t num_procs = estimate_nprocs(total_choice_n + P->AD_n + P->NA_n);
  size_t i = 0;

  for (i = 0; i < num_procs; ++i) {
    if (!init_prob_storage(&Q[i], P, &Q[0], O)) goto cleanup;
    if (!prob_storage_learnable(&Q[i])) {
      PyErr_SetString(PyExc_ValueError, "program is not learnable!");
      goto cleanup;
    }
  }

  return num_procs;
cleanup:
  for (size_t j = 1; j < i; ++j) free_prob_storage_contents(&Q[j], false);
  free_prob_storage_contents(&Q[0], true);
  return 0;
}

void free_prob_storage_contents(prob_storage_t *Q, bool free_shared) {
  for (size_t i = 0; i < Q->o; ++i) {
    free(Q->P[i].F);
    for (size_t j = 0; j < Q->m; ++j) free(Q->P[i].A[j]);
    free(Q->P[i].A);
    free(Q->P[i].R);
    for (size_t j = 0; j < Q->nr; ++j) free(Q->P[i].NR[j]);
    free(Q->P[i].NR);
    for (size_t j = 0; j < Q->na; ++j) free(Q->P[i].NA[j]);
    free(Q->P[i].NA);
  }
  for (size_t i = 0; i < Q->pr; ++i) array_uint8_t_free_contents(&Q->I_GR[i]);
  if (free_shared) {
    free(Q->I_F); free(Q->I_A); free(Q->I_PR); free(Q->I_GR);
    free(Q->I_NR); free(Q->I_NA);
    free(Q->O_NR); free(Q->O_NA);
  }
}
void free_prob_storage(prob_storage_t *Q) { free_prob_storage_contents(Q, true); free(Q); }

/* Initializes learnable indices U->I_F and U->I_A in storage S according to PF and AD of size n
 * and m respectively. If fail, goto fail. */
bool init_learnable_indices(program_t *P, prob_storage_t *U, count_storage_t *V, prob_storage_t *S,
    count_storage_t *C) {
  size_t n_lpf, n_lad, n_pr;
  uint16_t *I_F, *I_A, *I_PR;
  array_uint8_t *I_GR = NULL;
  if (U) {
    n_lpf = U->n; n_lad = U->m;
    I_F = U->I_F; I_A = U->I_A;
    n_pr = U->pr; I_PR = U->I_PR;
    I_GR = U->I_GR;
  } else {
    n_lpf = V->n, n_lad = V->m;
    I_F = V->I_F, I_A = V->I_A;
    /* Currently not supported. */
    n_pr = 0; I_PR = NULL;
  }

  if (!(n_lpf || n_lad || n_pr)) {
    prob_fact_t *PF = P->PF;
    annot_disj_t *AD = P->AD;
    size_t n = P->PF_n, m = P->AD_n, i;
    I_F = I_A = I_PR = NULL;
    for (i = n_lpf = 0; i < n; ++i) if (PF[i].learnable) ++n_lpf;
    if (n_lpf) {
      I_F = (uint16_t*) malloc(n_lpf*sizeof(uint16_t));
      if (!I_F) goto fail;
      for (size_t j = i = 0; i < n; ++i) if (PF[i].learnable) I_F[j++] = i;
    }
    for (i = n_lad = 0; i < m; ++i) if (AD[i].learnable) ++n_lad;
    if (n_lad) {
      I_A = (uint16_t*) malloc(n_lad*sizeof(uint16_t));
      if (!I_A) goto fail;
      for (size_t j = i = 0; i < m; ++i) if (AD[i].learnable) I_A[j++] = i;
    }
    if (S) {
      for (i = n_pr = 0; i < P->PR_n; ++i) if (P->PR[i].learnable) ++n_pr;
      if (n_pr) {
        I_PR = (uint16_t*) malloc(n_pr*sizeof(uint16_t));
        if (!I_PR) goto fail;
        for (size_t j = i = 0; i < P->PR_n; ++i) if (P->PR[i].learnable) I_PR[j++] = i;
      }
    }
  }
  if (S) {
    if (n_pr && !I_GR) {
      I_GR = (array_uint8_t*) malloc(n_pr*sizeof(array_uint8_t));
      if (!I_GR) goto fail;
      for (size_t i = 0; i < n_pr; ++i)
        if (!array_uint8_t_init(&I_GR[i])) {
          for (size_t j = 0; j < i; ++j) array_uint8_t_free_contents(&I_GR[j]);
          goto fail;
        }
    }
    S->I_F = I_F; S->I_A = I_A;
    S->n = n_lpf; S->m = n_lad;
    S->pr = n_pr; S->I_PR = I_PR; S->I_GR = I_GR;
  } else {
    C->I_F = I_F; C->I_A = I_A;
    C->n = n_lpf; C->m = n_lad;
  }
  return true;
fail:
  PyErr_SetString(PyExc_MemoryError, "could not allocate memory!");
  free(I_F); free(I_A); free(I_GR); free(I_PR);
  return false;
}

bool init_learnable_neural_indices(program_t *P, prob_storage_t *U, prob_storage_t *S) {
  uint16_t *I_NR = U->I_NR, *I_NA = U->I_NA;
  size_t n_lnr = U->nr, n_lna = U->na;
  uint16_t *O_NR = U->O_NR, *O_NA = U->O_NA;
  if (!n_lnr) {
    neural_rule_t *NR = P->NR;
    size_t i, n = P->NR_n;
    for (i = n_lnr = 0; i < n; ++i) if (NR[i].learnable) ++n_lnr;
    if (n_lnr) {
      I_NR = O_NR = NULL;
      I_NR = (uint16_t*) malloc(n_lnr*sizeof(uint16_t));
      if (!I_NR) goto fail;
      O_NR = (uint16_t*) malloc(n_lnr*sizeof(uint16_t));
      if (!O_NR) goto fail;
      size_t s = P->PF_n + P->CF_n;
      for (size_t j = i = 0; i < n; ++i) {
        if (NR[i].learnable) { I_NR[j] = i; O_NR[j++] = s; }
        s += NR[i].n*NR[i].o;
      }
    }
  }
  if (!n_lna) {
    neural_annot_disj_t *NA = P->NA;
    size_t i, m = P->NA_n;
    for (i = n_lna = 0; i < m; ++i) if (NA[i].learnable) ++n_lna;
    if (n_lna) {
      I_NA = O_NA = NULL;
      I_NA = (uint16_t*) malloc(n_lna*sizeof(uint16_t));
      if (!I_NA) goto fail;
      O_NA = (uint16_t*) malloc(n_lna*sizeof(uint16_t));
      if (!O_NA) goto fail;
      size_t s = P->AD_n;
      for (size_t j = i = 0; i < m; ++i) {
        if (NA[i].learnable) { I_NA[j] = i; O_NA[j++] = s; }
        s += NA[i].n*NA[i].o;
      }
    }
  }
  S->I_NR = I_NR; S->I_NA = I_NA; S->O_NR = O_NR; S->O_NA = O_NA;
  S->nr = n_lnr; S->na = n_lna;
  return true;
fail:
  PyErr_SetString(PyExc_MemoryError, "could not allocate memory!");
  free(I_NR); free(I_NA); free(O_NR); free(O_NA);
  return false;
}

/* Initializes learnable statements F and A in storage S according to PF and AD of size n and m
 * respectively. Initialized data is of type type_t. If fail, goto fail. */
bool init_learnable_storage(annot_disj_t *AD, prob_storage_t *U, prob_obs_storage_t *S) {
  uint16_t *I_A = U->I_A;
  size_t n_lpf = U->n, n_lad = U->m;
  double (*F)[2] = NULL;
  double **A = NULL;
  double (*R)[2] = NULL;
  if (n_lpf) {
    F = (double(*)[2]) calloc(n_lpf, sizeof(double[2]));
    if (!F) goto fail;
  }
  if (n_lad) {
    A = (double**) malloc(n_lad*sizeof(double*));
    if (!A) goto fail;
    for (size_t i = 0; i < n_lad; ++i) {
      A[i] = (double*) calloc(AD[I_A[i]].n, sizeof(double));
      if (!A[i]) {
        for (size_t j = 0; j < i; ++j) free(A[j]);
        goto fail;
      }
    }
  }
  if (U->pr) {
    R = (double(*)[2]) calloc(U->pr, sizeof(double[2]));
    if (!F) goto fail;
  }
  S->F = F; S->A = A; S->R = R;
  return true;
fail:
  free(F); free(A); free(R);
  return false;
}

bool init_learnable_neural_storage(neural_rule_t *NR, neural_annot_disj_t *NA, prob_storage_t *U,
    prob_obs_storage_t *S) {
  uint16_t *I_NA = U->I_NA;
  size_t n_lnr = U->nr, n_lna = U->na;
  double **R = NULL;
  double **A = NULL;
  if (n_lnr) {
    R = (double**) malloc(n_lnr*sizeof(double*));
    if (!R) goto fail;
    for (size_t i = 0; i < n_lnr; ++i) {
      /* R[i] = [P(not f(x; 0)), P(f(x; 0)), P(not f(x; 1)), P(f(x; 1)), P(not f(y; 0)), ...] */
      R[i] = (double*) calloc(2*NR[U->I_NR[i]].n*NR[U->I_NR[i]].o, sizeof(double));
      if (!R[i]) {
        for (size_t j = 0; j < i; ++j) free(R[j]);
        goto fail;
      }
    }
  }
  if (n_lna) {
    A = (double**) malloc(n_lna*sizeof(double*));
    if (!A) goto fail;
    for (size_t i = 0; i < n_lna; ++i) {
      /* A[i] = [P(f(x, 0; 0)), P(f(x, 1; 0)), P(f(x, 0; 1)), P(f(x, 1; 1)), P(f(y, 0; 0)), ...] */
      A[i] = (double*) calloc(NA[I_NA[i]].v*NA[I_NA[i]].n*NA[I_NA[i]].o, sizeof(double));
      if (!A[i]) {
        for (size_t j = 0; j < i; ++j) free(A[j]);
        goto fail;
      }
    }
  }
  S->NR = R; S->NA = A;
  return true;
fail:
  free(R); free(A);
  return false;
}

bool init_count_storage(count_storage_t *C, program_t *P, count_storage_t *U) {
  if (!init_learnable_indices(P, NULL, U, NULL, C)) goto cleanup;
  if (C->n) {
    C->F = (uint16_t(*)[2]) calloc(C->n, sizeof(uint16_t[2]));
    if (!C->F) goto cleanup;
  } else C->F = NULL;
  if (C->m) {
    C->A = (uint16_t**) malloc(C->m*sizeof(uint16_t*));
    if (!C->A) goto cleanup;
    for (size_t i = 0; i < C->m; ++i) {
      C->A[i] = (uint16_t*) calloc(P->AD[C->I_A[i]].n, sizeof(uint16_t));
      if (!C->A[i]) {
        for (size_t j = 0; j < i; ++j) free(C->A[j]);
        goto cleanup;
      }
    }
  } else C->A = NULL;
  return true;
cleanup:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
  free(C->F);
  free(C->A);
  return false;
}

bool init_prob_storage(prob_storage_t *Q, program_t *P, prob_storage_t *U, observations_t *O) {
  prob_obs_storage_t *po = NULL;
  po = (prob_obs_storage_t*) malloc(O->n*sizeof(prob_obs_storage_t));
  if (!po) goto cleanup;
  if (!init_learnable_indices(P, U, NULL, Q, NULL)) goto cleanup;
  if (!init_learnable_neural_indices(P, U, Q)) goto cleanup;
  for (size_t i = 0; i < O->n; ++i) {
    prob_obs_storage_t *o = po + i;
    if (!init_learnable_storage(P->AD, U, o)) goto cleanup;
    if (!init_learnable_neural_storage(P->NR, P->NA, U, o)) goto cleanup;
    o->o = 0.0; o->N = 0;
  }
  Q->o = O->n;
  Q->P = po;
  return true;
cleanup:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
  free(po);
  return false;
}


