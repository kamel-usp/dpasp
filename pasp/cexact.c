#include <math.h>

#include "cexact.h"

#include "cdata.h"
#include "carray.h"
#include "coptimize.h"
#include "cutils.h"

bool setup_polynomial(array_bool_t (**Pn)[4], array_double_t (**K)[4], program_t *P) {
  size_t i;

  *Pn = (array_bool_t(*)[4]) malloc(P->Q_n*sizeof(**Pn));
  if (!(*Pn)) return false;
  *K = (array_double_t(*)[4]) malloc(P->Q_n*sizeof(**K));
  if (!(*K)) return false;

  for (i = 0; i < P->Q_n; ++i)
    if (!(array_bool_init(&(*Pn)[i][0]) && array_bool_init(&(*Pn)[i][1]) && array_bool_init(&(*Pn)[i][2])
      && array_bool_init(&(*Pn)[i][3]) && array_double_init(&(*K)[i][0]) && array_double_init(&(*K)[i][1])
      && array_double_init(&(*K)[i][2]) && array_double_init(&(*K)[i][3]))) return false;

  return true;
}

bool setup_credal(double **L_CF, double **U_CF, double **X, program_t *P) {
  size_t i;

  *L_CF = (double*) malloc(P->CF_n*sizeof(double));
  if (!(*L_CF)) return false;
  *U_CF = (double*) malloc(P->CF_n*sizeof(double));
  if (!(*U_CF)) return false;
  for (i = 0; i < P->CF_n; ++i) (*L_CF)[i] = P->CF[i].l, (*U_CF)[i] = P->CF[i].u;

  *X = (double*) malloc(P->CF_n*sizeof(double));
  if (!(*X)) return false;

  return true;
}

bool neg_partial_cmp(bool x, bool _x, char s) {
  /* See page 36 of the lparse manual. This is the negation of the truth value of an atom. */
  if (s == QUERY_TERM_POS)
    return !(x && _x);
  else if (s == QUERY_TERM_UND)
    return x || !_x; /* ≡ !(!x && _x) */
  /* else s == QUERY_TERM_NEG */
  return _x; /* (x && _x) || (!x && _x) ≡ !(x && _x) && !(!x && _x); */
}

/* Adds the rule _a :- a. for every grounded atom a in the Herbrand Base, signalling _a as
 * potentially true. */
bool add_pt_hb(clingo_control_t *C, clingo_backend_t *B) {
  const clingo_symbolic_atoms_t *atoms;
  clingo_symbolic_atom_iterator_t it, end;
  /* Get symbolic atoms. */
  if (!clingo_control_symbolic_atoms(C, &atoms)) goto error;
  /* Get begin iterator. */
  if (!clingo_symbolic_atoms_begin(atoms, NULL, &it)) goto error;
  /* Get end iterator. */
  if (!clingo_symbolic_atoms_end(atoms, &end)) goto error;

  while (true) {
    bool is_end;
    clingo_symbol_t s, _s;
    clingo_atom_t a;
    clingo_literal_t l;
    const clingo_symbol_t *args;
    size_t argc;
    bool pos;
    const char *o_name;
    char name[200];
    /* Check if end of line. */
    if (!clingo_symbolic_atoms_iterator_is_equal_to(atoms, it, end, &is_end)) goto error;
    if (is_end) break;
    /* Get the associated symbol. */
    if (!clingo_symbolic_atoms_symbol(atoms, it, &s)) goto error;
    /* Get s's name. */
    if (!clingo_symbol_name(s, &o_name)) goto error;
    /* Get s's arguments. */
    if (!clingo_symbol_arguments(s, &args, &argc)) goto error;
    /* Get s's sign. */
    if (!clingo_symbol_is_positive(s, &pos)) goto error;
    /* Create _s. */
    strcpy(name+1, o_name);
    name[0] = '_';
    if (!clingo_symbol_create_function(name, args, argc, pos, &_s)) goto error;
    /* Get s's literal. */
    if (!clingo_symbolic_atoms_literal(atoms, it, &l)) goto error;
    /* Add to the backend. */
    if (!clingo_backend_add_atom(B, &_s, &a)) goto error;
    if (!clingo_backend_rule(B, false, &a, 1, &l, 1)) return false;
    /* Next! */
    if (!clingo_symbolic_atoms_next(atoms, it, &it)) goto error;

    wprintf(L"%s :- %s.\n", name, o_name);
  }

  return true;
error:
  return false;
}

#define DEBUG_PRINT(pid, msg) wprintf(L"pid %d: " msg "\n", pid);

#define MODEL_CONTAINS_QUERY true
#define MODEL_CONTAINS_EVI   false

bool model_contains(const clingo_model_t *M, query_t *q, size_t i, bool *c, bool query_or_evi, bool is_partial) {
  clingo_symbol_t x, x_u;
  uint8_t s;
  bool c_x;

  if (query_or_evi) {
    /* Query. */
    x = q->Q[i]; s = q->Q_s[i];
    if (is_partial) x_u = q->Q_u[i];
  } else {
    /* Evidence. */
    x = q->E[i]; s = q->E_s[i];
    if (is_partial) x_u = q->E_u[i];
  }

  if (!clingo_model_contains(M, x, &c_x)) return false;
  if (is_partial) {
    bool c_a;
    if (!clingo_model_contains(M, x_u, &c_a)) return false;
    if (neg_partial_cmp(c_x, c_a, s)) { *c = false; return true; }
  } else {
    if (c_x != s) { *c = false; return true; }
  }
  *c = true;
  return true;
}

void compute_total_choice(void *data) {
  storage_t *st = (storage_t*) data;
  size_t i, m;
  clingo_control_t *C = NULL;
  program_t *P = st->P;
  total_choice_t *theta = &st->theta;
  bool *cond_1 = st->cond_1, *cond_2 = st->cond_2, *cond_3 = st->cond_3, *cond_4 = st->cond_4;
  size_t *count_q_e = st->count_q_e, *count_e = st->count_e, *count_partial_q_e = st->count_partial_q_e;
  double *a = st->a, *b = st->b, *c = st->c, *d = st->d, p;
  array_bool_t (*Pn)[4] = st->Pn;
  array_double_t (*K)[4] = st->K;

  st->fail = true;

  /* Check SAT if partial and lstable_sat. */
  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, theta->theta_ad, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  size_t CF_n = P->CF_n;
  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  bool is_partial = P->sem, has_credal = P->CF_n;

  /* Add credal, probabilistic, and grounded probabilistic facts. */
  if (!prepare_control(&C, P, theta, theta->theta_ad, "0", false, NULL)) goto cleanup;
  /* Zero-initialize counters and flags. */
  memset(cond_1, 0, Q_n); memset(cond_2, 0, Q_n);
  memset(cond_3, 0, Q_n); memset(cond_4, 0, Q_n);
  memset(count_q_e, 0, Q_n_bytes);
  memset(count_e, 0, Q_n_bytes);
  memset(count_partial_q_e, 0, Q_n_bytes);
  /* Start solving. */ {
    bool ok = true;
    clingo_solve_handle_t *handle;
    clingo_solve_result_bitset_t solve_ret;
    const clingo_model_t *M;
    /* Get the solve handle. */
    if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
      goto solve_error;
    /* Iterate over all stable models. */
    for (m = 0; true; ++m) {
      /* m is the number of stable models according to <P,θ>, i.e. m = |Γ(θ)|. */
      if (!clingo_solve_handle_resume(handle)) goto solve_error;
      if (!clingo_solve_handle_model(handle, &M)) goto solve_error;
      if (M) {
        for (i = 0; i < Q_n; ++i) {
          size_t j;
          query_t *q = (P->Q)+i;
          bool all_e = true, all_q = true, c;
          /* all_e? - are all evidence symbols E from query q in M? */
          for (j = 0; j < q->E_n; ++j) {
            if (!model_contains(M, q, j, &c, MODEL_CONTAINS_EVI, is_partial)) goto solve_error;
            if (!c) { all_e = false; break; }
          }
          if (!all_e) continue;
          /* all_q? - are all query symbols Q from query q in M? */
          for (j = 0; j < q->Q_n; ++j) {
            if (!model_contains(M, q, j, &c, MODEL_CONTAINS_QUERY, is_partial)) goto solve_error;
            if (!c) { all_q = false; break; }
          }
          ++count_e[i];
          if (all_q) { cond_2[i] = true; ++count_q_e[i]; }
          else { cond_4[i] = true; ++count_partial_q_e[i]; }
        }
      } else break;
    }
    if (!clingo_solve_handle_get(handle, &solve_ret)) goto solve_error;
    goto solve_cleanup;
solve_error:
    ok = false;
solve_cleanup:
    if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
  }
  /* Probabilities are wrong when a total choice has no model. */
  if (m == 0) st->warn = true;
  /* Compute ℙ(θ). */
  p = prob_total_choice(P, theta);
  for (i = 0; i < Q_n; ++i) {
    /* Evaluate counts to judge whether cond_1 and/or cond_3 are true. */
    if (count_e[i] == m || P->Q[i].E_n == 0) {
      /* All stable models satisfy Q and E completely. */
      if (count_q_e[i] == m) cond_1[i] = true;
      /* All stable models satisfy E, but none satisfies Q completely. */
      if (count_partial_q_e[i] == m) cond_3[i] = true;
    }
    /* Add probability ℙ(θ) according to model satisfiabilities. */
    if (has_credal) {
      size_t j;
      if (cond_1[i] || cond_2[i] || cond_3[i] || cond_4[i]) {
        pthread_mutex_lock(st->mu);
        if (cond_1[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][0], CHOICE_IS_TRUE(theta, j))) goto cleanup;
          if (!array_double_append(&K[i][0], p)) goto cleanup;
        } if (cond_2[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][1], CHOICE_IS_TRUE(theta, j))) goto cleanup;
          if (!array_double_append(&K[i][1], p)) goto cleanup;
        } if (cond_3[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][2], CHOICE_IS_TRUE(theta, j))) goto cleanup;
          if (!array_double_append(&K[i][2], p)) goto cleanup;
        } if (cond_4[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][3], CHOICE_IS_TRUE(theta, j))) goto cleanup;
          if (!array_double_append(&K[i][3], p)) goto cleanup;
        }
        pthread_mutex_unlock(st->mu);
      }
    } else {
      a[i] += cond_1[i]*p;
      b[i] += cond_2[i]*p;
      c[i] += cond_3[i]*p;
      d[i] += cond_4[i]*p;
    }
  }

  st->fail = false;
cleanup:
  clingo_control_free(C);
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

void compute_total_choice_maxent(void *data) {
  storage_t *st = (storage_t*) data;
  size_t i, m;
  clingo_control_t *C = NULL;
  program_t *P = st->P;
  total_choice_t *theta = &st->theta;
  size_t *count_q_e = st->count_q_e, *count_e = st->count_e;
  double *a = st->a, *b = st->b, p;

  st->fail = true;

  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, theta->theta_ad, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  bool is_partial = P->sem;

  if (!prepare_control(&C, P, theta, theta->theta_ad, "0", false, NULL)) goto cleanup;

  memset(count_q_e, 0, Q_n_bytes);
  memset(count_e, 0, Q_n_bytes);
  /* Solving. */ {
    bool ok = true;
    clingo_solve_handle_t *handle;
    clingo_solve_result_bitset_t solve_ret;
    const clingo_model_t *M;

    if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
      goto solve_error;

    for (m = 0; true; ++m) {
      if (!clingo_solve_handle_resume(handle)) goto solve_error;
      if (!clingo_solve_handle_model(handle, &M)) goto solve_error;
      if (M) {
        for (i = 0; i < Q_n; ++i) {
          size_t j;
          query_t *q = (P->Q)+i;
          bool all_e = true, all_q = true, c;
          for (j = 0; j < q->E_n; ++j) {
            if (!model_contains(M, q, j, &c, MODEL_CONTAINS_EVI, is_partial)) goto solve_error;
            if (!c) { all_e = false; break; }
          }
          if (!all_e) continue;
          for (j = 0; j < q->Q_n; ++j) {
            if (!model_contains(M, q, j, &c, MODEL_CONTAINS_QUERY, is_partial)) goto solve_error;
            if (!c) { all_q = false; break; }
          }
          ++count_e[i];
          if (all_q) ++count_q_e[i];
        }
      } else break;
    }
    if (!clingo_solve_handle_get(handle, &solve_ret)) goto solve_error;
    goto solve_cleanup;
solve_error:
    ok = false;
solve_cleanup:
    if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
  }
  /* Probabilities are wrong when a total choice has no model. */
  if (m == 0) st->warn = true;
  p = prob_total_choice(P, theta);
  for (i = 0; i < Q_n; ++i) {
    a[i] += (count_q_e[i]*p)/m;
    b[i] += (count_e[i]*p)/m;
  }

  st->fail = false;
cleanup:
  clingo_control_free(C);
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

bool exact_enum(program_t *P, double **R, bool lstable_sat, psemantics_t psem, bool quiet) {
  bool has_credal = P->CF_n > 0, has_neural = P->NR_n + P->NA_n > 0;
  double *a, *b, *c, *d = c = b = a = NULL;
  size_t Q_n = P->Q_n, i;
  size_t total_choice_n = get_num_facts(P);
  total_choice_t theta;
  array_bool_t (*Pn)[4] = NULL;
  array_double_t (*K)[4] = NULL;
  double *X, *L_CF, *U_CF = L_CF = X = NULL;
  size_t num_procs = estimate_nprocs(total_choice_n + P->AD_n);
  threadpool pool = thpool_init(num_procs);
  bool busy_procs[NUM_PROCS] = {0}, exact_num_ok, warn = false;
  storage_t S[NUM_PROCS] = {{0}};
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wakeup = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t avail = PTHREAD_COND_INITIALIZER;
  void (*compute_func)(void*) = psem ? compute_total_choice_maxent : compute_total_choice;

  if (!init_total_choice(&theta, total_choice_n, P)) goto cleanup;

  if (has_credal) {
    if (!setup_credal(&L_CF, &U_CF, &X, P)) goto cleanup;
    if (!setup_polynomial(&Pn, &K, P)) goto cleanup;
  }

  for (i = 0; i < num_procs; ++i)
    if (!init_storage(&S[i], P, Pn, K, i, busy_procs, &mu, &wakeup, &avail, lstable_sat,
          total_choice_n, P->AD, P->AD_n))
      goto cleanup;

  for (i = 0; i < P->NR_n; ++i)
    if (!update_pr_neural_rule(&P->NR[i])) goto cleanup;
  for (i = 0; i < P->NA_n; ++i)
    if (!update_pr_neural_annot_disj(&P->NA[i])) goto cleanup;

  size_t data_stride = has_neural ? P->m_test : 1;
  /* If credal, then 2: lower and upper; else, then 1: sharp probability. */
  size_t sem_stride = psem == MAXENT_SEMANTICS ? 1 : 2;
  double *R_data = (double*) malloc(Q_n*sem_stride*data_stride*sizeof(double));
  if (!R_data) {
    PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for exact result!");
    goto cleanup;
  }
  *R = R_data;
  double *I = R_data;
  for (size_t ds = 0; ds < data_stride; ++ds) {
    do {
      do {
        if (!dispatch_job(&theta, &wakeup, busy_procs, S, num_procs, pool, &avail, compute_func))
          goto cleanup;
      } while (incr_total_choice_ad(&theta, P));
    } while (incr_total_choice(&theta));
    thpool_wait(pool);

    for (i = 0; i < num_procs; ++i) warn |= S[i].warn;

    if (!has_credal) {
      a = S[0].a; b = S[0].b; c = S[0].c; d = S[0].d;
      for (i = 1; i < num_procs; ++i) {
        size_t j;
        for (j = 0; j < Q_n; ++j) {
          a[j] += S[i].a[j];
          b[j] += S[i].b[j];
          c[j] += S[i].c[j];
          d[j] += S[i].d[j];
        }
      }
    }

    for (i = 0; i < Q_n; ++i) {
      size_t i_l = i*sem_stride;
      size_t i_u = i_l+1;
      if (has_credal) {
        if (P->Q[i].E_n == 0) {
          double _a, _b;
          bf(X, Pn[i][0].d, Pn[i][1].d, K[i][0].d, K[i][1].d, L_CF, U_CF, K[i][0].n, K[i][1].n,
              P->CF_n, &_a, &_b, true);
          I[i_l] = _a, I[i_u] = _b;
        } else {
          size_t _a = K[i][0].n, _b = K[i][1].n, _c = K[i][2].n, _d = K[i][3].n;
          if (_b + _d == 0) {
            fputws(L"Fail: ℙ(E) = 0!\n", stdout);
            I[i_l] = -INFINITY, I[i_u] = INFINITY;
          } else {
            if ((_b + _c == 0) && (_d > 0)) I[i_l] = 0, I[i_u] = 0;
            else if ((_a + _d == 0) && (_b > 0)) I[i_l] = 1, I[i_u] = 1;
            else {
              double min, max;
              bf_minmax(X, Pn[i][0].d, Pn[i][1].d, Pn[i][2].d, Pn[i][3].d, K[i][0].d, K[i][1].d,
                  K[i][2].d, K[i][3].d, L_CF, U_CF, _a, _b, _c, _d, P->CF_n, &min, &max);
              I[i_l] = min, I[i_u] = max;
            }
          }
        }
      } else {
        if (psem == MAXENT_SEMANTICS) I[i_l] = a[i]/b[i];
        else {
          double _a = a[i], _b = b[i], _c = c[i], _d = d[i];
          if (P->Q[i].E_n == 0) I[i_l] = _a, I[i_u] = _b;
          else {
            if (_b + _d == 0) {
              fputws(L"Fail: ℙ(E) = 0!\n", stdout);
              I[i_l] = -INFINITY, I[i_u] = INFINITY;
            } else {
              if ((_b + _c == 0) && (_d > 0)) I[i_l] = 0, I[i_u] = 0;
              else if ((_a + _d == 0) && (_b > 0)) I[i_l] = 1, I[i_u] = 1;
              else I[i_l] = _a/(_a + _d), I[i_u] = _b/(_b + _c);
            }
          }
        }
      }
      if (!quiet) {
        print_query(P->Q+i);
        if (psem == MAXENT_SEMANTICS) wprintf(L" = %f\n", I[i_l]);
        else wprintf(L" = [%f, %f]\n", I[i_l], I[i_u]);
      }
    }
    if (!quiet) fputws(L"---\n", stdout);

    /* Move memory for next batch. */
    I += Q_n*sem_stride;
    for (i = 0; i < P->NR_n; ++i) ++P->NR[i].P;
    for (i = 0; i < P->NA_n; ++i) P->NA[i].P += P->NA[i].v;
    /* Reset memory for next batch. */
    if ((data_stride > 1) && !P->CF_n) {
      size_t s = Q_n*sizeof(double);
      for (i = 0; i < num_procs; ++i) {
        memset(S[i].a, 0, s); memset(S[i].b, 0, s);
        memset(S[i].c, 0, s); memset(S[i].d, 0, s);
      }
    }
  }

  if (warn)
    fputws(L"Warning: found total choice with no model. Probabilities may be incorrect.\n", stdout);

  exact_num_ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  free_total_choice_contents(&theta);
  pthread_mutex_destroy(&mu); pthread_mutex_destroy(&wakeup); pthread_cond_destroy(&avail);
  thpool_destroy(pool);
  for (i = 0; i < num_procs; ++i) free_storage_contents(&S[i]);
  if (has_credal) {
    free(L_CF); free(U_CF); free(X);
    if (Pn) {
      for (i = 0; i < Q_n; ++i) {
        array_bool_free_contents(&Pn[i][0]); array_bool_free_contents(&Pn[i][1]);
        array_bool_free_contents(&Pn[i][2]); array_bool_free_contents(&Pn[i][3]);
      } free(Pn);
    } if (K) {
      for (i = 0; i < Q_n; ++i) {
        array_double_free_contents(&K[i][0]); array_double_free_contents(&K[i][1]);
        array_double_free_contents(&K[i][2]); array_double_free_contents(&K[i][3]);
      } free(K);
    }
  }
  return exact_num_ok;
}

/* Initializes learnable indices I_F and I_A in storage S according to PF and AD of size n and m
 * respectively. If fail, goto fail. */
bool init_learnable_indices(program_t *P, uint16_t *I_F, uint16_t *I_A, size_t n_lpf, size_t n_lad,
    prob_storage_t *S, count_storage_t *C) {
  if (!(n_lpf || n_lad)) {
    prob_fact_t *PF = P->PF;
    annot_disj_t *AD = P->AD;
    size_t n = P->PF_n, m = P->AD_n, i;
    I_F = I_A = NULL;
    n_lpf = n_lad = 0;
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
      for (size_t j = i = 0; i < m; ++i) if (PF[i].learnable) I_A[j++] = i;
    }
  }
  if (S) {
    S->I_F = I_F; S->I_A = I_A;
    S->n = n_lpf; S->m = n_lad;
  } else {
    C->I_F = I_F; C->I_A = I_A;
    C->n = n_lpf; C->m = n_lad;
  }
  return true;
fail:
  free(I_F); free(I_A);
  return false;
}

bool init_learnable_neural_indices(program_t *P, uint16_t *I_NR, uint16_t *I_NA, size_t n_lnr,
    size_t n_lna, uint16_t *O_NR, uint16_t *O_NA, prob_storage_t *S) {
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
      size_t s = P->PF_n;
      for (size_t j = i = 0; i < n; ++i) {
        if (NR[i].learnable) { I_NR[j] = i; O_NR[j++] = s; }
        s += NR[i].n;
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
        s += NA[i].n;
      }
    }
  }
  S->I_NR = I_NR; S->I_NA = I_NA; S->O_NR = O_NR; S->O_NA = O_NA;
  S->nr = n_lnr; S->na = n_lna;
  return true;
fail:
  free(I_NR); free(I_NA); free(O_NR); free(O_NA);
  return false;
}

/* Initializes learnable statements F and A in storage S according to PF and AD of size n and m
 * respectively. Initialized data is of type type_t. If fail, goto fail. */
bool init_learnable_storage(annot_disj_t *AD, uint16_t *I_A, size_t n_lpf, size_t n_lad,
    prob_obs_storage_t *S) {
  double (*F)[2] = NULL;
  double **A = NULL;
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
  S->F = F; S->A = A;
  return true;
fail:
  free(F); free(A);
  return false;
}

bool init_learnable_neural_storage(neural_annot_disj_t *NA, uint16_t *I_NA, size_t n_lnr,
    size_t n_lna, prob_obs_storage_t *S) {
  double (*R)[2] = NULL;
  double **A = NULL;
  if (n_lnr) {
    R = (double(*)[2]) calloc(n_lnr, sizeof(double[2]));
    if (!R) goto fail;
  }
  if (n_lna) {
    A = (double**) malloc(n_lna*sizeof(double*));
    if (!A) goto fail;
    for (size_t i = 0; i < n_lna; ++i) {
      A[i] = (double*) calloc(NA[I_NA[i]].v, sizeof(double));
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

bool init_count_storage(count_storage_t *C, program_t *P, uint16_t *I_F, size_t n_lpf,
    uint16_t *I_A, size_t n_lad) {
  if (!init_learnable_indices(P, I_F, I_A, n_lpf, n_lad, NULL, C)) goto cleanup;
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

void free_count_storage_contents(count_storage_t *C, bool free_shared) {
  free(C->F);
  free(C->A);
  if (free_shared) {
    free(C->I_F);
    free(C->I_A);
  }
}
void free_count_storage(count_storage_t *C) { free_count_storage_contents(C, true); free(C); }

void compute_model_count(void *args) {
  struct { count_storage_t *C; storage_t *S; } *pair = args;
  count_storage_t *cnt = pair->C;
  storage_t *st = pair->S;
  total_choice_t *theta = &st->theta;
  program_t *P = st->P;
  size_t i, m;
  clingo_control_t *C = NULL;

  st->fail = true;

  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, theta->theta_ad, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  if (!prepare_control(&C, P, theta, theta->theta_ad, "0", false, NULL)) goto cleanup;

  {
    bool ok = false;
    clingo_solve_handle_t *handle;
    clingo_solve_result_bitset_t solve_ret;

    if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
      goto solve_cleanup;

    for (m = 0; true; ++m) {
      if (!clingo_solve_handle_resume(handle)) goto solve_cleanup;
      if (!clingo_solve_handle_get(handle, &solve_ret)) goto solve_cleanup;
      if (solve_ret & clingo_solve_result_exhausted) break;
    }

    ok = true;
solve_cleanup:
    if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
  }

  /* Add counts to probabilistic facts that agree with total choice theta. */
  for (i = 0; i < cnt->n; ++i) cnt->F[i][bitvec_GET(&theta->pf, i)] += m;
  /* Add counts to annotated disjunctions that agree with total choice theta. */
  for (i = 0; i < cnt->m; ++i) cnt->A[i][theta->theta_ad[cnt->I_A[i]]] += m;

  st->fail = false;
cleanup:
  clingo_control_free(C);
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

bool count_models(program_t *P, bool lstable_sat, count_storage_t *ret) {
  total_choice_t theta;
  size_t total_choice_n = get_num_facts(P);
  size_t num_procs = estimate_nprocs(total_choice_n + P->AD_n);
  bool busy_procs[NUM_PROCS] = {0};
  count_storage_t C[NUM_PROCS] = {{0}};
  storage_t S[NUM_PROCS] = {{0}};
  size_t i;
  bool ok = false;
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wakeup = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t avail = PTHREAD_COND_INITIALIZER;
  threadpool pool = thpool_init(num_procs);
  struct { count_storage_t *C; storage_t *S; } pairs[NUM_PROCS] = {{0}};

  if (!ret) {
    PyErr_SetString(PyExc_ValueError, "received NULL count_storage_t as argument!");
    goto cleanup;
  }
  if (!init_total_choice(&theta, total_choice_n, P)) goto cleanup;

  for (i = 0; i < num_procs; ++i) {
    /* These are zero-initialized when i = 0. */
    uint16_t *I_F = C[0].I_F, *I_A = C[0].I_A;
    size_t n_lpf = C[0].n, n_lad = C[0].m;
    if (!init_count_storage(&C[i], P, I_F, n_lpf, I_A, n_lad))
      goto cleanup;
    if (!(C[0].n || C[0].m)) goto cleanup;
    S[i].pid = i; S[i].mu = &mu; S[i].wakeup = &wakeup; S[i].avail = &avail;
    S[i].busy_procs = busy_procs; S[i].lstable_sat = lstable_sat;
    S[i].P = P;
    if (!init_total_choice(&S[i].theta, total_choice_n, P)) goto cleanup;
    pairs[i].C = &C[i];
    pairs[i].S = &S[i];
  }

  do {
    do {
      int id = retr_free_proc(busy_procs, num_procs, &wakeup, &avail);
      if (!dispatch_job_with_payload(&theta, &wakeup, busy_procs, S, num_procs, pool, &avail, id,
            compute_model_count, &pairs[id])) goto cleanup;
    } while (incr_total_choice_ad(&theta, P));
  } while (incr_total_choice(&theta));
  thpool_wait(pool);

  for (i = 1; i < num_procs; ++i) {
    for (size_t j = 0; j < C[0].n; ++j) {
      C[0].F[j][0] += C[i].F[j][0];
      C[0].F[j][1] += C[i].F[j][1];
    }
    for (size_t j = 0; j < C[0].m; ++j)
      for (size_t c = 0; c < P->AD[C[0].I_A[j]].n; ++c)
        C[0].A[j][c] += C[i].A[j][c];
  }

  ret->n = C[0].n; ret->m = C[0].m;
  ret->F = C[0].F; ret->A = C[0].A;
  ret->I_F = C[0].I_F; ret->I_A = C[0].I_A;

  ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  free_total_choice_contents(&theta);
  pthread_mutex_destroy(&mu);
  pthread_mutex_destroy(&wakeup);
  pthread_cond_destroy(&avail);
  thpool_destroy(pool);
  /* First count_storage_t has returned values. */
  for (i = 1; i < num_procs; ++i) free_count_storage_contents(&C[i], false);
  if (!ok) free_count_storage_contents(&C[0], true);
  return ok;
}

bool init_prob_storage(prob_storage_t *Q, program_t *P, uint16_t *I_F, size_t n_lpf, uint16_t *I_A,
    size_t n_lad, uint16_t *I_NR, size_t n_lnr, uint16_t *I_NA, size_t n_lna, uint16_t *O_NR,
    uint16_t *O_NA, observations_t *O) {
  prob_obs_storage_t *po = NULL;
  po = (prob_obs_storage_t*) malloc(O->n*sizeof(prob_obs_storage_t));
  if (!po) goto cleanup;
  if (!init_learnable_indices(P, I_F, I_A, n_lpf, n_lad, Q, NULL)) goto cleanup;
  if (!init_learnable_neural_indices(P, I_NR, I_NA, n_lnr, n_lna, O_NR, O_NA, Q)) goto cleanup;
  for (size_t i = 0; i < O->n; ++i) {
    prob_obs_storage_t *o = po + i;
    if (!init_learnable_storage(P->AD, I_A, Q->n, Q->m, o)) goto cleanup;
    if (!init_learnable_neural_storage(P->NA, I_NA, Q->nr, Q->na, o)) goto cleanup;
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

size_t init_prob_storage_seq(prob_storage_t Q[NUM_PROCS], program_t *P, observations_t *O) {
  size_t total_choice_n = get_num_facts(P);
  size_t num_procs = estimate_nprocs(total_choice_n + P->AD_n);
  size_t i = 0;

  for (i = 0; i < num_procs; ++i) {
    uint16_t *I_F = Q[0].I_F, *I_A = Q[0].I_A, *I_NR = Q[0].I_NR, *I_NA = Q[0].I_NA;
    uint16_t *O_NR = Q[0].O_NR, *O_NA = Q[0].O_NA;
    size_t n_lpf = Q[0].n, n_lad = Q[0].m, n_lnr = Q[0].nr, n_lna = Q[0].na;
    if (!init_prob_storage(&Q[i], P, I_F, n_lpf, I_A, n_lad, I_NR, n_lnr, I_NA, n_lna, O_NR, O_NA, O))
      goto cleanup;
    if (!(Q[0].n || Q[0].m || Q[0].nr || Q[0].na)) goto cleanup;
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
    free(Q->P[i].NR);
    for (size_t j = 0; j < Q->na; ++j) free(Q->P[i].NA[j]);
    free(Q->P[i].NA);
  }
  if (free_shared) {
    free(Q->I_F); free(Q->I_A);
    free(Q->I_NR); free(Q->I_NA);
    free(Q->O_NR); free(Q->O_NA);
  }
}
void free_prob_storage(prob_storage_t *Q) { free_prob_storage_contents(Q, true); free(Q); }

void compute_prob_obs(void *args) {
  struct { prob_storage_t *C; storage_t *S; observations_t *O; bool derive; } *tuple = args;
  prob_storage_t *prob = tuple->C;
  storage_t *st = tuple->S;
  observations_t *obs = tuple->O;
  total_choice_t *theta = &st->theta;
  program_t *P = st->P;
  size_t i, N;
  clingo_control_t *C = NULL;

  st->fail = true;

  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, theta->theta_ad, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  if (!prepare_control(&C, P, theta, theta->theta_ad, "0", false, NULL)) goto cleanup;

  /* Reset observation counting. */
  for (size_t i = 0; i < obs->n; ++i) prob->P[i].N = 0;

  {
    bool ok = true;
    clingo_solve_handle_t *handle;
    const clingo_model_t *M;

    if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
      goto solve_error;

    for (N = 0; true; ++N) {
      if (!clingo_solve_handle_resume(handle)) goto solve_error;
      if (!clingo_solve_handle_model(handle, &M)) goto solve_error;
      if (M) {
        for (i = 0; i < obs->n; ++i) {
          for (size_t j = 0; j < obs->m; ++j) {
            bool contains_atom;
            if (obs->dense) {
              if (!obs->V[i][j]) break;
              if (!clingo_model_contains(M, obs->V[i][j], &contains_atom)) goto solve_error;
            } else {
              if (obs->S[i][j] == OBSERVATION_MIS) continue;
              if (!clingo_model_contains(M, obs->A[j], &contains_atom)) goto solve_error;
            }
            if (contains_atom != obs->S[i][j]) goto next_obs;
          }
          /* Count models that are consistent with the observation. */
          ++prob->P[i].N;
next_obs: ;
        }
      } else break;
    }
    goto solve_cleanup;
solve_error:
    ok = false;
solve_cleanup:
    if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
  }

  /* Only multiply after model counting to avoid numeric errors. */
  double p = prob_total_choice_prob(P, theta)/N;
  for (i = 0; i < obs->n; ++i) {
    prob_obs_storage_t *pr = &prob->P[i];
    if (!pr->N) continue;
    double p_o = pr->N * p * prob_total_choice_neural(P, theta, i, true);
    pr->o += p_o;
    if (tuple->derive) {
      for (size_t j = 0; j < prob->n; ++j) {
        bool u = bitvec_GET(&theta->pf, prob->I_F[j]);
        double q = P->PF[prob->I_F[j]].p;
        pr->F[j][u] += p_o/(u*q + (!u)*(1-q));
      }
      for (size_t j = 0; j < prob->m; ++j) {
        uint8_t u = theta->theta_ad[prob->I_A[j]];
        pr->A[j][u] += p_o/(P->AD[prob->I_A[j]].P[u]);
      }
      for (size_t j = 0; j < prob->nr; ++j) {
        neural_rule_t *R = &P->NR[prob->I_NR[j]];
        float *q = R->P + i*R->n;
        for (size_t g = 0; g < R->n; ++g) {
          bool u = bitvec_GET(&theta->pf, prob->O_NR[g]);
          pr->NR[j][u] += p_o/(u*q[g] + (!u)*(1-q[g]));
        }
      }
      for (size_t j = 0; j < prob->na; ++j) {
        neural_annot_disj_t *A = &P->NA[prob->I_NA[j]];
        float *q = A->P + i*A->n*A->v;
        for (size_t g = 0; g < A->n; ++g) {
          uint8_t u = theta->theta_ad[prob->O_NA[j]];
          pr->NA[j][u] += p_o/q[g*A->v+u];
        }
      }
    } else {
      for (size_t j = 0; j < prob->n; ++j)
        pr->F[j][bitvec_GET(&theta->pf, j)] += p_o;
      for (size_t j = 0; j < prob->m; ++j)
        pr->A[j][theta->theta_ad[prob->I_A[j]]] += p_o;
    }
  }

  st->fail = false;
cleanup:
  clingo_control_free(C);
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

bool prob_obs(program_t *P, observations_t *obs, bool lstables_sat, prob_storage_t *ret, bool derive) {
  prob_storage_t Q[NUM_PROCS] = {0};
  size_t num_procs = init_prob_storage_seq(Q, P, obs);

  if (!num_procs) goto cleanup;
  if (!ret) {
    PyErr_SetString(PyExc_ValueError, "received NULL prob_storage_t as argument!");
    goto cleanup;
  }
  if (!prob_obs_reuse(P, obs, lstables_sat, ret, Q, derive)) goto cleanup;

  return true;
cleanup:
  /* First prob_storage_t has returned values. */
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  /* If num_procs = 0, then Q has already been cleaned up by init_prob_storage_seq. Otherwise, the
   * error comes from elsewhere (e.g. prob_obs_reuse), and so the freeing must be done. */
  if (num_procs) free_prob_storage_contents(&Q[0], true);
  return false;
}

bool prob_obs_reuse(program_t *P, observations_t *obs, bool lstable_sat, prob_storage_t *ret,
    prob_storage_t Q[NUM_PROCS], bool derive) {
  total_choice_t theta;
  size_t total_choice_n = get_num_facts(P);
  size_t num_procs = estimate_nprocs(total_choice_n + P->AD_n);
  bool busy_procs[NUM_PROCS] = {0};
  storage_t S[NUM_PROCS] = {{0}};
  size_t i;
  bool ok = false;
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wakeup = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t avail = PTHREAD_COND_INITIALIZER;
  threadpool pool = thpool_init(num_procs);
  struct { prob_storage_t *Q; storage_t *S; observations_t *O; bool derive; } tuple[NUM_PROCS] = {{0}};

  if (!init_total_choice(&theta, total_choice_n, P)) goto cleanup;

  for (i = 0; i < num_procs; ++i) {
    S[i].pid = i; S[i].mu = &mu; S[i].wakeup = &wakeup; S[i].avail = &avail;
    S[i].busy_procs = busy_procs; S[i].lstable_sat = lstable_sat;
    S[i].P = P;
    if (!init_total_choice(&S[i].theta, total_choice_n, P)) goto cleanup;
    tuple[i].Q = &Q[i]; tuple[i].S = &S[i];
    tuple[i].O = obs; tuple[i].derive = derive;
    /* Fill probs with zero. */
    for (size_t j = 0; j < obs->n; ++j) {
      prob_obs_storage_t *pr = &Q[i].P[j];
      memset(pr->F, 0, Q[0].n*sizeof(double[2]));
      for (size_t j = 0; j < Q[0].m; ++j) memset(pr->A[j], 0, STORAGE_AD_DIM(P, &Q[0], j)*sizeof(double));
      memset(pr->NR, 0, Q[0].nr*sizeof(double[2]));
      for (size_t j = 0; j < Q[0].na; ++j) memset(pr->NA[j], 0, P->NA[Q[0].I_NA[j]].v*sizeof(double));
      pr->o = 0.0;
    }
  }

  do {
    do {
      int id = retr_free_proc(busy_procs, num_procs, &wakeup, &avail);
      if (!dispatch_job_with_payload(&theta, &wakeup, busy_procs, S, num_procs, pool, &avail, id,
            compute_prob_obs, &tuple[id])) goto cleanup;
    } while (incr_total_choice_ad(&theta, P));
  } while (incr_total_choice(&theta));
  thpool_wait(pool);

  for (i = 1; i < num_procs; ++i) {
    for (size_t o = 0; o < obs->n; ++o) {
      prob_obs_storage_t *qr = &Q[0].P[o];
      prob_obs_storage_t *pr = &Q[i].P[o];
      for (size_t j = 0; j < Q[0].n; ++j) {
        qr->F[j][0] += pr->F[j][0];
        qr->F[j][1] += pr->F[j][1];
      }
      for (size_t j = 0; j < Q[0].m; ++j) {
        size_t k = P->AD[Q[0].I_A[j]].n;
        for (size_t c = 0; c < k; ++c)
          qr->A[j][c] += pr->A[j][c];
      }
      for (size_t j = 0; j < Q[0].nr; ++j) {
        qr->NR[j][0] += pr->NR[j][0];
        qr->NR[j][1] += pr->NR[j][1];
      }
      for (size_t j = 0; j < Q[0].na; ++j) {
        size_t k = P->NA[Q[0].I_NA[j]].v;
        for (size_t c = 0; c < k; ++c)
          qr->NA[j][c] += pr->NA[j][c];
      }
      qr->o += pr->o;
    }
  }

  if (ret) {
    ret->n = Q[0].n; ret->m = Q[0].m; ret->o = Q[0].o;
    ret->nr = Q[0].nr; ret->na = Q[0].na;
    ret->I_F = Q[0].I_F; ret->I_A = Q[0].I_A;
    ret->I_NR = Q[0].I_NR; ret->I_NA = Q[0].I_NA;
    for (size_t o = 0; o < obs->n; ++o) {
      prob_obs_storage_t *qr = &ret->P[o];
      prob_obs_storage_t *pr = &Q[0].P[o];
      qr->F = pr->F; qr->A = pr->A;
      qr->NR = pr->NR; qr->NA = pr->NA;
      qr->o = pr->o;
    }
  }

  ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  free_total_choice_contents(&theta);
  for (size_t i = 0; i < num_procs; ++i) free_storage_contents(&S[i]);
  pthread_mutex_destroy(&mu);
  pthread_mutex_destroy(&wakeup);
  pthread_cond_destroy(&avail);
  thpool_destroy(pool);
  return ok;
}

