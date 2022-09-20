#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <time.h>
#include <pthread.h>

#include "cprogram.h"
#include "carray.h"
#include "coptimize.h"
#include "cutils.h"
#include "cground.h"
#include "cinf.h"

#include "../thpool/thpool.h"

const clingo_part_t EXACT_DEFAULT_PARTS[] = {{"base", NULL, 0}};
const struct timespec TIME_HALF_SEC = { .tv_sec = 0, .tv_nsec = 500000000L };
const struct timespec TIME_QUARTER_SEC = { .tv_sec = 0, .tv_nsec = 250000000L };

#ifndef NUM_PROCS
#define NUM_PROCS 1
#endif

#define IS_TRUE(t, i) (((t) >> (i)) % 2)
#define PROCS_STR(x) #x ",compete"
#define PROCS_XSTR(x) PROCS_STR(x)
#define NUM_PROCS_CONFIG_STR PROCS_XSTR(NUM_PROCS)

static inline bool setup_config(clingo_control_t *C, const char *nmodels, bool parallelize_clingo) {
  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;

  /* Get the control's configuration. */
  if (!clingo_control_configuration(C, &cfg)) return false;
  /* Set to enumerate all stable models. */
  if (!clingo_configuration_root(cfg, &cfg_root)) return false;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) return false;
  if (!clingo_configuration_value_set(cfg, cfg_sub, nmodels)) return false;
  if (parallelize_clingo) {
    /* Set parallel_mode to "NUM_PROCS,compete", where NUM_PROCS is the #procs in this machine. */
    if (!clingo_configuration_map_at(cfg, cfg_root, "solve.parallel_mode", &cfg_sub)) return false;
    if (!clingo_configuration_value_set(cfg, cfg_sub, NUM_PROCS_CONFIG_STR)) return false;
  }

  return true;
}

static inline bool setup_polynomial(array_bool_t (**Pn)[4], array_double_t (**K)[4], program_t *P) {
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

static inline bool setup_credal(double **L_CF, double **U_CF, double **X, program_t *P) {
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

static inline bool neg_partial_cmp(bool x, bool _x, char s) {
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
static inline bool add_pt_hb(clingo_control_t *C, clingo_backend_t *B) {
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

static inline bool prepare_control(clingo_control_t **C, program_t *P, unsigned long long int theta,
    const char *nmodels, bool parallelize_clingo) {
  size_t i, gr_n = P->gr_pr.n;
  clingo_backend_t *back = NULL;
  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, C)) return false;
  /* Config to enumerate all models. */
  if (!setup_config(*C, nmodels, false)) return false;
  /* Add the purely logical part. */
  if (!clingo_control_add(*C, "base", NULL, 0, P->P)) return false;
  /* Add grounded probabilistic rules. */
  if (P->gr_P.d) if (!clingo_control_add(*C, "base", NULL, 0, P->gr_P.d)) return false;
  /* Get the control's backend. */
  if (!clingo_control_backend(*C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) return false;
  /* Add the credal facts according to the total rule. */
  for (i = 0; i < P->CF_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Add the probabilistic facts according to the total rule. */
  for (i = 0; i < P->PF_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i + P->CF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Add the grounded probabilistic rules according to the total rule. */
  for (i = 0; i < gr_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i + P->CF_n + P->PF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->gr_PF.d[i], &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Cleanup backend. */
  if (!clingo_backend_end(back)) return false;
  /* Ground atoms. */
  if(!clingo_control_ground(*C, EXACT_DEFAULT_PARTS, 1, NULL, NULL)) return false;
  return true;
}

static inline bool has_total_model(program_t *P, unsigned long long int theta, bool *has) {
  clingo_control_t *C = NULL;
  clingo_solve_handle_t *handle;
  clingo_solve_result_bitset_t res;
  /* Prepare control according to the stable semantics. */
  if (!prepare_control(&C, P->stable, theta, "1", false)) goto cleanup;
  /* Solve and determine if there exists a (total) model. */
  if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle)) goto cleanup;
  if (!clingo_solve_handle_get(handle, &res)) goto cleanup;
  *has = (res & clingo_solve_result_satisfiable);
  /* Cleanup. */
  clingo_control_free(C);
  return true;
cleanup:
  clingo_control_free(C);
  return false;
}

#define DEBUG_PRINT(pid, msg) wprintf(L"pid %d: " msg "\n", pid);

#define MODEL_CONTAINS_QUERY true
#define MODEL_CONTAINS_EVI   false

inline static bool model_contains(const clingo_model_t *M, query_t *q, size_t i, bool *c, bool query_or_evi, bool is_partial) {
  clingo_symbol_t x, x_u;
  char s;
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

static void compute_total_choice(void *data) {
  storage_t *st = (storage_t*) data;
  size_t i, m;
  clingo_control_t *C = NULL;
  program_t *P = st->P;
  unsigned long long int theta = st->theta;
  bool *cond_1 = st->cond_1, *cond_2 = st->cond_2, *cond_3 = st->cond_3, *cond_4 = st->cond_4;
  size_t *count_q_e = st->count_q_e, *count_e = st->count_e, *count_partial_q_e = st->count_partial_q_e;
  double *a = st->a, *b = st->b, *c = st->c, *d = st->d, p;
  array_bool_t (*Pn)[4] = st->Pn;
  array_double_t (*K)[4] = st->K;

  /* Check SAT if partial and lstable_sat. */
  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  size_t CF_n = P->CF_n, PF_n = P->PF_n;
  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  unsigned long long int theta_CF = theta & ((1 << CF_n)-1);
  bool is_partial = P->sem, has_credal = P->CF_n;

  /* Add credal, probabilistic, and grounded probabilistic facts. */
  if (!prepare_control(&C, P, theta, "0", false)) goto cleanup;
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
  /* Compute ℙ(θ). */
  p = prob_total_choice(P->PF, PF_n, &P->gr_pr, theta >> CF_n);
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
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][0], (theta_CF >> j) % 2)) goto cleanup;
          if (!array_double_append(&K[i][0], p)) goto cleanup;
        } if (cond_2[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][1], (theta_CF >> j) % 2)) goto cleanup;
          if (!array_double_append(&K[i][1], p)) goto cleanup;
        } if (cond_3[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][2], (theta_CF >> j) % 2)) goto cleanup;
          if (!array_double_append(&K[i][2], p)) goto cleanup;
        } if (cond_4[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][3], (theta_CF >> j) % 2)) goto cleanup;
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
  clingo_control_free(C);
  st->fail = false;
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
  return;
cleanup:
  clingo_control_free(C);
  st->fail = true;
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

static void compute_total_choice_plog(void *data) {
  storage_t *st = (storage_t*) data;
  size_t i, m;
  clingo_control_t *C = NULL;
  program_t *P = st->P;
  unsigned long long int theta = st->theta;
  size_t *count_q_e = st->count_q_e, *count_e = st->count_e;
  double *a = st->a, *b = st->b, p;

  if (P->sem == LSTABLE_SEMANTICS && st->lstable_sat) {
    bool has;
    if (!has_total_model(P, theta, &has)) goto cleanup;
    if (has) P = P->stable;
  }

  size_t PF_n = P->PF_n;
  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  bool is_partial = P->sem;

  if (!prepare_control(&C, P, theta, "0", false)) goto cleanup;

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

  p = prob_total_choice(P->PF, PF_n, &P->gr_pr, theta);
  for (i = 0; i < Q_n; ++i) {
    a[i] += (count_q_e[i]*p)/m;
    b[i] += (count_e[i]*p)/m;
  }

  clingo_control_free(C);
  st->fail = false;
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
  return;
cleanup:
  clingo_control_free(C);
  st->fail = true;
  pthread_mutex_lock(st->wakeup);
  st->busy_procs[st->pid] = false;
  pthread_cond_signal(st->avail);
  pthread_mutex_unlock(st->wakeup);
}

#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif

static bool exact_enum(program_t *P, double (*R)[2], bool lstable_sat, psemantics_t psem) {
  bool has_credal = P->CF_n > 0;
  double *a, *b, *c, *d = c = b = a = NULL;
  size_t Q_n = P->Q_n, gr_n = P->gr_pr.n, i;
  size_t total_choice_n = has_credal ? P->PF_n+P->CF_n+gr_n : P->PF_n+gr_n;
  unsigned long long int theta, theta_max;
  array_bool_t (*Pn)[4] = NULL;
  array_double_t (*K)[4] = NULL;
  double *X, *L_CF, *U_CF = L_CF = X = NULL;
  size_t num_procs = min(NUM_PROCS, 1 << total_choice_n);
  threadpool pool = thpool_init(num_procs);
  bool busy_procs[NUM_PROCS] = {0}, exact_num_ok;
  storage_t S[NUM_PROCS];
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wakeup = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t avail = PTHREAD_COND_INITIALIZER;
  void (*compute_func)(void*) = psem ? compute_total_choice_plog : compute_total_choice;

  if (total_choice_n > 62) {
    fputws(L"exact inference only supports up to 62 probabilistic objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  if (has_credal) {
    if (!setup_credal(&L_CF, &U_CF, &X, P)) goto cleanup;
    if (!setup_polynomial(&Pn, &K, P)) goto cleanup;
  }

  for (i = 0; i < num_procs; ++i)
    if (!init_storage(&S[i], P, Pn, K, i, busy_procs, &mu, &wakeup, &avail, lstable_sat))
      goto cleanup;

  for (theta = 0; theta < theta_max; ++theta) {
    size_t id = (size_t) -1;
    /* The line below does not produce a problematic race condition since it will, at worst, skip
     * the i-th busy_procs and have to iterate NUM_PROCS all over again. */
    pthread_mutex_lock(&wakeup);
    while (true) {
      for (i = 0, id = (size_t) -1; i < num_procs; ++i) {
        if (!busy_procs[i]) { id = i; break; }
      }
      if (id != (size_t) -1) break;
      pthread_cond_wait(&avail, &wakeup);
    }
    busy_procs[id] = true;
    pthread_mutex_unlock(&wakeup);
    S[id].theta = theta;
    if (S[id].fail || thpool_add_work(pool, compute_func, (void*) &S[id])) {
      goto cleanup;
    }
  }
  thpool_wait(pool);

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
    if (has_credal) {
      if (P->Q[i].E_n == 0) {
        double _a, _b;
        bf(X, Pn[i][0].d, Pn[i][1].d, K[i][0].d, K[i][1].d, L_CF, U_CF, K[i][0].n, K[i][1].n,
            P->CF_n, &_a, &_b, true);
        R[i][0] = _a, R[i][1] = _b;
      } else {
        size_t _a = K[i][0].n, _b = K[i][1].n, _c = K[i][2].n, _d = K[i][3].n;
        if (_b + _d == 0) {
          fputws(L"Fail: ℙ(E) = 0!\n", stdout);
          R[i][0] = -INFINITY, R[i][1] = INFINITY;
        } else {
          if ((_b + _c == 0) && (_d > 0)) R[i][0] = 0, R[i][1] = 0;
          else if ((_a + _d == 0) && (_b > 0)) R[i][0] = 1, R[i][1] = 1;
          else {
            double min, max;
            bf_minmax(X, Pn[i][0].d, Pn[i][1].d, Pn[i][2].d, Pn[i][3].d, K[i][0].d, K[i][1].d,
                K[i][2].d, K[i][3].d, L_CF, U_CF, _a, _b, _c, _d, P->CF_n, &min, &max);
            /*bf(X, Pn[i][0].d, Pn[i][3].d, K[i][0].d, K[i][3].d, L_CF, U_CF, _a, _d, CF_n, &min, &max, false);*/
            R[i][0] = min, R[i][1] = max;
          }
        }
      }
    } else {
      if (psem == PLOG_SEMANTICS) {
        double _a = a[i], _b = b[i];
        R[i][0] = R[i][1] = _a/_b;
      } else {
        double _a = a[i], _b = b[i], _c = c[i], _d = d[i];
        if (P->Q[i].E_n == 0) R[i][0] = _a, R[i][1] = _b;
        else {
          if (_b + _d == 0) {
            fputws(L"Fail: ℙ(E) = 0!\n", stdout);
            R[i][0] = -INFINITY, R[i][1] = INFINITY;
          } else {
            if ((_b + _c == 0) && (_d > 0)) R[i][0] = 0, R[i][1] = 0;
            else if ((_a + _d == 0) && (_b > 0)) R[i][0] = 1, R[i][1] = 1;
            else R[i][0] = _a/(_a + _d), R[i][1] = _b/(_b + _c);
          }
        }
      }
    }
    print_query(P->Q+i); wprintf(L" = [%f, %f]\n", R[i][0], R[i][1]);
  }

  exact_num_ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
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

static bool exact_enum_seq(program_t *P, double (*R)[2]) {
  bool *cond_1, *cond_2, *cond_3, *cond_4 = cond_3 = cond_2 = cond_1 = NULL, exact_num_ok = false;
  size_t *count_q_e, *count_e, *count_partial_q_e = count_e = count_q_e = NULL;
  size_t CF_n = P->CF_n, i, PF_n = P->PF_n, gr_n = P->gr_pr.n;
  bool has_credal = CF_n > 0, is_partial = P->sem;
  double *a, *b, *c, *d = c = b = a = NULL;
  size_t err_code = 0, total_choice_n = has_credal ? PF_n+CF_n+gr_n : P->PF_n+gr_n;
  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  unsigned long long int theta, theta_max;
  double p, *X, *L_CF, *U_CF = L_CF = X = NULL;
  array_bool_t (*Pn)[4] = NULL;
  array_double_t (*K)[4] = NULL;
  bool parallelize_clingo = total_choice_n < 8;

  if (total_choice_n > 62) {
    fputws(L"exact inference only supports up to 62 probabilistic objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  err_code = 1;

  if (!setup_conds(&cond_1, &cond_2, &cond_3, &cond_4, Q_n*sizeof(bool))) goto cleanup;
  if (!setup_counts(&count_q_e, &count_e, &count_partial_q_e, Q_n_bytes)) goto cleanup;
  if (!has_credal) { if (!setup_abcd(&a, &b, &c, &d, Q_n, sizeof(double))) goto cleanup; }
  else {
    if (!setup_credal(&L_CF, &U_CF, &X, P)) goto cleanup;
    if (!setup_polynomial(&Pn, &K, P)) goto cleanup;
  }

  /* TODO: ground probabilistic rules with free variables. */

  for (theta = 0; theta < theta_max; ++theta) {
    size_t m;
    clingo_control_t *C = NULL;
    clingo_backend_t *back = NULL;
    unsigned long long int theta_CF = theta & ((1 << CF_n)-1);

    err_code = 2;
    /* Create new clingo controller. */
    if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto cleanup;
    if (!setup_config(C, "0", parallelize_clingo)) goto cleanup;
    /* Add the purely logical part. */
    if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto theta_cleanup;
    /* Add grounded probabilistic rules. */
    if (P->gr_P.d) if (!clingo_control_add(C, "base", NULL, 0, P->gr_P.d)) goto theta_cleanup;
    /* Get the control's backend. */
    if (!clingo_control_backend(C, &back)) goto theta_cleanup;
    /* Startup the backend. */
    err_code = 3;
    if (!clingo_backend_begin(back)) goto theta_cleanup;
    /* Add the credal facts according to the total rule. */
    for (i = 0; i < CF_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i)) continue;
      if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) goto theta_cleanup;
      if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto theta_cleanup;
    }
    /* Add the probabilistic facts according to the total rule. */
    for (i = 0; i < PF_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i + CF_n)) continue;
      if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto theta_cleanup;
      if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto theta_cleanup;
    }
    /* Add the grounded probabilistic rules according to the total rule. */
    for (i = 0; i < gr_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i + CF_n + PF_n)) continue;
      if (!clingo_backend_add_atom(back, &P->gr_PF.d[i], &a)) goto theta_cleanup;
      if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto theta_cleanup;
    }
    /* Cleanup backend. */
    if (!clingo_backend_end(back)) goto theta_cleanup;
    err_code = 4;
    /* Ground atoms. */
    if(!clingo_control_ground(C, EXACT_DEFAULT_PARTS, 1, NULL, NULL)) goto theta_cleanup;
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

      err_code = 5;
      /* Get the solve handle. */
      if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
        goto solve_error;
      /* Iterate over all stable models. */
      for (m = 0; true; ++m) {
        err_code = 6;
        /* m is the number of stable models according to <P,θ>, i.e. m = |Γ(θ)|. */
        if (!clingo_solve_handle_resume(handle)) goto solve_error;
        if (!clingo_solve_handle_model(handle, &M)) goto solve_error;
        if (M) {
          for (i = 0; i < Q_n; ++i) {
            size_t j;
            query_t *q = (P->Q)+i;
            bool all_e = true, all_q = true, c;

            err_code = 7;
            /* all_e? - are all evidence symbols E from query q in M? */
            for (j = 0; j < q->E_n; ++j) {
              if (!clingo_model_contains(M, q->E[j], &c)) goto solve_error;
              if (is_partial) {
                bool c_a;
                if (!clingo_model_contains(M, q->E_u[j], &c_a)) goto solve_error;
                if (neg_partial_cmp(c, c_a, q->E_s[j])) { all_e = false; break; }
              } else {
                if (c != q->E_s[j]) { all_e = false; break; }
              }
            }
            if (!all_e) continue;
            /* all_q? - are all query symbols Q from query q in M? */
            for (j = 0; j < q->Q_n; ++j) {
              if (!clingo_model_contains(M, q->Q[j], &c)) goto solve_error;
              if (is_partial) {
                bool c_a;
                if (!clingo_model_contains(M, q->Q_u[j], &c_a)) goto solve_error;
                if (neg_partial_cmp(c, c_a, q->Q_s[j])) { all_q = false; break; }
              } else {
                if (c != q->Q_s[j]) { all_q = false; break; }
              }
            }
            ++count_e[i];
            if (all_q) { cond_2[i] = true; ++count_q_e[i]; }
            else { cond_4[i] = true; ++count_partial_q_e[i]; }
          }
        } else break;
      }
      err_code = 8;
      if (!clingo_solve_handle_get(handle, &solve_ret)) goto solve_error;
      goto solve_cleanup;
solve_error:
      ok = false;
solve_cleanup:
      if (!(clingo_solve_handle_close(handle) && ok)) goto theta_cleanup;
    }
    err_code = 9;
    /* Compute ℙ(θ). */
    p = prob_total_choice(P->PF, PF_n, &P->gr_pr, theta >> CF_n);
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
        if (cond_1[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][0], (theta_CF >> j) % 2)) goto theta_cleanup;
          if (!array_double_append(&K[i][0], p)) goto theta_cleanup;
        } if (cond_2[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][1], (theta_CF >> j) % 2)) goto theta_cleanup;
          if (!array_double_append(&K[i][1], p)) goto theta_cleanup;
        } if (cond_3[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][2], (theta_CF >> j) % 2)) goto theta_cleanup;
          if (!array_double_append(&K[i][2], p)) goto theta_cleanup;
        } if (cond_4[i]) {
          for (j = 0; j < CF_n; ++j) if (!array_bool_append(&Pn[i][3], (theta_CF >> j) % 2)) goto theta_cleanup;
          if (!array_double_append(&K[i][3], p)) goto theta_cleanup;
        }
      } else {
        a[i] += cond_1[i]*p;
        b[i] += cond_2[i]*p;
        c[i] += cond_3[i]*p;
        d[i] += cond_4[i]*p;
      }
    }
    clingo_control_free(C);
    continue;
theta_cleanup:
    clingo_control_free(C);
    goto cleanup;
  }

  err_code = 10;
  for (i = 0; i < Q_n; ++i) {
    if (has_credal) {
      if (P->Q[i].E_n == 0) {
        double _a, _b;
        bf(X, Pn[i][0].d, Pn[i][1].d, K[i][0].d, K[i][1].d, L_CF, U_CF, K[i][0].n, K[i][1].n, CF_n, &_a, &_b, true);
        R[i][0] = _a, R[i][1] = _b;
      } else {
        size_t _a = K[i][0].n, _b = K[i][1].n, _c = K[i][2].n, _d = K[i][3].n;
        if (_b + _d == 0) {
          fputws(L"Fail: ℙ(E) = 0!\n", stdout);
          R[i][0] = -INFINITY, R[i][1] = INFINITY;
        } else {
          if ((_b + _c == 0) && (_d > 0)) R[i][0] = 0, R[i][1] = 0;
          else if ((_a + _d == 0) && (_b > 0)) R[i][0] = 1, R[i][1] = 1;
          else {
            double min, max;
            bf_minmax(X, Pn[i][0].d, Pn[i][1].d, Pn[i][2].d, Pn[i][3].d, K[i][0].d, K[i][1].d,
                K[i][2].d, K[i][3].d, L_CF, U_CF, _a, _b, _c, _d, CF_n, &min, &max);
            /*bf(X, Pn[i][0].d, Pn[i][3].d, K[i][0].d, K[i][3].d, L_CF, U_CF, _a, _d, CF_n, &min, &max, false);*/
            R[i][0] = min, R[i][1] = max;
          }
        }
      }
    } else {
      double _a = a[i], _b = b[i], _c = c[i], _d = d[i];
      if (P->Q[i].E_n == 0) R[i][0] = _a, R[i][1] = _b;
      else {
        if (_b + _d == 0) {
          fputws(L"Fail: ℙ(E) = 0!\n", stdout);
          R[i][0] = -INFINITY, R[i][1] = INFINITY;
        } else {
          if ((_b + _c == 0) && (_d > 0)) R[i][0] = 0, R[i][1] = 0;
          else if ((_a + _d == 0) && (_b > 0)) R[i][0] = 1, R[i][1] = 1;
          else R[i][0] = _a/(_a + _d), R[i][1] = _b/(_b + _c);
        }
      }
    }
    print_query(P->Q+i); wprintf(L" = [%f, %f]\n", R[i][0], R[i][1]);
  }

  exact_num_ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d|%lu: %s\n", clingo_error_code(), err_code, clingo_error_message());
  free(cond_1); free(cond_2); free(cond_3); free(cond_4);
  free(count_q_e); free(count_e); free(count_partial_q_e);
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
  } else {
    free(a); free(b); free(c); free(d);
  }
  return exact_num_ok;
}

#define EXACT_ENUM 0

static PyObject* exact_opt(PyObject *self, PyObject *args, PyObject *kwargs, int choice) {
  program_t p = {0};
  PyObject *py_P, *py_R = NULL;
  double (*R)[2] = NULL;
  size_t i;
  bool r = false, parallel = true, lstable_sat = true;
  const char *psem_arg = "credal";
  static char *kwlist[] = { "", "parallel", "lstable_sat", "psemantics", NULL };
  psemantics_t psem = CREDAL_SEMANTICS;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|bbs", kwlist, &py_P, &parallel, &lstable_sat,
        &psem_arg))
    return NULL;

  if (!strcmp(psem_arg, "plog")) { psem = PLOG_SEMANTICS; }
  else if (strcmp(psem_arg, "credal")) {
    PyErr_SetString(PyExc_ValueError, "psemantics must either be \"credal\" or \"plog\"!");
    goto cleanup;
  }

  if (!from_python_program(py_P, &p)) return NULL;

  R = (double (*)[2]) malloc(p.Q_n*sizeof(*R));
  if (!R) goto cleanup;

  if (needs_ground(&p)) {
    if (!ground(&p)) goto cleanup;
    if (p.stable) if (!ground(p.stable)) goto cleanup;
  }

  if (parallel) {
    if (!exact_enum(&p, R, lstable_sat, psem)) goto badval;
  } else {
    if (!exact_enum_seq(&p, R)) goto badval;
  }

  py_R = PyTuple_New(p.Q_n);
  if (!py_R) {
    PyErr_SetString(PyExc_MemoryError, "could not create new py_R tuple!");
    goto cleanup;
  } for (i = 0; i < p.Q_n; ++i) {
    PyObject *py_R_i = PyTuple_New(2);
    if (!py_R_i) {
      PyErr_SetString(PyExc_MemoryError, "could not create new py_R_i tuple!");
      goto cleanup;
    }
    PyTuple_SET_ITEM(py_R_i, 0, PyFloat_FromDouble(R[i][0]));
    PyTuple_SET_ITEM(py_R_i, 1, PyFloat_FromDouble(R[i][1]));
    PyTuple_SET_ITEM(py_R, i, py_R_i);
  }
  r = true;
  goto cleanup;
badval:
  PyErr_SetString(PyExc_Exception, "clingo or unknown error!");
cleanup:
  free_program_contents(&p);
  free(R);
  if (!r) Py_XDECREF(py_R);
  return r ? py_R : NULL;
}

static inline PyObject* exact(PyObject *self, PyObject *args, PyObject *kwargs) {
  return exact_opt(self, args, kwargs, EXACT_ENUM);
}

static PyMethodDef CexactMethods[] = {
  {"exact", (PyCFunction)(void(*)(void)) exact, METH_VARARGS | METH_KEYWORDS,
    "Runs exact inference in order to answer the queries in `P`."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef cexactmodule = {
  PyModuleDef_HEAD_INIT,
  "cexact",
  "Exact inference functions from the C side.",
  -1,
  CexactMethods,
};

PyMODINIT_FUNC PyInit_cexact(void) {
  PyObject *m;

  m = PyModule_Create(&cexactmodule);
  if (!m) return NULL;
  if (import_cprogram() < 0) return NULL;
  if (import_carray() < 0) return NULL;
  if (import_coptimize() < 0) return NULL;
  if (import_cutils() < 0) return NULL;
  if (import_cground() < 0) return NULL;

  return m;
}

