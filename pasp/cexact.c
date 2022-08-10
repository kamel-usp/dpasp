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

#include "../thpool/thpool.h"

static double prob_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr, unsigned long long int theta) {
  size_t i = 0, m = gr_pr->n;
  double p = 1.0;
  bool t;
  for (; i < n; ++i) {
    t = (theta >> i) % 2;
    p *= t*phi[i].p + (!t)*(1.0-phi[i].p);
  }
  for (i = 0; i < m; ++i) {
    t = (theta >> (n + i)) % 2;
    p *= t*(gr_pr->d[i]) + (!t)*(1.0-(gr_pr->d[i]));
  }
  return p;
}

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

static inline bool setup_control(clingo_control_t **C, bool parallelize_clingo) {
  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;
  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, C)) return false;
  /* Get the control's configuration. */
  if (!clingo_control_configuration(*C, &cfg)) return false;
  /* Set to enumerate all stable models. */
  if (!clingo_configuration_root(cfg, &cfg_root)) return false;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) return false;
  if (!clingo_configuration_value_set(cfg, cfg_sub, "0")) return false;
  if (parallelize_clingo) {
    /* Set parallel_mode to "NUM_PROCS,compete", where NUM_PROCS is the #procs in this machine. */
    if (!clingo_configuration_map_at(cfg, cfg_root, "solve.parallel_mode", &cfg_sub)) return false;
    if (!clingo_configuration_value_set(cfg, cfg_sub, NUM_PROCS_CONFIG_STR)) return false;
  }
  return true;
}

static inline bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n) {
  *cond_1 = (bool*) malloc(n);
  if (!(*cond_1)) return false;
  *cond_2 = (bool*) malloc(n);
  if (!(*cond_2)) return false;
  *cond_3 = (bool*) malloc(n);
  if (!(*cond_3)) return false;
  *cond_4 = (bool*) malloc(n);
  if (!(*cond_4)) return false;
  return true;
}

static inline bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n) {
  *count_q_e = (size_t*) malloc(n);
  if (!(*count_q_e)) return false;
  *count_e = (size_t*) malloc(n);
  if (!(*count_e)) return false;
  *count_partial_q_e = (size_t*) malloc(n);
  if (!(*count_partial_q_e)) return false;
  return true;
}

static inline bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s) {
  *a = (double*) calloc(n, s);
  if (!(*a)) return false;
  *b = (double*) calloc(n, s);
  if (!(*b)) return false;
  *c = (double*) calloc(n, s);
  if (!(*c)) return false;
  *d = (double*) calloc(n, s);
  if (!(*d)) return false;
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
  return !_x; /* ≡ !(x && _x) && !(!x && _x); */
}

typedef struct storage {
  bool *cond_1, *cond_2, *cond_3, *cond_4;
  size_t *count_q_e, *count_e, *count_partial_q_e;
  double *a, *b, *c, *d;
  array_bool_t (*Pn)[4];
  array_double_t (*K)[4];
  program_t *P;
  unsigned long long int theta;
  bool fail, *busy_procs;
  size_t pid;
  pthread_mutex_t *mu, *wakeup;
  pthread_cond_t *avail;
} storage_t;

static inline bool init_storage(storage_t *s, program_t *P, array_bool_t (*Pn)[4],
    array_double_t (*K)[4], size_t id, bool *busy_procs, pthread_mutex_t *mu,
    pthread_mutex_t *wakeup, pthread_cond_t *avail) {
  s->cond_1 = s->cond_2 = s->cond_3 = s->cond_4 = NULL;
  s->count_q_e = s->count_e = s->count_partial_q_e = NULL;
  s->a = s->b = s->c = s->d = NULL;
  s->Pn = Pn; s->K = K; s->P = P;
  s->mu = mu; s->wakeup = wakeup; s->avail = avail;
  if (!setup_conds(&s->cond_1, &s->cond_2, &s->cond_3, &s->cond_4, P->Q_n*sizeof(bool))) goto error;
  if (!setup_counts(&s->count_q_e, &s->count_e, &s->count_partial_q_e, P->Q_n*sizeof(size_t))) goto error;
  if (!P->CF_n) { if (!setup_abcd(&s->a, &s->b, &s->c, &s->d, P->Q_n, sizeof(double))) goto error; }
  s->busy_procs = busy_procs;
  s->pid = id;
  s->fail = false;
  return true;
error:
  PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for init_storage!");
  return false;
}

static inline void free_storage_contents(storage_t *s) {
  free(s->cond_1); free(s->cond_2); free(s->cond_3); free(s->cond_4);
  free(s->count_q_e); free(s->count_e); free(s->count_partial_q_e);
  if (!s->P->CF_n) { free(s->a); free(s->b); free(s->c); free(s->d); }
}

#define DEBUG_PRINT(pid, msg) wprintf(L"pid %d: " msg "\n", pid);

static void compute_total_choice(void *data) {
  storage_t *st = (storage_t*) data;
  size_t i, m;
  clingo_control_t *C = NULL;
  clingo_backend_t *back = NULL;
  program_t *P = st->P;
  size_t CF_n = P->CF_n, PF_n = P->PF_n, gr_n = P->gr_pr.n;
  size_t Q_n = P->Q_n, Q_n_bytes = Q_n*sizeof(size_t);
  unsigned long long int theta = st->theta, theta_CF = theta & ((1 << CF_n)-1);
  bool *cond_1 = st->cond_1, *cond_2 = st->cond_2, *cond_3 = st->cond_3, *cond_4 = st->cond_4;
  size_t *count_q_e = st->count_q_e, *count_e = st->count_e, *count_partial_q_e = st->count_partial_q_e;
  double *a = st->a, *b = st->b, *c = st->c, *d = st->d, p;
  bool is_partial = P->sem, has_credal = P->CF_n;
  array_bool_t (*Pn)[4] = st->Pn;
  array_double_t (*K)[4] = st->K;

  if (!setup_control(&C, false)) goto cleanup;
  /* Add the purely logical part. */
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto cleanup;
  /* Add grounded probabilistic rules. */
  if (P->gr_P.d) if (!clingo_control_add(C, "base", NULL, 0, P->gr_P.d)) goto cleanup;
  /* Get the control's backend. */
  if (!clingo_control_backend(C, &back)) goto cleanup;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) goto cleanup;
  /* Add the credal facts according to the total rule. */
  for (i = 0; i < CF_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the probabilistic facts according to the total rule. */
  for (i = 0; i < PF_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i + CF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the grounded probabilistic rules according to the total rule. */
  for (i = 0; i < gr_n; ++i) {
    clingo_atom_t a;
    if (!IS_TRUE(theta, i + CF_n + PF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->gr_PF.d[i], &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Cleanup backend. */
  if (!clingo_backend_end(back)) goto cleanup;
  /* Ground atoms. */
  if(!clingo_control_ground(C, EXACT_DEFAULT_PARTS, 1, NULL, NULL)) goto cleanup;
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

static bool exact_enum(program_t *P, double (*R)[2]) {
  bool has_credal = P->CF_n > 0;
  double *a, *b, *c, *d = c = b = a = NULL;
  size_t Q_n = P->Q_n, gr_n = P->gr_pr.n, i;
  size_t total_choice_n = has_credal ? P->PF_n+P->CF_n+gr_n : P->PF_n+gr_n;
  unsigned long long int theta, theta_max;
  array_bool_t (*Pn)[4] = NULL;
  array_double_t (*K)[4] = NULL;
  double *X, *L_CF, *U_CF = L_CF = X = NULL;
  threadpool pool = thpool_init(NUM_PROCS);
  bool busy_procs[NUM_PROCS] = {0}, exact_num_ok;
  storage_t S[NUM_PROCS];
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wakeup = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t avail = PTHREAD_COND_INITIALIZER;

  if (total_choice_n > 62) {
    fputws(L"exact inference only supports up to 62 probabilistic objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  if (has_credal) {
    if (!setup_credal(&L_CF, &U_CF, &X, P)) goto cleanup;
    if (!setup_polynomial(&Pn, &K, P)) goto cleanup;
  }

  for (i = 0; i < NUM_PROCS; ++i)
    if (!init_storage(&S[i], P, Pn, K, i, busy_procs, &mu, &wakeup, &avail)) goto cleanup;

  for (theta = 0; theta < theta_max; ++theta) {
    size_t id = (size_t) -1;
    /* The line below does not produce a problematic race condition since it will, at worst, skip
     * the i-th busy_procs and have to iterate NUM_PROCS all over again. */
    pthread_mutex_lock(&wakeup);
    while (true) {
      for (i = 0, id = (size_t) -1; i < NUM_PROCS; ++i) {
        if (!busy_procs[i]) { id = i; break; }
      }
      if (id != (size_t) -1) break;
      pthread_cond_wait(&avail, &wakeup);
    }
    busy_procs[id] = true;
    pthread_mutex_unlock(&wakeup);
    S[id].theta = theta;
    if (S[id].fail || thpool_add_work(pool, compute_total_choice, (void*) &S[id])) {
      goto cleanup;
    }
  }
  thpool_wait(pool);

  if (!has_credal) {
    a = S[0].a; b = S[0].b; c = S[0].c; d = S[0].d;
    for (i = 1; i < NUM_PROCS; ++i) {
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
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  pthread_mutex_destroy(&mu); pthread_mutex_destroy(&wakeup); pthread_cond_destroy(&avail);
  thpool_destroy(pool);
  for (i = 0; i < NUM_PROCS; ++i) free_storage_contents(&S[i]);
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

#define EXACT_ENUM 0

static PyObject* exact_opt(PyObject *self, PyObject *args, int choice) {
  program_t p;
  PyObject *py_P, *py_R = NULL;
  double (*R)[2] = NULL;
  size_t i;
  bool r = false;

  if (!PyArg_ParseTuple(args, "O", &py_P)) return NULL;
  if (!from_python_program(py_P, &p)) return NULL;

  R = (double (*)[2]) malloc(p.Q_n*sizeof(*R));
  if (!R) goto cleanup;

  if (needs_ground(&p)) {
    if (!ground(&p)) goto cleanup;
  }

  if (!exact_enum(&p, R)) goto badval;

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

static inline PyObject* exact(PyObject *self, PyObject *args) {
  return exact_opt(self, args, EXACT_ENUM);
}

static PyMethodDef CexactMethods[] = {
  {"exact", exact, METH_VARARGS, "Runs exact inference in order to answer the queries in `P`."},
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

#ifdef PASP_DEBUG
int main(void) { return 0; }
#endif
