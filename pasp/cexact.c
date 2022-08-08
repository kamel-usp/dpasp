#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>

#include "cprogram.h"
#include "carray.h"
#include "coptimize.h"
#include "cutils.h"
#include "cground.h"

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

#define IS_TRUE(t, i) (((t) >> (i)) % 2)

static inline bool setup_control(clingo_control_t **C) {
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

static bool exact_enum(program_t *P, double (*R)[2]) {
  bool *cond_1, *cond_2, *cond_3, *cond_4 = cond_3 = cond_2 = cond_1 = NULL, exact_num_ok = false;
  size_t *count_q_e, *count_e, *count_partial_q_e = count_e = count_q_e = NULL;
  double *a, *b, *c, *d = c = b = a = NULL, p;
  size_t gr_n = P->gr_pr.n, err_code = 0;
  size_t Q_n = P->Q_n, total_choice_n = P->PF_n + gr_n, Q_n_bytes = Q_n*sizeof(size_t), i;
  unsigned long long int theta, theta_max;

  if (total_choice_n > 62) {
    fputws(L"exact inference only supports up to 62 probabilistic objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  err_code = 1;

  if (!setup_conds(&cond_1, &cond_2, &cond_3, &cond_4, Q_n*sizeof(bool))) goto cleanup;
  if (!setup_counts(&count_q_e, &count_e, &count_partial_q_e, Q_n_bytes)) goto cleanup;
  if (!setup_abcd(&a, &b, &c, &d, Q_n, sizeof(double))) goto cleanup;

  /* TODO: ground probabilistic rules with free variables. */

  for (theta = 0; theta < theta_max; ++theta) {
    size_t m;
    clingo_control_t *C = NULL;
    clingo_backend_t *back = NULL;

    err_code = 2;
    if (!setup_control(&C)) goto theta_cleanup;
    /* Add the purely logical part. */
    if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto theta_cleanup;
    /* Add grounded probabilistic rules. */
    if (P->gr_P.d) if (!clingo_control_add(C, "base", NULL, 0, P->gr_P.d)) goto theta_cleanup;
    /* Get the control's backend. */
    if (!clingo_control_backend(C, &back)) goto theta_cleanup;
    /* Startup the backend. */
    err_code = 3;
    if (!clingo_backend_begin(back)) goto theta_cleanup;
    /* Add the probabilistic facts according to the total rule. */
    for (i = 0; i < P->PF_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i)) continue;
      if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto theta_cleanup;
      if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto theta_cleanup;
    }
    /* Add the grounded probabilistic rules according to the total rule. */
    for (i = 0; i < gr_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i + P->PF_n)) continue;
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
              if (c != q->E_s[j]) { all_e = false; break; }
            }
            if (!all_e) continue;
            /* all_q? - are all query symbols Q from query q in M? */
            for (j = 0; j < q->Q_n; ++j) {
              if (!clingo_model_contains(M, q->Q[j], &c)) goto solve_error;
              if (c != q->Q_s[j]) { all_q = false; break; }
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
    p = prob_total_choice(P->PF, P->PF_n, &P->gr_pr, theta);
    for (i = 0; i < Q_n; ++i) {
      /* Evaluate counts to judge whether cond_1 and/or cond_3 are true. */
      if (count_e[i] == m || P->Q[i].E_n == 0) {
        /* All stable models satisfy Q and E completely. */
        if (count_q_e[i] == m) cond_1[i] = true;
        /* All stable models satisfy E, but none satisfies Q completely. */
        if (count_partial_q_e[i] == m) cond_3[i] = true;
      }
      /* Add probability ℙ(θ) according to model satisfiabilities. */
      a[i] += cond_1[i]*p;
      b[i] += cond_2[i]*p;
      c[i] += cond_3[i]*p;
      d[i] += cond_4[i]*p;
    }
    clingo_control_free(C);
    continue;
theta_cleanup:
    clingo_control_free(C);
    goto cleanup;
  }

  err_code = 10;
  for (i = 0; i < Q_n; ++i) {
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
    print_query(P->Q+i); wprintf(L" = [%f, %f]\n", R[i][0], R[i][1]);
  }

  exact_num_ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d|%lu: %s\n", clingo_error_code(), err_code, clingo_error_message());
  free(cond_1); free(cond_2); free(cond_3); free(cond_4);
  free(count_q_e); free(count_e); free(count_partial_q_e);
  free(a); free(b); free(c); free(d);
  return exact_num_ok;
}

static bool exact_sym(program_t *P, double (*R)[2]) {
  bool *cond_1, *cond_2, *cond_3, *cond_4 = cond_3 = cond_2 = cond_1 = NULL, exact_num_ok = false;
  size_t *count_q_e, *count_e, *count_partial_q_e = count_e = count_q_e = NULL;
  size_t CF_n = P->CF_n, i, PF_n = P->PF_n, gr_n = P->gr_pr.n;
  size_t Q_n = P->Q_n, total_choice_n = PF_n+CF_n+gr_n, Q_n_bytes = Q_n*sizeof(size_t);
  unsigned long long int theta, theta_max;
  double p, *X, *L_CF, *U_CF = L_CF = X = NULL;
  array_bool_t (*Pn)[4] = NULL;
  array_double_t (*K)[4] = NULL;

  if (total_choice_n > 62) {
    fputws(L"exact inference only supports up to 62 probabilistic/credal objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  if (!setup_conds(&cond_1, &cond_2, &cond_3, &cond_4, Q_n*sizeof(bool))) goto cleanup;
  if (!setup_counts(&count_q_e, &count_e, &count_partial_q_e, Q_n_bytes)) goto cleanup;

  L_CF = (double*) malloc(CF_n*sizeof(double));
  if (!L_CF) goto cleanup;
  U_CF = (double*) malloc(CF_n*sizeof(double));
  if (!U_CF) goto cleanup;
  for (i = 0; i < CF_n; ++i) L_CF[i] = P->CF[i].l, U_CF[i] = P->CF[i].u;

  X = (double*) malloc(CF_n*sizeof(double));
  if (!X) goto cleanup;

  Pn = (array_bool_t(*)[4]) malloc(Q_n*sizeof(*Pn));
  if (!Pn) goto cleanup;
  K = (array_double_t(*)[4]) malloc(Q_n*sizeof(*K));
  if (!K) goto cleanup;
  for (i = 0; i < Q_n; ++i)
    if (!(array_bool_init(&Pn[i][0]) && array_bool_init(&Pn[i][1]) && array_bool_init(&Pn[i][2])
      && array_bool_init(&Pn[i][3]) && array_double_init(&K[i][0]) && array_double_init(&K[i][1])
      && array_double_init(&K[i][2]) && array_double_init(&K[i][3]))) goto cleanup;

  /* TODO: ground probabilistic rules with free variables. */

  for (theta = 0; theta < theta_max; ++theta) {
    size_t m;
    clingo_control_t *C = NULL;
    clingo_backend_t *back = NULL;
    unsigned long long int theta_CF = theta & ((1 << CF_n)-1);

    if (!setup_control(&C)) goto theta_cleanup;
    /* Add the purely logical part. */
    if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto theta_cleanup;
    /* Add grounded probabilistic rules. */
    if (P->gr_P.d) if (!clingo_control_add(C, "base", NULL, 0, P->gr_P.d)) goto theta_cleanup;
    /* Get the control's backend. */
    if (!clingo_control_backend(C, &back)) goto theta_cleanup;
    /* Startup the backend. */
    if (!clingo_backend_begin(back)) goto theta_cleanup;
    /* Add the credal facts according to the total rule. */
    for (i = 0; i < CF_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i)) continue;
      if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) goto theta_cleanup;
      if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto theta_cleanup;
    }
    /* Add the probabilistic facts according to the total rule. */
    for (; i < CF_n+PF_n; ++i) {
      clingo_atom_t a;
      if (!IS_TRUE(theta, i)) continue;
      if (!clingo_backend_add_atom(back, &P->PF[i-CF_n].cl_f, &a)) goto theta_cleanup;
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
              if (c != q->E_s[j]) { all_e = false; break; }
            }
            if (!all_e) continue;
            /* all_q? - are all query symbols Q from query q in M? */
            for (j = 0; j < q->Q_n; ++j) {
              if (!clingo_model_contains(M, q->Q[j], &c)) goto solve_error;
              if (c != q->Q_s[j]) { all_q = false; break; }
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
      if (!(clingo_solve_handle_close(handle) && ok)) goto theta_cleanup;
    }
    /* Compute ℙ(θ). */
    p = prob_total_choice(P->PF, PF_n, &P->gr_pr, theta >> CF_n);
    for (i = 0; i < Q_n; ++i) {
      size_t j;
      /* Evaluate counts to judge whether cond_1 and/or cond_3 are true. */
      if (count_e[i] == m || P->Q[i].E_n == 0) {
        /* All stable models satisfy Q and E completely. */
        if (count_q_e[i] == m) cond_1[i] = true;
        /* All stable models satisfy E, but none satisfies Q completely. */
        if (count_partial_q_e[i] == m) cond_3[i] = true;
      }
      /* Add probability ℙ(θ) according to model satisfiabilities. */
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
    }
    clingo_control_free(C);
    continue;
theta_cleanup:
    clingo_control_free(C);
    goto cleanup;
  }

  for (i = 0; i < Q_n; ++i) {
    if (P->Q[i].E_n == 0) {
      double a, b;
      bf(X, Pn[i][0].d, Pn[i][1].d, K[i][0].d, K[i][1].d, L_CF, U_CF, K[i][0].n, K[i][1].n, CF_n, &a, &b, true);
      R[i][0] = a, R[i][1] = b;
    } else {
      size_t a = K[i][0].n, b = K[i][1].n, c = K[i][2].n, d = K[i][3].n;
      if (b + d == 0) {
        fputws(L"Fail: ℙ(E) = 0!\n", stdout);
        R[i][0] = -INFINITY, R[i][1] = INFINITY;
      } else {
        if ((b + c == 0) && (d > 0)) R[i][0] = 0, R[i][1] = 0;
        else if ((a + d == 0) && (b > 0)) R[i][0] = 1, R[i][1] = 1;
        else {
          double min, max;
          bf_minmax(X, Pn[i][0].d, Pn[i][1].d, Pn[i][2].d, Pn[i][3].d, K[i][0].d, K[i][1].d,
              K[i][2].d, K[i][3].d, L_CF, U_CF, a, b, c, d, CF_n, &min, &max);
          /*bf(X, Pn[i][0].d, Pn[i][3].d, K[i][0].d, K[i][3].d, L_CF, U_CF, a, d, CF_n, &min, &max, false);*/
          R[i][0] = min, R[i][1] = max;
        }
      }
    }
    print_query(P->Q+i); wprintf(L" = [%f, %f]\n", R[i][0], R[i][1]);
  }

  exact_num_ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success)
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  free(cond_1); free(cond_2); free(cond_3); free(cond_4);
  free(count_q_e); free(count_e); free(count_partial_q_e);
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

  if (p.CF_n > 0) {
    if (!exact_sym(&p, R)) goto badval;
  } else {
    if (!exact_enum(&p, R)) goto badval;
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
