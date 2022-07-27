#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>

#include "cprogram.h"

static double prob_total_choice(prob_fact_t *phi, bool *theta, size_t n) {
  size_t i = 0;
  double p = 1.0;
  for (; i < n; ++i) p *= theta[i]*phi[i].p + (!theta[i])*(1.0-phi[i].p);
  return p;
}

const clingo_part_t EXACT_DEFAULT_PARTS[] = {{"base", NULL, 0}};

static void undef_atom_ignore(clingo_warning_t code, const char *msg, void *data) {
  if (code == clingo_warning_atom_undefined) return;
  wprintf(L"clingo | error code %d: %s\n", code, msg);
  (void) data;
}

static bool exact_enum(program_t *P, double (*R)[2]) {
  bool *cond_1, *cond_2, *cond_3, *cond_4 = cond_3 = cond_2 = cond_1 = NULL, exact_num_ok = false;
  size_t *count_q_e, *count_e, *count_partial_q_e = count_e = count_q_e = NULL;
  double *a, *b, *c, *d = c = b = a = NULL, p;
  size_t Q_n = P->Q_n, total_choice_n = P->PF_n, Q_n_bytes = Q_n*sizeof(size_t), i;
  unsigned long long int theta_i, theta_max;
  bool *theta = NULL;

  if (total_choice_n > 63) {
    fputws(L"exact inference only supports up to 63 probabilistic objects (facts or propositional rules)!\n", stdout);
    return false;
  } else theta_max = 1 << total_choice_n;

  cond_1 = (bool*) malloc(Q_n*sizeof(bool));
  if (!cond_1) goto cleanup;
  cond_2 = (bool*) malloc(Q_n*sizeof(bool));
  if (!cond_2) goto cleanup;
  cond_3 = (bool*) malloc(Q_n*sizeof(bool));
  if (!cond_3) goto cleanup;
  cond_4 = (bool*) malloc(Q_n*sizeof(bool));
  if (!cond_4) goto cleanup;

  count_q_e = (size_t*) malloc(Q_n_bytes);
  if (!count_q_e) goto cleanup;
  count_e = (size_t*) malloc(Q_n_bytes);
  if (!count_e) goto cleanup;
  count_partial_q_e = (size_t*) malloc(Q_n_bytes);
  if (!count_partial_q_e) goto cleanup;

  a = (double*) calloc(Q_n, sizeof(double));
  if (!a) goto cleanup;
  b = (double*) calloc(Q_n, sizeof(double));
  if (!b) goto cleanup;
  c = (double*) calloc(Q_n, sizeof(double));
  if (!c) goto cleanup;
  d = (double*) calloc(Q_n, sizeof(double));
  if (!d) goto cleanup;

  theta = (bool*) malloc(total_choice_n*sizeof(bool));
  if (!theta) goto cleanup;

  /* TODO: ground probabilistic rules with free variables. */

  for (theta_i = 0; theta_i < theta_max; ++theta_i) {
    size_t m;
    clingo_control_t *C = NULL;
    clingo_configuration_t *cfg = NULL;
    clingo_id_t cfg_root, cfg_sub;
    clingo_backend_t *back = NULL;

    /* Set the total choice accordingly. */
    for (i = 0; i < total_choice_n; ++i) theta[i] = (theta_i >> i) % 2;
    /* Create new clingo controller. */
    if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C))
      goto theta_cleanup;
    /* Get the control's configuration. */
    if (!clingo_control_configuration(C, &cfg)) goto theta_cleanup;
    /* Set to enumerate all stable models. */
    if (!clingo_configuration_root(cfg, &cfg_root)) goto theta_cleanup;
    if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) goto theta_cleanup;
    if (!clingo_configuration_value_set(cfg, cfg_sub, "0")) goto theta_cleanup;
    /* Add the purely logical part. */
    if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto theta_cleanup;
    /* Get the control's backend. */
    if (!clingo_control_backend(C, &back)) goto theta_cleanup;
    /* Startup the backend. */
    if (!clingo_backend_begin(back)) goto theta_cleanup;
    /* Add the probabilistic facts according to the total rule. */
    for (i = 0; i < P->PF_n; ++i) {
      clingo_atom_t a;
      if (!theta[i]) continue;
      if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto theta_cleanup;
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
    p = prob_total_choice(P->PF, theta, total_choice_n);
    for (i = 0; i < Q_n; ++i) {
      /* Evaluate counts to judge whether cond_1 and/or cond_3 are true. */
      if (count_e[i] == m || P->Q->E_n == 0) {
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
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  free(cond_1); free(cond_2); free(cond_3); free(cond_4);
  free(count_q_e); free(count_e); free(count_partial_q_e);
  free(a); free(b); free(c); free(d);
  free(theta);
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

  if ((choice == EXACT_ENUM) && !exact_enum(&p, R)) goto cleanup;

  py_R = PyTuple_New(p.Q_n);
  if (!py_R) goto error;
  for (i = 0; i < p.Q_n; ++i) {
    PyObject *py_R_i = PyTuple_New(2);
    if (!py_R_i) goto error;
    PyTuple_SET_ITEM(py_R_i, 0, PyFloat_FromDouble(R[i][0]));
    PyTuple_SET_ITEM(py_R_i, 1, PyFloat_FromDouble(R[i][1]));
    PyTuple_SET_ITEM(py_R, i, py_R_i);
  }
  r = true;
  goto cleanup;
error:
  PyErr_SetString(PyExc_MemoryError, "could not create new tuple!");
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

  return m;
}

#ifdef PASP_DEBUG
int main(void) { return 0; }
#endif
