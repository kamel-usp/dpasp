#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <locale.h>
#include <wchar.h>

#define CPROGRAM_MODULE
#include "cprogram.h"

#include "cutils.h"

static inline void print_prob_fact(prob_fact_t *pf) { wprintf(L"%f::%s", pf->p, pf->f); }
static inline void free_prob_fact_contents(prob_fact_t *pf) { if (pf) Py_DECREF(pf->f_obj); }
static inline void free_prob_fact(prob_fact_t *pf) { free_prob_fact_contents(pf); free(pf); }

static inline void print_prob_rule(prob_rule_t *pr) { wprintf(L"%f::%s", pr->p, pr->f); }
static inline void free_prob_rule_contents(prob_rule_t *pr) {
  if (pr) { Py_DECREF(pr->f_obj); Py_XDECREF(pr->unify_obj); }
}
static inline void free_prob_rule(prob_rule_t *pr) { free_prob_rule_contents(pr); free(pr); }

static inline void print_credal_fact(credal_fact_t *cf) { wprintf(L"[%f, %f]::%s", cf->l, cf->u, cf->f); }
static inline void free_credal_fact_contents(credal_fact_t *cf) { if (cf) Py_DECREF(cf->f_obj); }
static inline void free_credal_fact(credal_fact_t *cf) { free_credal_fact_contents(cf); free(cf); }

static inline void print_annot_disj(annot_disj_t *ad) {
  size_t i;
  for (i = 0; i < ad->n; ++i) {
    wprintf(L"%f::%s", ad->P[i], ad->F[i]);
    if (i != ad->n-1) fputws(L"; ", stdout);
  }
}
static inline void free_annot_disj_contents(annot_disj_t *ad) {
  if (!ad) return;
  free(ad->P);
  free(ad->F);
  free(ad->F_obj);
  free(ad->cl_F);
}
static inline void free_annot_disj(annot_disj_t *ad) { free_annot_disj_contents(ad); free(ad); }

static bool print_query_with_buffer(query_t *q, string_t *s) {
  size_t i;
  bool has_E = q->E_n > 0;

  fputws(L"ℙ(", stdout);
  for (i = 0; i < q->Q_n; ++i) {
    if (q->Q_s[i] == QUERY_TERM_NEG) fputws(L"not ", stdout);
    else if (q->Q_s[i] == QUERY_TERM_UND) fputws(L"undef ", stdout);
    if (!string_from_symbol(q->Q[i], s)) return false;
    wprintf(L"%s", s->s, q->Q[i]);
    if (i != q->Q_n-1) fputws(L", ", stdout);
    else {
      if (has_E) fputws(L" | ", stdout);
      else fputws(L")", stdout);
    }
  }
  for(i = 0; i < q->E_n; ++i) {
    if (q->E_s[i] == QUERY_TERM_NEG) fputws(L"not ", stdout);
    else if (q->E_s[i] == QUERY_TERM_UND) fputws(L"undef ", stdout);
    if (!string_from_symbol(q->E[i], s)) return false;
    wprintf(L"%s", s->s, q->E[i]);
    if (i != q->E_n-1) fputws(L", ", stdout);
    else fputws(L")", stdout);
  }

  return true;
}
static inline bool print_query(query_t *Q) {
  string_t s = {NULL, 0};
  bool r = print_query_with_buffer(Q, &s);
  free (s.s);
  return r;
}

static inline void free_query_contents(query_t *Q) {
  if (!Q) return;
  free(Q->Q);
  free(Q->Q_s);
  free(Q->Q_u);
  free(Q->E);
  free(Q->E_s);
  free(Q->E_u);
}
static inline void free_query(query_t *Q) { free_query_contents(Q); free(Q); }

static void print_program(program_t *P) {
  size_t i;
  string_t s = {NULL, 0};
  wprintf(L"<Logic Program:\n%s,\nProbabilistic Facts:\n", P->P);
  for (i = 0; i < P->PF_n; ++i) { print_prob_fact(P->PF + i); fputws(L", ", stdout); }
  fputws(L"\nCredal Facts:\n", stdout);
  for (i = 0; i < P->CF_n; ++i) { print_credal_fact(P->CF + i); fputws(L", ", stdout); }
  fputws(L"\nAnnotated Disjunctions:\n", stdout);
  for (i = 0; i < P->AD_n; ++i) { print_annot_disj(P->AD + i); fputws(L", ", stdout); }
  fputws(L"\nProbabilistic Rules:\n", stdout);
  for (i = 0; i < P->PR_n; ++i) { print_prob_rule(P->PR + i); fputws(L", ", stdout); }
  fputws(L"\nQueries:\n", stdout);
  for (i = 0; i < P->Q_n; ++i) { print_query_with_buffer(P->Q + i, &s); fputws(L", ", stdout); }
  fputws(L">\n", stdout);
  free(s.s);
}
static inline void free_program_contents(program_t *P) {
  size_t i;
  if (!P) return;
  Py_XDECREF(P->P_obj);
  for (i = 0; i < P->PF_n; ++i) free_prob_fact_contents(&P->PF[i]);
  free(P->PF);
  for (i = 0; i < P->PR_n; ++i) free_prob_rule_contents(&P->PR[i]);
  free(P->PR);
  for (i = 0; i < P->Q_n; ++i) free_query_contents(&P->Q[i]);
  free(P->Q);
  for (i = 0; i < P->CF_n; ++i) free_credal_fact_contents(&P->CF[i]);
  free(P->CF);
  for (i = 0; i < P->AD_n; ++i) free_annot_disj_contents(&P->AD[i]);
  free(P->AD);
  array_clingo_symbol_t_free_contents(&P->gr_PF);
  array_char_free_contents(&P->gr_P);
  array_double_free_contents(&P->gr_pr);
  if (P->stable) free_program(P->stable);
}
static inline void free_program(program_t *P) { free_program_contents(P); free(P); }

static bool from_python_prob_rule(PyObject *py_pr, prob_rule_t *pr) {
  PyObject *py_p, *py_f, *py_is_prop, *py_unify = py_is_prop = py_f = py_p = NULL;
  double p;
  const char *f;
  long is_prop_l;
  const char *unify = NULL;
  bool r = false;

  py_p = PyObject_GetAttrString(py_pr, "p");
  if (!py_p) {
    PyErr_SetString(PyExc_AttributeError, "could not access field p of supposed ProbRule object!");
    goto cleanup;
  }
  p = PyFloat_AsDouble(py_p);
  if ((p == -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field p of ProbRule must be a floating-point number!");
    goto cleanup;
  }

  py_f = PyObject_GetAttrString(py_pr, "f");
  if (!py_f) {
    PyErr_SetString(PyExc_AttributeError, "could not access field f of supposed ProbRule object!");
    goto cleanup;
  }
  f = PyUnicode_AsUTF8(py_f);
  if (!f) {
    PyErr_SetString(PyExc_TypeError, "field f of ProbRule must be a string!");
    goto cleanup;
  }

  py_is_prop = PyObject_GetAttrString(py_pr, "is_prop");
  if (!py_is_prop) {
    PyErr_SetString(PyExc_AttributeError, "could not access field is_prop of supposed ProbRule object!");
    goto cleanup;
  }
  is_prop_l = PyLong_AsLong(py_is_prop);
  if ((is_prop_l == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field is_prop of ProbRule must be a bool!");
    goto cleanup;
  }

  if (!is_prop_l) {
    py_unify = PyObject_GetAttrString(py_pr, "unify");
    if (!py_unify) {
      PyErr_SetString(PyExc_AttributeError, "could not access field unify of supposed ProbRule object!");
      goto cleanup;
    }
    unify = PyUnicode_AsUTF8(py_unify);
    if (!unify) {
      PyErr_SetString(PyExc_TypeError, "field unify of ProbRule must be a string!");
      goto cleanup;
    }
  }

  pr->p = p;
  pr->f = f;
  pr->f_obj = py_f;
  pr->is_prop = (bool) is_prop_l;
  pr->unify = unify;
  pr->unify_obj = py_unify;
  r = true;

cleanup:
  Py_XDECREF(py_p);
  if (!r) { Py_XDECREF(py_f); Py_XDECREF(py_unify); }
  Py_XDECREF(py_is_prop);
  return r;
}

static bool from_python_prob_fact(PyObject *py_pf, prob_fact_t *pf) {
  PyObject *py_p, *py_f, *py_cl_f, *py_cl_f_rep = py_cl_f = py_f = py_p = NULL;
  double p;
  const char *f;
  clingo_symbol_t cl_f;
  bool r = false;

  py_p = PyObject_GetAttrString(py_pf, "p");
  if (!py_p) {
    PyErr_SetString(PyExc_AttributeError, "could not access field p of supposed ProbFact object!");
    goto cleanup;
  }
  p = PyFloat_AsDouble(py_p);
  if ((p == -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field p of ProbFact must be a floating-point number!");
    goto cleanup;
  }

  py_f = PyObject_GetAttrString(py_pf, "f");
  if (!py_f) {
    PyErr_SetString(PyExc_AttributeError, "could not access field f of supposed ProbFact object!");
    goto cleanup;
  }
  f = PyUnicode_AsUTF8(py_f);
  if (!f) {
    PyErr_SetString(PyExc_TypeError, "field f of ProbFact must be a string!");
    goto cleanup;
  }

  py_cl_f = PyObject_GetAttrString(py_pf, "cl_f");
  if (!py_cl_f) {
    PyErr_SetString(PyExc_AttributeError, "could not access field cl_f of supposed ProbFact object!");
    goto cleanup;
  }
  py_cl_f_rep = PyObject_GetAttrString(py_cl_f, "_rep");
  if (!py_cl_f_rep) {
    PyErr_SetString(PyExc_AttributeError, "could not access field _rep of supposed Symbol object!");
    goto cleanup;
  }
  cl_f = PyLong_AsUnsignedLong(py_cl_f_rep);
  if ((cl_f == (clingo_symbol_t) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field cl_f of ProbFact must be a Symbol!");
    goto cleanup;
  }

  pf->p = p;
  pf->f = f;
  pf->f_obj = py_f;
  pf->cl_f = cl_f;
  r = true;

cleanup:
  Py_XDECREF(py_p);
  if (!r) Py_XDECREF(py_f);
  Py_XDECREF(py_cl_f_rep);
  Py_XDECREF(py_cl_f);
  return r;
}

static bool from_python_credal_fact(PyObject *py_cf, credal_fact_t *cf) {
  PyObject *py_l, *py_u, *py_f, *py_cl_f, *py_cl_f_rep = py_cl_f = py_f = py_u = py_l = NULL;
  double l, u;
  clingo_symbol_t cl_f;
  const char *f;
  bool r = false;

  py_l = PyObject_GetAttrString(py_cf, "l");
  if (!py_l) {
    PyErr_SetString(PyExc_AttributeError, "could not access field l of supposed CredalFact object!");
    goto cleanup;
  }
  l = PyFloat_AsDouble(py_l);
  if ((l == -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field l of CredalFact must be a floating-point number!");
    goto cleanup;
  }

  py_u = PyObject_GetAttrString(py_cf, "u");
  if (!py_u) {
    PyErr_SetString(PyExc_AttributeError, "could not access field u of supposed CredalFact object!");
    goto cleanup;
  }
  u = PyFloat_AsDouble(py_u);
  if ((u == -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field u of CredalFact must be a floating-point number!");
    goto cleanup;
  }

  py_f = PyObject_GetAttrString(py_cf, "f");
  if (!py_f) {
    PyErr_SetString(PyExc_AttributeError, "could not access field f of supposed CredalFact object!");
    goto cleanup;
  }
  f = PyUnicode_AsUTF8(py_f);
  if (!f) {
    PyErr_SetString(PyExc_TypeError, "field f of CredalFact must be a string!");
    goto cleanup;
  }

  py_cl_f = PyObject_GetAttrString(py_cf, "cl_f");
  if (!py_cl_f) {
    PyErr_SetString(PyExc_AttributeError, "could not access field cl_f of supposed CredalFact object!");
    goto cleanup;
  }
  py_cl_f_rep = PyObject_GetAttrString(py_cl_f, "_rep");
  if (!py_cl_f_rep) {
    PyErr_SetString(PyExc_AttributeError, "could not access field _rep of supposed Symbol object!");
    goto cleanup;
  }
  cl_f = PyLong_AsUnsignedLong(py_cl_f_rep);
  if ((cl_f == (clingo_symbol_t) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field cl_f of CredalFact must be a Symbol!");
    goto cleanup;
  }

  cf->l = l;
  cf->u = u;
  cf->f = f;
  cf->f_obj = py_f;
  cf->cl_f = cl_f;
  r = true;

cleanup:
  Py_XDECREF(py_l);
  Py_XDECREF(py_u);
  if (!r) Py_XDECREF(py_f);
  Py_XDECREF(py_cl_f_rep);
  Py_XDECREF(py_cl_f);
  return r;
}

static bool from_python_query(PyObject *py_q, query_t *q, semantics_t sem) {
  PyObject *py_Q, *py_E, *py_Q_L, *py_E_L = py_Q_L = py_E = py_Q = NULL;
  clingo_symbol_t *Q, *E = Q = NULL;
  clingo_symbol_t *Q_u, *E_u = Q_u = NULL;
  char *Q_s, *E_s = Q_s = NULL;
  size_t i;

  py_Q = PyObject_GetAttrString(py_q, "Q");
  if (!py_Q) {
    PyErr_SetString(PyExc_AttributeError, "could not access field Q of supposed Query object!");
    goto cleanup;
  }
  py_E = PyObject_GetAttrString(py_q, "E");
  if (!py_E) {
    PyErr_SetString(PyExc_AttributeError, "could not access field E of supposed Query object!");
    goto cleanup;
  }

  py_Q_L = PySequence_Fast(py_Q, "field Query.Q must either be a list or tuple!");
  if (!py_Q_L) goto cleanup;
  py_E_L = PySequence_Fast(py_E, "field Query.E must either be a list or tuple!");
  if (!py_E) goto cleanup;

  q->Q_n = PySequence_Fast_GET_SIZE(py_Q_L);
  q->E_n = PySequence_Fast_GET_SIZE(py_E_L);

  Q = (clingo_symbol_t*) malloc(q->Q_n*sizeof(clingo_symbol_t));
  if (!Q) goto nomem;
  E = (clingo_symbol_t*) malloc(q->E_n*sizeof(clingo_symbol_t));
  if (!E) goto nomem;
  Q_s = (char*) malloc(q->Q_n*sizeof(char));
  if (!Q_s) goto nomem;
  E_s = (char*) malloc(q->E_n*sizeof(char));
  if (!E_s) goto nomem;
  if (sem) {
    Q_u = (clingo_symbol_t*) malloc(q->Q_n*sizeof(clingo_symbol_t));
    if (!Q_u) goto nomem;
    E_u = (clingo_symbol_t*) malloc(q->E_n*sizeof(clingo_symbol_t));
    if (!E_u) goto nomem;
  }

  for (i = 0; i < q->Q_n; ++i) {
    PyObject *rep = NULL;
    PyObject *t = PySequence_Fast(PySequence_Fast_GET_ITEM(py_Q_L, i), "elements of Query.Q must either be tuples or lists!");
    if (!t) goto cleanup;
    if (PySequence_Fast_GET_SIZE(t) < 3) {
      PyErr_SetString(PyExc_ValueError, "Query.Q elements must be tuples (or lists) of size 3!");
      goto cleanup;
    }
    rep = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 0), "_rep");
    if (!rep) goto cleanup;
    Q[i] = PyLong_AsUnsignedLong(rep);
    Q_s[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(t, 1));
    if (sem) { /* sem != (STABLE_SEMANTICS = 0) */
      PyObject *u = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 2), "_rep");
      if (!u) { Py_DECREF(rep); Py_DECREF(t); goto cleanup; }
      Q_u[i] = PyLong_AsUnsignedLong(u);
      Py_DECREF(u);
    }
    Py_DECREF(rep);
    Py_DECREF(t);
  }

  for (i = 0; i < q->E_n; ++i) {
    PyObject *rep = NULL;
    PyObject *t = PySequence_Fast(PySequence_Fast_GET_ITEM(py_E_L, i), "elements of Query.E must either be tuples or lists!");
    if (!t) goto cleanup;
    if (PySequence_Fast_GET_SIZE(t) < 2) {
      PyErr_SetString(PyExc_ValueError, "Query.E elements must be tuples (or lists) of size 2!");
      goto cleanup;
    }
    rep = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 0), "_rep");
    if (!rep) goto cleanup;
    E[i] = PyLong_AsUnsignedLong(rep);
    E_s[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(t, 1));
    if (sem) { /* sem != (STABLE_SEMANTICS = 0) */
      PyObject *u = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 2), "_rep");
      if (!u) { Py_DECREF(rep); Py_DECREF(t); goto cleanup; }
      E_u[i] = PyLong_AsUnsignedLong(u);
      Py_DECREF(u);
    }
    Py_DECREF(rep);
    Py_DECREF(t);
  }

  Py_DECREF(py_Q);
  Py_DECREF(py_E);
  Py_DECREF(py_Q_L);
  Py_DECREF(py_E_L);

  q->Q = Q;
  q->Q_s = Q_s;
  q->Q_u = Q_u;
  q->E = E;
  q->E_s = E_s;
  q->E_u = E_u;

  return true;
nomem:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
cleanup:
  Py_XDECREF(py_Q);
  Py_XDECREF(py_E);
  Py_XDECREF(py_Q_L);
  Py_XDECREF(py_E_L);
  free(Q); free(E);
  free(Q_s); free(E_s);
  free(Q_u); free(E_u);
  return false;
}

static bool from_python_ad(PyObject *py_ad, annot_disj_t *ad) {
  PyObject *py_P, *py_F, *py_cl_F = py_F = py_P = NULL;
  PyObject *py_P_L, *py_F_L, *py_cl_F_L = py_F_L = py_P_L = NULL;
  PyObject **F_obj = NULL;
  double *P = NULL;
  const char **F = NULL;
  clingo_symbol_t *cl_F = NULL;
  size_t i, n;

  py_P = PyObject_GetAttrString(py_ad, "P");
  if (!py_P) {
    PyErr_SetString(PyExc_AttributeError, "could not access field P of supposed AnnotatedDisjunction object!");
    goto cleanup;
  }
  py_F = PyObject_GetAttrString(py_ad, "F");
  if (!py_F) {
    PyErr_SetString(PyExc_AttributeError, "could not access field F of supposed AnnotatedDisjunction object!");
    goto cleanup;
  }
  py_cl_F = PyObject_GetAttrString(py_ad, "cl_F");
  if (!py_cl_F) {
    PyErr_SetString(PyExc_AttributeError, "could not access field cl_F of supposed AnnotatedDisjunction object!");
    goto cleanup;
  }

  py_P_L = PySequence_Fast(py_P, "field AnnotatedDisjunction.P must either be a list or tuple!");
  if (!py_P_L) goto cleanup;
  py_F_L = PySequence_Fast(py_F, "field AnnotatedDisjunction.F must either be a list or tuple!");
  if (!py_F_L) goto cleanup;
  py_cl_F_L = PySequence_Fast(py_cl_F, "field AnnotatedDisjunction.cl_F must either be a list or tuple!");
  if (!py_cl_F_L) goto cleanup;

  n = PySequence_Fast_GET_SIZE(py_P_L);
  P = (double*) malloc(n*sizeof(double));
  if (!P) goto nomem;
  F = (const char**) malloc(n*sizeof(const char*));
  if (!F) goto nomem;
  cl_F = (clingo_symbol_t*) malloc(n*sizeof(clingo_symbol_t));
  if (!cl_F) goto nomem;
  F_obj = (PyObject**) malloc(n*sizeof(PyObject*));
  if (!F_obj) goto nomem;

  for (i = 0; i < n; ++i) {
    PyObject *rep;
    P[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_P_L, i));
    if ((P[i] == -1) && !PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "field P of AnnotatedDisjunction must be a list of floats!");
      goto cleanup;
    }
    F_obj[i] = PySequence_Fast_GET_ITEM(py_F_L, i);
    F[i] = PyUnicode_AsUTF8(F_obj[i]);
    if (!F[i]) {
      PyErr_SetString(PyExc_TypeError, "field F of AnnotatedDisjunction must be a list of strings!");
      goto cleanup;
    }
    rep = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(py_cl_F_L, i), "_rep");
    if (!rep) {
      PyErr_SetString(PyExc_AttributeError, "field cl_F of AnnotatedDisjunction must be a list of Symbols!");
      goto cleanup;
    }
    cl_F[i] = PyLong_AsUnsignedLong(rep);
    if ((cl_F[i] == (clingo_symbol_t) -1) && !PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "field _rep of elements in AnnotatedDisjunction.cl_F must be a Symbol!");
      goto cleanup;
    }
    Py_DECREF(rep);
  }

  ad->P = P;
  ad->F = F;
  ad->F_obj = F_obj;
  ad->cl_F = cl_F;
  ad->n = n;

  Py_DECREF(py_P);
  Py_DECREF(py_F);
  Py_DECREF(py_cl_F);
  Py_DECREF(py_P_L);
  Py_DECREF(py_F_L);
  Py_DECREF(py_cl_F_L);

  return true;
nomem:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
cleanup:
  Py_XDECREF(py_P);
  Py_XDECREF(py_F);
  Py_XDECREF(py_cl_F);
  Py_XDECREF(py_P_L);
  Py_XDECREF(py_F_L);
  Py_XDECREF(py_cl_F_L);
  free(P); free(F);
  free(cl_F); free(F_obj);

  return false;
}

static bool from_python_program(PyObject *py_P, program_t *P) {
  PyObject *py_P_P, *py_P_PF, *py_P_PF_L, *py_P_PR, *py_P_PR_L, *py_P_Q, *py_P_Q_L, *py_P_CF, *py_P_AD, *py_P_CF_L, *py_P_sem = NULL;
  PyObject *py_P_AD_L = py_P_AD = py_P_CF_L = py_P_CF = py_P_Q_L = py_P_Q = py_P_PR_L = py_P_PR = py_P_PF_L = py_P_PF = py_P_P = NULL;
  PyObject *py_P_stable = NULL;
  const char *P_P;
  prob_fact_t *PF = NULL;
  prob_rule_t *PR = NULL;
  query_t *Q = NULL;
  credal_fact_t *CF = NULL;
  annot_disj_t *AD = NULL;
  program_t *stable = NULL;
  semantics_t sem;
  size_t i;

  py_P_P = PyObject_GetAttrString(py_P, "P");
  if (!py_P_P) {
    PyErr_SetString(PyExc_AttributeError, "could not access field P of supposed Program object!");
    goto cleanup;
  }
  P_P = PyUnicode_AsUTF8(py_P_P);
  if (!P_P) {
    PyErr_SetString(PyExc_TypeError, "field P of Program must be a string!");
    goto cleanup;
  }

  py_P_PF = PyObject_GetAttrString(py_P, "PF");
  if (!py_P_PF) {
    PyErr_SetString(PyExc_AttributeError, "could not access field PF of supposed Program object!");
    goto cleanup;
  }
  py_P_PR = PyObject_GetAttrString(py_P, "PR");
  if (!py_P_PR) {
    PyErr_SetString(PyExc_AttributeError, "could not access field PR of supposed Program object!");
    goto cleanup;
  }
  py_P_Q = PyObject_GetAttrString(py_P, "Q");
  if (!py_P_Q) {
    PyErr_SetString(PyExc_AttributeError, "could not access field Q of supposed Program object!");
    goto cleanup;
  }
  py_P_CF = PyObject_GetAttrString(py_P, "CF");
  if (!py_P_CF) {
    PyErr_SetString(PyExc_AttributeError, "could not access field CF of supposed Program object!");
    goto cleanup;
  }
  py_P_AD = PyObject_GetAttrString(py_P, "AD");
  if (!py_P_AD) {
    PyErr_SetString(PyExc_AttributeError, "could not access field AD of supposed Program object!");
    goto cleanup;
  }

  py_P_sem = PyObject_GetAttrString(py_P, "semantics");
  if (!py_P_sem) {
    PyErr_SetString(PyExc_AttributeError, "could not access field semantics of supposed Program object!");
    goto cleanup;
  }
  sem = PyLong_AsUnsignedLong(py_P_sem);
  if ((sem == (unsigned long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field semantics of Program must be a Semantics!");
    goto cleanup;
  }

  py_P_stable = PyObject_GetAttrString(py_P, "stable");
  if (py_P_stable != Py_None) {
    stable = (program_t*) malloc(sizeof(program_t));
    if (!from_python_program(py_P_stable, stable)) goto cleanup;
  } else if (!py_P_stable && sem) {
    PyErr_SetString(PyExc_TypeError, "field stable of Program must be a Program!");
    goto cleanup;
  }

  py_P_PF_L = PySequence_Fast(py_P_PF, "field Program.PF must either be a list or tuple!");
  if (!py_P_PF_L) goto cleanup;
  py_P_PR_L = PySequence_Fast(py_P_PR, "field Program.PR must either be a list or tuple!");
  if (!py_P_PR_L) goto cleanup;
  py_P_Q_L = PySequence_Fast(py_P_Q, "field Program.Q must either be a list or tuple!");
  if (!py_P_Q_L) goto cleanup;
  py_P_CF_L = PySequence_Fast(py_P_CF, "field Program.CF must either be a list or tuple!");
  if (!py_P_CF_L) goto cleanup;
  py_P_AD_L = PySequence_Fast(py_P_AD, "field Program.AD must either be a list or tuple!");
  if (!py_P_AD_L) goto cleanup;

  P->PF_n = PySequence_Fast_GET_SIZE(py_P_PF_L);
  P->PR_n = PySequence_Fast_GET_SIZE(py_P_PR_L);
  P->Q_n = PySequence_Fast_GET_SIZE(py_P_Q_L);
  P->CF_n = PySequence_Fast_GET_SIZE(py_P_CF_L);
  P->AD_n = PySequence_Fast_GET_SIZE(py_P_AD_L);

  PF = (prob_fact_t*) malloc(P->PF_n*sizeof(prob_fact_t));
  if (!PF) goto nomem;
  PR = (prob_rule_t*) malloc(P->PR_n*sizeof(prob_rule_t));
  if (!PR) goto nomem;
  Q = (query_t*) malloc(P->Q_n*sizeof(query_t));
  if (!Q) goto nomem;
  CF = (credal_fact_t*) malloc(P->CF_n*sizeof(credal_fact_t));
  if (!CF) goto nomem;
  AD = (annot_disj_t*) malloc(P->AD_n*sizeof(annot_disj_t));
  if (!AD) goto nomem;

  for (i = 0; i < P->PF_n; ++i)
    if (!from_python_prob_fact(PySequence_Fast_GET_ITEM(py_P_PF_L, i), &PF[i])) goto cleanup;
  for (i = 0; i < P->PR_n; ++i)
    if (!from_python_prob_rule(PySequence_Fast_GET_ITEM(py_P_PR_L, i), &PR[i])) goto cleanup;
  for (i = 0; i < P->Q_n; ++i)
    if (!from_python_query(PySequence_Fast_GET_ITEM(py_P_Q_L, i), &Q[i], sem)) goto cleanup;
  for (i = 0; i < P->CF_n; ++i)
    if (!from_python_credal_fact(PySequence_Fast_GET_ITEM(py_P_CF_L, i), &CF[i])) goto cleanup;
  for (i = 0; i < P->AD_n; ++i)
    if (!from_python_ad(PySequence_Fast_GET_ITEM(py_P_AD_L, i), &AD[i])) goto cleanup;

  P->P = P_P;
  P->P_obj = py_P_P;
  P->PF = PF;
  P->PR = PR;
  P->Q = Q;
  P->CF = CF;
  P->AD = AD;

  P->gr_PF.d = NULL; P->gr_PF.n = P->gr_PF.c = 0;
  P->gr_P.d = NULL; P->gr_P.n = P->gr_P.c = 0;
  P->gr_pr.d = NULL; P->gr_pr.n = P->gr_pr.c = 0;

  P->sem = sem;
  P->stable = stable;
  P->py_P = py_P;

  Py_DECREF(py_P_PF);
  Py_DECREF(py_P_PR);
  Py_DECREF(py_P_Q);
  Py_DECREF(py_P_CF);
  Py_DECREF(py_P_AD);
  Py_DECREF(py_P_PF_L);
  Py_DECREF(py_P_PR_L);
  Py_DECREF(py_P_Q_L);
  Py_DECREF(py_P_CF_L);
  Py_DECREF(py_P_AD_L);
  Py_DECREF(py_P_sem);
  Py_XDECREF(py_P_stable);

  return true;
nomem:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
cleanup:
  Py_XDECREF(py_P_P);
  Py_XDECREF(py_P_PF);
  Py_XDECREF(py_P_PR);
  Py_XDECREF(py_P_Q);
  Py_XDECREF(py_P_CF);
  Py_XDECREF(py_P_AD);
  Py_XDECREF(py_P_PF_L);
  Py_XDECREF(py_P_PR_L);
  Py_XDECREF(py_P_Q_L);
  Py_XDECREF(py_P_CF_L);
  Py_XDECREF(py_P_AD_L);
  Py_XDECREF(py_P_sem);
  Py_XDECREF(py_P_stable);
  free(PF);
  free(PR);
  free(Q);
  free(CF);
  free(AD);
  free(stable);
  return false;
}

static PyMethodDef CprogramMethods[] = {
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef cprogrammodule = {
  PyModuleDef_HEAD_INIT,
  "cprogram",
  "Program functions from the C side.",
  -1,
  CprogramMethods,
};

PyMODINIT_FUNC PyInit_cprogram(void) {
  PyObject *m;
  static void* PyCprogram_API[PyCprogram_API_pointers];
  PyObject *c_api_object;

  m = PyModule_Create(&cprogrammodule);
  if (!m) return NULL;
  if (import_cutils() < 0) return NULL;
  if (import_carray() < 0) return NULL;

  PyCprogram_API[PyCprogram_print_prob_fact_NUM] = (void*) print_prob_fact;
  PyCprogram_API[PyCprogram_free_prob_fact_contents_NUM] = (void*) free_prob_fact_contents;
  PyCprogram_API[PyCprogram_free_prob_fact_NUM] = (void*) free_prob_fact;
  PyCprogram_API[PyCprogram_print_credal_fact_NUM] = (void*) print_credal_fact;
  PyCprogram_API[PyCprogram_free_credal_fact_contents_NUM] = (void*) free_credal_fact_contents;
  PyCprogram_API[PyCprogram_free_credal_fact_NUM] = (void*) free_credal_fact;
  PyCprogram_API[PyCprogram_print_query_NUM] = (void*) print_query;
  PyCprogram_API[PyCprogram_free_query_contents_NUM] = (void*) free_query_contents;
  PyCprogram_API[PyCprogram_free_query_NUM] = (void*) free_query;
  PyCprogram_API[PyCprogram_print_program_NUM] = (void*) print_program;
  PyCprogram_API[PyCprogram_free_program_contents_NUM] = (void*) free_program_contents;
  PyCprogram_API[PyCprogram_free_program_NUM] = (void*) free_program;
  PyCprogram_API[PyCprogram_from_python_prob_fact_NUM] = (void*) from_python_prob_fact;
  PyCprogram_API[PyCprogram_from_python_credal_fact_NUM] = (void*) from_python_credal_fact;
  PyCprogram_API[PyCprogram_from_python_query_NUM] = (void*) from_python_query;
  PyCprogram_API[PyCprogram_from_python_program_NUM] = (void*) from_python_program;
  PyCprogram_API[PyCprogram_print_prob_rule_NUM] = (void*) print_prob_rule;
  PyCprogram_API[PyCprogram_free_prob_rule_contents_NUM] = (void*) free_prob_rule_contents;
  PyCprogram_API[PyCprogram_free_prob_rule_NUM] = (void*) free_prob_rule;

  c_api_object = PyCapsule_New((void*) PyCprogram_API, "cprogram._C_API", NULL);

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
    Py_XDECREF(c_api_object);
    Py_DECREF(m);
    return NULL;
  }

  setlocale(LC_CTYPE, "");

  return m;
}

#ifdef PASP_DEBUG
int main() { return 0; }
#endif
