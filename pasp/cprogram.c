#include "cprogram.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <locale.h>
#include <wchar.h>

#include "cutils.h"

/* Implement dynamic array of prob_fact_t's. */
ARRAY_IMPL(prob_fact_t)

void print_prob_fact(prob_fact_t *pf) {
  if (pf->learnable) wprintf(L"%f?::%s", pf->p, pf->f);
  else wprintf(L"%f::%s", pf->p, pf->f);
}
void free_prob_fact_contents(prob_fact_t *pf) { if (pf) Py_XDECREF(pf->f_obj); }
void free_prob_fact(prob_fact_t *pf) { free_prob_fact_contents(pf); free(pf); }

void print_prob_rule(prob_rule_t *pr) { wprintf(L"%f::%s", pr->p, pr->f); }
void free_prob_rule_contents(prob_rule_t *pr) {
  if (pr) { Py_DECREF(pr->f_obj); Py_XDECREF(pr->unify_obj); array_uint8_t_free_contents(&pr->PF); }
}
void free_prob_rule(prob_rule_t *pr) { free_prob_rule_contents(pr); free(pr); }

void print_credal_fact(credal_fact_t *cf) { wprintf(L"[%f, %f]::%s", cf->l, cf->u, cf->f); }
void free_credal_fact_contents(credal_fact_t *cf) { if (cf) Py_DECREF(cf->f_obj); }
void free_credal_fact(credal_fact_t *cf) { free_credal_fact_contents(cf); free(cf); }

void print_annot_disj(annot_disj_t *ad) {
  size_t i;
  for (i = 0; i < ad->n; ++i) {
    if (ad->learnable) wprintf(L"%f?::%s", ad->P[i], ad->F[i]);
    else wprintf(L"%f::%s", ad->P[i], ad->F[i]);
    if (i != ad->n-1) fputws(L"; ", stdout);
  }
}
void free_annot_disj_contents(annot_disj_t *ad) {
  if (!ad) return;
  free(ad->P);
  free(ad->F);
  free(ad->F_obj);
  free(ad->cl_F);
}
void free_annot_disj(annot_disj_t *ad) { free_annot_disj_contents(ad); free(ad); }

bool update_pr_neural_rule(neural_rule_t *nr) {
  PyArrayObject *py_P = (PyArrayObject*) PyObject_CallMethod(nr->self, "pr", NULL);
  if (!py_P) return false;
  nr->P = PyArray_DATA(py_P);
  return true;
}
bool update_forward_neural_rule(neural_rule_t *nr, size_t start, size_t end) {
  PyArrayObject *py_P = (PyArrayObject*) PyObject_CallMethod(nr->self, "forward", "kk",
      start, end);
  if (!py_P) return false;
  nr->P = PyArray_DATA(py_P);
  return true;
}
bool backward_neural_rule(neural_rule_t *nr) {
  return PyObject_CallMethod(nr->self, "backward", NULL);
}

void free_neural_rule_contents(neural_rule_t *nr) {}
void free_neural_rule(neural_rule_t *nr) { free_neural_rule_contents(nr); free(nr); }

bool update_pr_neural_annot_disj(neural_annot_disj_t *na) {
  PyArrayObject *py_P = (PyArrayObject*) PyObject_CallMethod(na->self, "pr", NULL);
  if (!py_P) return false;
  na->P = PyArray_DATA(py_P);
  return true;
}
bool update_forward_neural_annot_disj(neural_annot_disj_t *na, size_t start, size_t end) {
  PyArrayObject *py_P = (PyArrayObject*) PyObject_CallMethod(na->self, "forward", "kk",
      start, end);
  if (!py_P) return false;
  na->P = PyArray_DATA(py_P);
  return true;
}
bool backward_neural_annot_disj(neural_annot_disj_t *na) {
  return PyObject_CallMethod(na->self, "backward", NULL);
}

void free_neural_annot_disj_contents(neural_annot_disj_t *na) {}
void free_neural_annot_disj(neural_annot_disj_t *na) { free_neural_annot_disj_contents(na); free(na); }

bool print_query_with_buffer(query_t *q, string_t *s) {
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
bool print_query(query_t *Q) {
  string_t s = {NULL, 0};
  bool r = print_query_with_buffer(Q, &s);
  free(s.s);
  return r;
}

void free_query_contents(query_t *Q) {
  if (!Q) return;
  free(Q->Q);
  free(Q->Q_s);
  free(Q->Q_u);
  free(Q->E);
  free(Q->E_s);
  free(Q->E_u);
}
void free_query(query_t *Q) { free_query_contents(Q); free(Q); }

void free_varquery_contents(varquery_t *VQ) {
  if (!VQ) return;
  free(VQ->Q_s);
  free(VQ->E_s);
  Py_DECREF(VQ->py_gr_rule);
}

void free_varquery(varquery_t *VQ) { free_varquery_contents(VQ); free(VQ); }

void print_program(program_t *P) {
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
void free_program_contents(program_t *P) {
  size_t i;
  if (!P) return;
  Py_XDECREF(P->P_obj);
  for (i = 0; i < P->PF_n; ++i) free_prob_fact_contents(&P->PF[i]);
  free(P->PF);
  for (i = 0; i < P->PR_n; ++i) free_prob_rule_contents(&P->PR[i]);
  free(P->PR);
  for (i = 0; i < P->Q_n; ++i) free_query_contents(&P->Q[i]);
  free(P->Q);
  for (i = 0; i < P->VQ_n; ++i) free_varquery_contents(&P->VQ[i]);
  free(P->VQ);
  for (i = 0; i < P->CF_n; ++i) free_credal_fact_contents(&P->CF[i]);
  free(P->CF);
  for (i = 0; i < P->AD_n; ++i) free_annot_disj_contents(&P->AD[i]);
  free(P->AD);
  for (i = 0; i < P->NR_n; ++i) free_neural_rule_contents(&P->NR[i]);
  free(P->NR);
  for (i = 0; i < P->NA_n; ++i) free_neural_annot_disj_contents(&P->NA[i]);
  free(P->NA);
  Py_XDECREF(P->py_gr_P);
  if (P->stable) free_program(P->stable);
}
void free_program(program_t *P) { free_program_contents(P); free(P); }

bool from_python_prob_rule(PyObject *py_pr, prob_rule_t *pr) {
  PyObject *py_p, *py_f, *py_is_prop, *py_unify = py_is_prop = py_f = py_p = NULL;
  PyObject *py_pf_ids, *py_learnable, *py_sharing = py_learnable = py_pf_ids = NULL;
  double p;
  const char *f;
  long is_prop_l, learnable, sharing;
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

  py_learnable = PyObject_GetAttrString(py_pr, "learnable");
  if (!py_learnable) {
    PyErr_SetString(PyExc_AttributeError, "could not access field learnable of supposed ProbRule object!");
    goto cleanup;
  }
  learnable = PyLong_AsLong(py_learnable);
  if ((learnable == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field learnable of ProbRule must be a bool!");
    goto cleanup;
  }

  py_sharing = PyObject_GetAttrString(py_pr, "sharing");
  if (!py_sharing) {
    PyErr_SetString(PyExc_AttributeError, "could not access field sharing of supposed ProbRule object!");
    goto cleanup;
  }
  sharing = PyLong_AsLong(py_sharing);
  if ((sharing == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field sharing of ProbRule must be a bool!");
    goto cleanup;
  }

  py_pf_ids = PyObject_GetAttrString(py_pr, "pf_ids");
  if (!py_pf_ids) {
    PyErr_SetString(PyExc_AttributeError, "could not access field pf_ids of supposed ProbRule object!");
    goto cleanup;
  }
  if (PyList_Check(py_pf_ids)) {
    size_t pf_ids_n = PyList_GET_SIZE(py_pf_ids);
    if (!array_uint8_t_initn(&pr->PF, pf_ids_n)) goto cleanup;
    for (size_t i = 0; i < pf_ids_n; ++i) {
      PyObject *e = PyList_GET_ITEM(py_pf_ids, i);
      uint8_t u = PyLong_AsUnsignedLong(e);
      if ((u == (uint8_t) -1) && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "field pf_ids of ProbRule must be a list of integers!");
        goto cleanup;
      }
      pr->PF.d[i] = u;
    }
  } else { /* py_pf_ids should be Py_None, but we don't have to check for this case. */
    pr->PF.d = NULL; pr->PF.n = 0; pr->PF.c = 0;
  }

  pr->p = p;
  pr->f = f;
  pr->f_obj = py_f;
  pr->is_prop = (bool) is_prop_l;
  pr->unify = unify;
  pr->unify_obj = py_unify;
  pr->learnable = learnable;
  pr->sharing = sharing;
  pr->self = py_pr;
  r = true;

cleanup:
  Py_XDECREF(py_p);
  if (!r) { Py_XDECREF(py_f); Py_XDECREF(py_unify); }
  Py_XDECREF(py_is_prop);
  Py_XDECREF(py_pf_ids);
  Py_XDECREF(py_learnable);
  Py_XDECREF(py_sharing);
  return r;
}

bool from_python_prob_fact(PyObject *py_pf, prob_fact_t *pf) {
  PyObject *py_p, *py_f, *py_cl_f, *py_cl_f_rep, *py_learnable = py_cl_f_rep = py_cl_f = py_f = py_p = NULL;
  double p;
  const char *f;
  clingo_symbol_t cl_f;
  long learnable;
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
  cl_f = PyLong_AsUnsignedLongLong(py_cl_f_rep);
  if ((cl_f == (clingo_symbol_t) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field cl_f of ProbFact must be a Symbol!");
    goto cleanup;
  }

  py_learnable = PyObject_GetAttrString(py_pf, "learnable");
  if (!py_learnable) {
    PyErr_SetString(PyExc_AttributeError, "could not access field learnable of supposed ProbFact object!");
    goto cleanup;
  }
  learnable = PyLong_AsLong(py_learnable);
  if ((learnable == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field learnable of ProbFact must be a bool!");
    goto cleanup;
  }

  pf->p = p;
  pf->f = f;
  pf->f_obj = py_f;
  pf->cl_f = cl_f;
  pf->learnable = (bool) learnable;
  /* This might introduce a bug if py_pf expires during a C call (same with AD). Should keep this
   * in mind going forward. For now it won't... Probably. The solution would be to Py_INCREF(py_pf)
   * here and later Py_DECREF(py_pf) when freeing PF. */
  pf->self = py_pf;
  r = true;

cleanup:
  Py_XDECREF(py_p);
  if (!r) Py_XDECREF(py_f);
  Py_XDECREF(py_cl_f_rep);
  Py_XDECREF(py_cl_f);
  Py_XDECREF(py_learnable);
  return r;
}

bool from_python_credal_fact(PyObject *py_cf, credal_fact_t *cf) {
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
  cl_f = PyLong_AsUnsignedLongLong(py_cl_f_rep);
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

bool from_python_query(PyObject *py_q, query_t *q, semantics_t sem) {
  PyObject *py_Q, *py_E, *py_Q_L, *py_E_L = py_Q_L = py_E = py_Q = NULL;
  clingo_symbol_t *Q, *E = Q = NULL;
  clingo_symbol_t *Q_u, *E_u = Q_u = NULL;
  uint8_t *Q_s, *E_s = Q_s = NULL;
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
  Q_s = (uint8_t*) malloc(q->Q_n*sizeof(uint8_t));
  if (!Q_s) goto nomem;
  E_s = (uint8_t*) malloc(q->E_n*sizeof(uint8_t));
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
    Q[i] = PyLong_AsUnsignedLongLong(rep);
    Q_s[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(t, 1));
    if (sem) { /* sem != (STABLE_SEMANTICS = 0) */
      PyObject *u = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 2), "_rep");
      if (!u) { Py_DECREF(rep); Py_DECREF(t); goto cleanup; }
      Q_u[i] = PyLong_AsUnsignedLongLong(u);
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
    E[i] = PyLong_AsUnsignedLongLong(rep);
    E_s[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(t, 1));
    if (sem) { /* sem != (STABLE_SEMANTICS = 0) */
      PyObject *u = PyObject_GetAttrString(PySequence_Fast_GET_ITEM(t, 2), "_rep");
      if (!u) { Py_DECREF(rep); Py_DECREF(t); goto cleanup; }
      E_u[i] = PyLong_AsUnsignedLongLong(u);
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

bool from_python_varquery(PyObject *py_vq, varquery_t *vq) {
  PyObject *py_gr_rule, *py_Q, *py_E = py_Q = py_gr_rule = NULL;
  const char *gr_rule = NULL;
  uint8_t *Q_s, *E_s = Q_s = NULL;
  size_t Q_n, E_n;
  bool ok = false;

  py_gr_rule = PyObject_GetAttrString(py_vq, "gr_rule");
  if (!py_gr_rule) {
    PyErr_SetString(PyExc_AttributeError, "could not access field gr_rule of supposed VarQuery object!");
    goto cleanup;
  }
  gr_rule = PyUnicode_AsUTF8(py_gr_rule);
  if (!gr_rule) {
    PyErr_SetString(PyExc_TypeError, "field gr_rule of VarQuery must be a string!");
    goto cleanup;
  }
  py_Q = PyObject_GetAttrString(py_vq, "Q_s");
  if (!py_gr_rule) {
    PyErr_SetString(PyExc_AttributeError, "could not access field Q_s of supposed VarQuery object!");
    goto cleanup;
  }
  Q_n = PyList_GET_SIZE(py_Q);
  Q_s = (uint8_t*) malloc(Q_n*sizeof(uint8_t));
  if (!Q_s) goto nomem;
  for (size_t i = 0; i < Q_n; ++i) Q_s[i] = PyLong_AsLong(PyList_GET_ITEM(py_Q, i));
  py_E = PyObject_GetAttrString(py_vq, "E_s");
  if (!py_gr_rule) {
    PyErr_SetString(PyExc_AttributeError, "could not access field E_s of supposed VarQuery object!");
    goto cleanup;
  }
  E_n = PyList_GET_SIZE(py_E);
  E_s = (uint8_t*) malloc(E_n*sizeof(uint8_t));
  if (!E_s) goto nomem;
  for (size_t i = 0; i < E_n; ++i) E_s[i] = PyLong_AsLong(PyList_GET_ITEM(py_E, i));

  vq->gr_rule = gr_rule;
  vq->py_gr_rule = py_gr_rule;
  vq->Q_n = Q_n; vq->Q_s = Q_s;
  vq->E_n = E_n; vq->E_s = E_s;
  vq->self = py_vq;

  ok = true;
  goto cleanup;
nomem:
  PyErr_SetString(PyExc_MemoryError, "no free memory available!");
  free(Q_s); free(E_s);
cleanup:
  if (!ok) Py_XDECREF(py_gr_rule);
  Py_XDECREF(py_Q);
  Py_XDECREF(py_E);
  return ok;
}

bool from_python_ad(PyObject *py_ad, annot_disj_t *ad) {
  PyObject *py_P, *py_F, *py_cl_F = py_F = py_P = NULL;
  PyObject *py_P_L, *py_F_L, *py_cl_F_L = py_F_L = py_P_L = NULL;
  PyObject **F_obj = NULL, *py_learnable = NULL;
  double *P = NULL;
  const char **F = NULL;
  clingo_symbol_t *cl_F = NULL;
  long learnable;
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
  py_learnable = PyObject_GetAttrString(py_ad, "learnable");
  if (!py_learnable) {
    PyErr_SetString(PyExc_AttributeError, "could not access field learnable of supposed AnnotatedDisjunction object!");
    goto cleanup;
  }
  learnable = PyLong_AsLong(py_learnable);
  if ((learnable == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field learnable of AnnotatedDisjunction must be a bool!");
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
    cl_F[i] = PyLong_AsUnsignedLongLong(rep);
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
  ad->learnable = (bool) learnable;
  ad->self = py_ad;

  Py_XDECREF(py_P);
  Py_XDECREF(py_F);
  Py_XDECREF(py_cl_F);
  Py_XDECREF(py_P_L);
  Py_XDECREF(py_F_L);
  Py_XDECREF(py_cl_F_L);
  Py_XDECREF(py_learnable);

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

bool from_python_neural_rule(PyObject *py_nr, neural_rule_t *nr) {
  PyObject *py_learnable = NULL, *py_tensor_dw = NULL, *py_o = NULL;
  PyArrayObject *py_H, *py_B, *py_S, *py_dw = py_S = py_B = py_H = NULL;
  clingo_symbol_t *H = NULL, *B = NULL;
  float *dw;
  bool *S = NULL;
  long learnable;
  size_t n, k = 0, o;
  bool ok = false;

  py_learnable = PyObject_GetAttrString(py_nr, "learnable");
  if (!py_learnable) {
    PyErr_SetString(PyExc_AttributeError, "could not access field learnable of supposed NeuralRule object!");
    goto cleanup;
  }
  learnable = PyLong_AsLong(py_learnable);
  if ((learnable == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field learnable of NeuralRule must be a bool!");
    goto cleanup;
  }

  py_o = PyObject_GetAttrString(py_nr, "outcomes");
  if (!py_o) {
    PyErr_SetString(PyExc_AttributeError, "could not access field outcomes of supposed NeuralRule object!");
    goto cleanup;
  }
  o = PyLong_AsUnsignedLong(py_o);
  if ((o == (size_t) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field outcomes of NeuralRule must be an integer!");
    goto cleanup;
  }

  py_H = (PyArrayObject*) PyObject_GetAttrString(py_nr, "H");
  if (!py_H) {
    PyErr_SetString(PyExc_AttributeError, "could not access field H of supposed NeuralRule object!");
    goto cleanup;
  }
  H = (clingo_symbol_t*) PyArray_DATA(py_H);
  /* The number of groundings is the number of heads divided by the number of outcomes. */
  n = PyArray_SIZE(py_H)/o;

  PyObject *_py_B = PyObject_GetAttrString(py_nr, "B");
  if (!_py_B) {
    PyErr_SetString(PyExc_AttributeError, "could not access field B of supposed NeuralRule object!");
    goto cleanup;
  }
  if (_py_B != Py_None) {
    py_B = (PyArrayObject*) _py_B;
    k = PyArray_SIZE(py_B)/n;
    B = PyArray_DATA(py_B);

    py_S = (PyArrayObject*) PyObject_GetAttrString(py_nr, "S");
    if (!py_S) {
      PyErr_SetString(PyExc_AttributeError, "could not access field S of supposed NeuralRule object!");
      goto cleanup;
    }
    S = PyArray_DATA(py_S);
  }

  py_tensor_dw = PyObject_GetAttrString(py_nr, "dw");
  if (!py_tensor_dw) {
    PyErr_SetString(PyExc_AttributeError, "could not access field dw of supposed NeuralRule object!");
    goto cleanup;
  }
  if (learnable) {
    py_dw = (PyArrayObject*) PyObject_CallMethod(py_tensor_dw, "numpy", NULL);
    if (!py_dw) {
      PyErr_SetString(PyExc_AttributeError, "could not call method numpy in tensor NeuralRule.dw!");
      goto cleanup;
    }
    dw = (float*) PyArray_DATA(py_dw);
  } else dw = NULL;

  nr->n = n; nr->k = k;
  nr->P = NULL;
  nr->H = H; nr->B = B; nr->S = S;
  nr->dw = dw;
  nr->learnable = learnable;
  nr->self = py_nr;
  nr->o = o;

  ok = true;
cleanup:
  if (!ok) { free(H); free(B); free(S); }
  Py_XDECREF(py_H); Py_XDECREF(py_B); Py_XDECREF(py_S); Py_XDECREF(py_learnable);
  Py_XDECREF(py_tensor_dw); Py_XDECREF(py_dw); Py_XDECREF(py_o);
  return ok;
}

bool from_python_neural_ad(PyObject *py_nad, neural_annot_disj_t *nad) {
  PyObject *py_learnable = NULL, *py_o = NULL;
  PyArrayObject *py_H, *py_B, *py_S = py_B = py_H = NULL;
  clingo_symbol_t *H = NULL, *B = NULL;
  bool *S = NULL;
  long learnable;
  size_t n, v, k = 0, o;
  bool ok = false;

  py_learnable = PyObject_GetAttrString(py_nad, "learnable");
  if (!py_learnable) {
    PyErr_SetString(PyExc_AttributeError, "could not access field learnable of supposed NeuralRule object!");
    goto cleanup;
  }
  learnable = PyLong_AsLong(py_learnable);
  if ((learnable == (long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field learnable of NeuralRule must be a bool!");
    goto cleanup;
  }

  PyObject *py_V = PyObject_GetAttrString(py_nad, "vals");
  if (!py_V) {
    PyErr_SetString(PyExc_AttributeError, "could not access field vals of supposed NeuralRule object!");
    goto cleanup;
  }
  PyObject *py_V_L = PySequence_Fast(py_V, "field NeuralAD.vals must either be a list or tuple!");
  if (!py_V_L) { Py_DECREF(py_V); goto cleanup; }
  v = PySequence_Fast_GET_SIZE(py_V_L);
  Py_DECREF(py_V_L);

  py_o = PyObject_GetAttrString(py_nad, "outcomes");
  if (!py_o) {
    PyErr_SetString(PyExc_AttributeError, "could not access field outcomes of supposed NeuralRule object!");
    goto cleanup;
  }
  o = PyLong_AsUnsignedLong(py_o);
  if ((o == (size_t) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field outcomes of NeuralRule must be an integer!");
    goto cleanup;
  }

  py_H = (PyArrayObject*) PyObject_GetAttrString(py_nad, "H");
  if (!py_H) {
    PyErr_SetString(PyExc_AttributeError, "could not access field H of supposed NeuralRule object!");
    goto cleanup;
  }
  H = (clingo_symbol_t*) PyArray_DATA(py_H);
  /* The number of groundings is the number of heads divided by the outcomes and values. */
  n = PyArray_SIZE(py_H)/(v*o);

  PyObject *_py_B = PyObject_GetAttrString(py_nad, "B");
  if (!_py_B) {
    PyErr_SetString(PyExc_AttributeError, "could not access field B of supposed NeuralRule object!");
    goto cleanup;
  }
  if (_py_B != Py_None) {
    py_B = (PyArrayObject*) _py_B;
    k = PyArray_SIZE(py_B)/n;
    B = PyArray_DATA(py_B);

    py_S = (PyArrayObject*) PyObject_GetAttrString(py_nad, "S");
    if (!py_S) {
      PyErr_SetString(PyExc_AttributeError, "could not access field S of supposed NeuralRule object!");
      goto cleanup;
    }
    S = PyArray_DATA(py_S);
  }

  nad->n = n; nad->k = k; nad->v = v;
  nad->P = NULL;
  nad->H = H; nad->B = B; nad->S = S;
  nad->dw = NULL;
  nad->learnable = learnable;
  nad->self = py_nad;
  nad->o = o;

  ok = true;
cleanup:
  if (!ok) { free(H); free(B); free(S); }
  Py_XDECREF(py_H); Py_XDECREF(py_B); Py_XDECREF(py_S); Py_XDECREF(py_learnable);
  Py_XDECREF(py_o);
  return ok;
}

bool from_python_program(PyObject *py_P, program_t *P) {
  import_array();
  PyObject *py_P_P, *py_P_PF, *py_P_PF_L, *py_P_PR, *py_P_PR_L, *py_P_Q, *py_P_Q_L, *py_P_CF, *py_P_AD, *py_P_CF_L, *py_P_sem = NULL;
  PyObject *py_P_AD_L = py_P_AD = py_P_CF_L = py_P_CF = py_P_Q_L = py_P_Q = py_P_PR_L = py_P_PR = py_P_PF_L = py_P_PF = py_P_P = NULL;
  PyObject *py_P_NR, *py_P_NR_L, *py_P_NA, *py_P_NA_L = py_P_NA = py_P_NR_L = py_P_NR = NULL;
  PyObject *py_m = NULL, *py_P_stable = NULL, *py_gr_P = NULL;
  PyObject *py_P_VQ, *py_P_VQ_L = py_P_VQ = NULL;
  const char *P_P, *gr_P;
  prob_fact_t *PF = NULL;
  prob_rule_t *PR = NULL;
  query_t *Q = NULL;
  varquery_t *VQ = NULL;
  credal_fact_t *CF = NULL;
  annot_disj_t *AD = NULL;
  neural_rule_t *NR = NULL;
  neural_annot_disj_t *NA = NULL;
  program_t *stable = NULL;
  semantics_t sem;
  size_t i, m_train, m_test;
  bool is_ground;

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
  py_P_VQ = PyObject_GetAttrString(py_P, "VQ");
  if (!py_P_VQ) {
    PyErr_SetString(PyExc_AttributeError, "could not access field VQ of supposed Program object!");
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
  py_P_NR = PyObject_GetAttrString(py_P, "NR");
  if (!py_P_NR) {
    PyErr_SetString(PyExc_AttributeError, "could not access field NR of supposed Program object!");
    goto cleanup;
  }
  py_P_NA = PyObject_GetAttrString(py_P, "NA");
  if (!py_P_NA) {
    PyErr_SetString(PyExc_AttributeError, "could not access field NA of supposed Program object!");
    goto cleanup;
  }
  PyObject *py_is_ground = PyObject_GetAttrString(py_P, "is_ground");
  if (!py_is_ground) {
    PyErr_SetString(PyExc_AttributeError, "could not access field is_ground of supposed Program object!");
  }
  is_ground = PyLong_AsLong(py_is_ground);
  Py_DECREF(py_is_ground);

  py_m = PyObject_GetAttrString(py_P, "m_test");
  if (!py_m) {
    PyErr_SetString(PyExc_AttributeError, "could not access field m_test of supposed Program object!");
    goto cleanup;
  }
  m_test = PyLong_AsUnsignedLong(py_m);
  if ((m_test == (unsigned long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field m_test of Program must be an integer!");
    goto cleanup;
  }
  Py_DECREF(py_m);
  py_m = PyObject_GetAttrString(py_P, "m_train");
  if (!py_m) {
    PyErr_SetString(PyExc_AttributeError, "could not access field m_train of supposed Program object!");
    goto cleanup;
  }
  m_train = PyLong_AsUnsignedLong(py_m);
  if ((m_train == (unsigned long) -1) && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "field m_train of Program must be an integer!");
    goto cleanup;
  }

  py_P_sem = PyObject_GetAttrString(py_P, "semantics");
  if (!py_P_sem) {
    PyErr_SetString(PyExc_AttributeError, "could not access field semantics of supposed Program object!");
    goto cleanup;
  }
  sem = PyLong_AsUnsignedLong(py_P_sem);
  if ((sem == (semantics_t) ((unsigned long) -1)) && !PyErr_Occurred()) {
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

  py_gr_P = PyObject_GetAttrString(py_P, "gr_P");
  if (!py_gr_P) {
    PyErr_SetString(PyExc_AttributeError, "could not access field gr_P of supposed Program object!");
    goto cleanup;
  }
  gr_P = PyUnicode_AsUTF8(py_gr_P);
  if (!gr_P) {
    PyErr_SetString(PyExc_TypeError, "field gr_P of Program must be a string!");
    goto cleanup;
  }

  py_P_PF_L = PySequence_Fast(py_P_PF, "field Program.PF must either be a list or tuple!");
  if (!py_P_PF_L) goto cleanup;
  py_P_PR_L = PySequence_Fast(py_P_PR, "field Program.PR must either be a list or tuple!");
  if (!py_P_PR_L) goto cleanup;
  py_P_Q_L = PySequence_Fast(py_P_Q, "field Program.Q must either be a list or tuple!");
  if (!py_P_Q_L) goto cleanup;
  py_P_VQ_L = PySequence_Fast(py_P_VQ, "field Program.VQ must either be a list or tuple!");
  if (!py_P_VQ_L) goto cleanup;
  py_P_CF_L = PySequence_Fast(py_P_CF, "field Program.CF must either be a list or tuple!");
  if (!py_P_CF_L) goto cleanup;
  py_P_AD_L = PySequence_Fast(py_P_AD, "field Program.AD must either be a list or tuple!");
  if (!py_P_AD_L) goto cleanup;
  py_P_NR_L = PySequence_Fast(py_P_NR, "field Program.NR must either be a list or tuple!");
  if (!py_P_NR_L) goto cleanup;
  py_P_NA_L = PySequence_Fast(py_P_NA, "field Program.NA must either be a list or tuple!");
  if (!py_P_NA_L) goto cleanup;

  P->PF_n = PySequence_Fast_GET_SIZE(py_P_PF_L);
  P->PR_n = PySequence_Fast_GET_SIZE(py_P_PR_L);
  P->Q_n = PySequence_Fast_GET_SIZE(py_P_Q_L);
  P->VQ_n = PySequence_Fast_GET_SIZE(py_P_VQ_L);
  P->CF_n = PySequence_Fast_GET_SIZE(py_P_CF_L);
  P->AD_n = PySequence_Fast_GET_SIZE(py_P_AD_L);
  P->NR_n = PySequence_Fast_GET_SIZE(py_P_NR_L);
  P->NA_n = PySequence_Fast_GET_SIZE(py_P_NA_L);

  PF = (prob_fact_t*) malloc(P->PF_n*sizeof(prob_fact_t));
  if (!PF) goto nomem;
  PR = (prob_rule_t*) malloc(P->PR_n*sizeof(prob_rule_t));
  if (!PR) goto nomem;
  Q = (query_t*) malloc(P->Q_n*sizeof(query_t));
  if (!Q) goto nomem;
  VQ = (varquery_t*) malloc(P->VQ_n*sizeof(varquery_t));
  if (!VQ) goto nomem;
  CF = (credal_fact_t*) malloc(P->CF_n*sizeof(credal_fact_t));
  if (!CF) goto nomem;
  AD = (annot_disj_t*) malloc(P->AD_n*sizeof(annot_disj_t));
  if (!AD) goto nomem;
  NR = (neural_rule_t*) malloc(P->NR_n*sizeof(neural_rule_t));
  if (!NR) goto nomem;
  NA = (neural_annot_disj_t*) malloc(P->NA_n*sizeof(neural_annot_disj_t));
  if (!NA) goto nomem;

  for (i = 0; i < P->PF_n; ++i)
    if (!from_python_prob_fact(PySequence_Fast_GET_ITEM(py_P_PF_L, i), &PF[i])) goto cleanup;
  for (i = 0; i < P->PR_n; ++i)
    if (!from_python_prob_rule(PySequence_Fast_GET_ITEM(py_P_PR_L, i), &PR[i])) goto cleanup;
  for (i = 0; i < P->Q_n; ++i)
    if (!from_python_query(PySequence_Fast_GET_ITEM(py_P_Q_L, i), &Q[i], sem)) goto cleanup;
  for (i = 0; i < P->VQ_n; ++i)
    if (!from_python_varquery(PySequence_Fast_GET_ITEM(py_P_VQ_L, i), &VQ[i])) goto cleanup;
  for (i = 0; i < P->CF_n; ++i)
    if (!from_python_credal_fact(PySequence_Fast_GET_ITEM(py_P_CF_L, i), &CF[i])) goto cleanup;
  for (i = 0; i < P->AD_n; ++i)
    if (!from_python_ad(PySequence_Fast_GET_ITEM(py_P_AD_L, i), &AD[i])) goto cleanup;
  for (i = 0; i < P->NR_n; ++i)
    if (!from_python_neural_rule(PySequence_Fast_GET_ITEM(py_P_NR_L, i), &NR[i])) goto cleanup;
  for (i = 0; i < P->NA_n; ++i)
    if (!from_python_neural_ad(PySequence_Fast_GET_ITEM(py_P_NA_L, i), &NA[i])) goto cleanup;

  P->P = P_P;
  P->P_obj = py_P_P;
  P->PF = PF;
  P->PR = PR;
  P->Q = Q;
  P->VQ = VQ;
  P->CF = CF;
  P->AD = AD;
  P->NR = NR;
  P->NA = NA;

  P->m_test = m_test; P->m_train = m_train;

  P->gr_P = gr_P;
  P->py_gr_P = py_gr_P;
  P->is_ground = is_ground;

  P->sem = sem;
  P->stable = stable;
  P->py_P = py_P;

  Py_DECREF(py_P_PF);
  Py_DECREF(py_P_PR);
  Py_DECREF(py_P_Q);
  Py_DECREF(py_P_VQ);
  Py_DECREF(py_P_CF);
  Py_DECREF(py_P_AD);
  Py_DECREF(py_P_NR);
  Py_DECREF(py_P_NA);
  Py_DECREF(py_P_PF_L);
  Py_DECREF(py_P_PR_L);
  Py_DECREF(py_P_Q_L);
  Py_DECREF(py_P_CF_L);
  Py_DECREF(py_P_AD_L);
  Py_DECREF(py_P_NR_L);
  Py_DECREF(py_P_NA_L);
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
  Py_XDECREF(py_P_VQ);
  Py_XDECREF(py_P_CF);
  Py_XDECREF(py_P_AD);
  Py_XDECREF(py_P_NR);
  Py_XDECREF(py_P_NA);
  Py_XDECREF(py_P_PF_L);
  Py_XDECREF(py_P_PR_L);
  Py_XDECREF(py_P_Q_L);
  Py_XDECREF(py_P_CF_L);
  Py_XDECREF(py_P_AD_L);
  Py_XDECREF(py_P_NR_L);
  Py_XDECREF(py_P_NA_L);
  Py_XDECREF(py_P_sem);
  Py_XDECREF(py_P_stable);
  Py_XDECREF(py_gr_P);
  free(PF);
  free(PR);
  free(Q);
  free(VQ);
  free(CF);
  free(AD);
  free(NR);
  free(NA);
  free(stable);
  return false;
}
