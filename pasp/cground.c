#include <stdbool.h>
#include <pthread.h>

#include "cutils.h"
#include "cinf.h"
#include "cprogram.h"
#include "carray.h"
#include "../bitvector/bitvector.h"

#include "cground.h"

size_t unique_ground_pfact_id() {
  static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  static size_t i = 0;
  pthread_mutex_lock(&mu);
  ++i;
  pthread_mutex_unlock(&mu);
  return i;
}

#define GROUND_MAX_PROBRULE_LINE_LEN 400
#define GROUND_MAX_SYM_LEN 50

bool unify_callback(const clingo_location_t *loc, const char *name, const clingo_symbol_t *args,
    size_t argc, void* data, clingo_symbol_callback_t sym_callback, void *sym_data) {
  int b, h, i, j, id;
  size_t s_n, cursor, _cursor = 0;
  char line[GROUND_MAX_PROBRULE_LINE_LEN], s[GROUND_MAX_SYM_LEN];
  char _line[GROUND_MAX_PROBRULE_LINE_LEN];
  void **pack = (void**) data;
  array_prob_fact_t *PF = (array_prob_fact_t*) pack[0];
  array_char_t *S = (array_char_t*) pack[1];
  program_t *P = (program_t*) pack[2];
  clingo_symbol_t ground_pf;
  bool shared_grounding;
  double pr;
  array_uint8_t *pf_ids = NULL;

  /* Get the ID (i.e. index) of probabilistic rule. */
  if (!clingo_symbol_number(args[0], &id)) goto error;
  pr = P->PR[id].p;

  /* Check if probabilistic rule is learnable. */
  int l;
  if (!clingo_symbol_number(args[2], &l)) goto error;

  /* Get probabilistic rule ID. */
  int pr_id;
  if (!clingo_symbol_number(args[3], &pr_id)) goto error;
  if ((shared_grounding = pr_id >= 0) && l && pack[3]) pf_ids = &((array_uint8_t*) pack[3])[pr_id];

  /* Get number of head arguments. */
  if (!clingo_symbol_number(args[4], &h)) goto error;
  /* Get number of body subgoals. */
  if (!clingo_symbol_number(args[5], &b)) goto error;

#define ARGS_START 6

  /* Get rule name. */
  if (!clingo_symbol_to_string_size(args[1], &s_n)) goto error;
  if (!clingo_symbol_to_string(args[1], s, s_n)) goto error;
  memcpy(line, s, s_n); line[s_n-1] = '('; cursor = s_n;
  if (P->sem) { memcpy(_line+1, s, s_n); _line[0] = '_'; _line[s_n] = '('; _cursor = s_n+1; }
  /* Fill out grounded head arguments. */
  for (i = 0, j = ARGS_START; i < h; ++i) {
    if (!clingo_symbol_to_string_size(args[i+j], &s_n)) goto error;
    if (!clingo_symbol_to_string(args[i+j], s, s_n)) goto error;
    if (i != h-1) { s[s_n-1] = ','; s[s_n++] = ' '; }
    memcpy(line+cursor, s, s_n); cursor += s_n;
    if (P->sem) { memcpy(_line+_cursor, s, s_n); _cursor += s_n; }
  }
  strcat(line, ") :- "); cursor += 4;
  if (P->sem) { strcat(_line,  ") :- "); _cursor += 4; }
  /* Fill out grounded body subgoals. */
  for (i = 0, j += h; i < b; i += 2) {
    int pos;
    if (!clingo_symbol_number(args[i+j], &pos)) goto error;
    if (!clingo_symbol_to_string_size(args[i+j+1], &s_n)) goto error;
    if (!clingo_symbol_to_string(args[i+j+1], s, s_n)) goto error;
    if (!pos) {
      memcpy(line+cursor, "not ", 4); cursor += 4;
      if (P->sem) { memcpy(_line+_cursor, "not ", 4); _cursor += 4; }
    }
    memcpy(line+cursor, s, s_n);
    cursor += s_n;
    line[cursor-1] = ','; line[cursor++] = ' ';
    if (P->sem) {
      /* If subgoal is negative, then remove _. */
      memcpy(_line+_cursor+1, s+(!pos), s_n);
      /* Otherwise, add _. */
      if (pos) _line[_cursor] = '_';
      _cursor += s_n+pos;
      _line[_cursor-1] = ','; _line[_cursor++] = ' ';
    }
  }
  /* Add the probabilistic fact. */
  s_n = sprintf(s, "__unique_grid_%lu", unique_ground_pfact_id());
  if (!clingo_parse_term(s, NULL, NULL, 20, &ground_pf)) goto error;
  s[s_n++] = '.'; s[s_n++] = '\0';
  memcpy(line+cursor, s, s_n);
  cursor += s_n;
  if (P->sem) { memcpy(_line+_cursor, s, s_n); _cursor += s_n; }

  /* Add to the corresponding probabilistic rule PF ID. */
  if (pf_ids) if (!array_uint8_t_append(pf_ids, P->PF_n + P->CF_n + PF->n)) goto error;
  /* Add the newly created probabilistic fact to the set of new PFs. */
  prob_fact_t pf = {pr, NULL, NULL, ground_pf, l && !shared_grounding, NULL};
  if (!clingo_symbol_name(ground_pf, &pf.f)) goto error;
  if (!array_prob_fact_t_append(PF, pf)) goto error;
  /* Add the grounded rule to the logic part. */
  if (!array_char_writeln(S, line, cursor+1)) goto error;
  if (P->sem) { if (!array_char_writeln(S, _line, _cursor+1)) goto error; }

  /* Pass the actual head arguments down to the original rule. */
  return sym_callback(args + ARGS_START, h, sym_data);
error:
  clingo_set_error(clingo_error_runtime, "could not pre-ground probabilistic rules!");
  return false;
}

bool partial_update_program(program_t *P, array_char_t *gr_P, array_prob_fact_t_t *gr_PF) {
  PyObject *py_gr_P = NULL;
  PyObject *py_mdl = NULL, *py_pf_type = NULL;
  PyObject *py_P_PF = NULL;
  PyObject *py_pf_ids = NULL;
  prob_fact_t *PF = NULL;
  bool ok = false;

  py_gr_P = PyUnicode_DecodeUTF8(gr_P->d, gr_P->n-1, NULL);
  if (!py_gr_P) {
    PyErr_SetString(PyExc_UnicodeDecodeError, "could not decode gr_P as a UTF-8 string!");
    goto cleanup;
  }
  if (PyObject_SetAttrString(P->py_P, "gr_P", py_gr_P)) {
    PyErr_SetString(PyExc_AttributeError, "could not attribute value to Program.gr_P!");
    goto cleanup;
  }

  for (size_t i = 0; i < P->PR_n; ++i) {
    if (!P->PR[i].learnable) continue;
    array_uint8_t *pr_pfs = &P->PR[i].PF;
    if (pr_pfs->c) {
      py_pf_ids = PyList_New(pr_pfs->n);
      for (size_t j = 0; j < pr_pfs->n; ++j)
        PyList_SET_ITEM(py_pf_ids, j, PyLong_FromUnsignedLong(pr_pfs->d[j]));
      if (PyObject_SetAttrString(P->py_P, "pf_ids", py_pf_ids)) {
        PyErr_SetString(PyExc_AttributeError, "could not attribute value to Program.pf_ids!");
        goto cleanup;
      }
    }
  }

  /* Add new probabilistic facts to P->PF and update Python side. */
  size_t n = P->PF_n + gr_PF->n;

  py_mdl = PyImport_ImportModule("pasp.program");
  if (!py_mdl) goto cleanup;
  py_pf_type = PyObject_GetAttrString(py_mdl, "ProbFact");
  if (!py_pf_type) goto cleanup;

  py_P_PF = PyObject_GetAttrString(P->py_P, "PF");
  if (!py_P_PF) goto cleanup;

  PF = (prob_fact_t*) realloc(P->PF, n*sizeof(prob_fact_t));
  if (!PF) {
    PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for grounding!");
    goto cleanup;
  }
  for (size_t i = 0; i < gr_PF->n; ++i) {
    PF[P->PF_n+i] = gr_PF->d[i];
    prob_fact_t *pf = PF + P->PF_n + i;
    PyObject *args = Py_BuildValue("ds", pf->p, pf->f);
    pf->self = PyObject_Call(py_pf_type, args, NULL);
    if (!pf->self) goto cleanup;
    /* Update Python object with new PF. */
    if (PyList_Append(py_P_PF, pf->self)) goto cleanup;
  }
  P->PF = PF;
  P->PF_n = n;
  P->gr_P = PyUnicode_AsUTF8(py_gr_P);
  P->py_gr_P = py_gr_P;

  ok = true;
cleanup:
  Py_XDECREF(py_mdl);
  Py_XDECREF(py_pf_type);
  Py_XDECREF(py_P_PF);
  return ok;
}

bool ground_all(program_t *P, prob_storage_t *Q) {
  size_t i;
  clingo_control_t *C = NULL;
  array_prob_fact_t gr_PF = {0};
  array_char_t gr_P = {0};
  void *pack[4] = {(void*) &gr_PF, (void*) &gr_P, (void*) P, NULL};
  bool ok = false;

  if (P->gr_P[0]) return true;

  if (!array_prob_fact_t_init(&gr_PF)) goto error;
  if (!array_char_init(&gr_P)) goto error;

  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto error;

  if (!add_all_atoms_as_choice(C, P)) goto error;
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto error;
  for (i = 0; i < P->PR_n; ++i)
    if (!P->PR[i].is_prop) if (!clingo_control_add(C, "base", NULL, 0, P->PR[i].unify)) goto error;

  if (Q) {
    pack[3] = (void*) Q->I_GR;
    for (i = 0; i < Q->pr; ++i) array_uint8_t_clear(&Q->I_GR[i]);
  }

  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, unify_callback, (void*) pack)) goto error;

  if (!partial_update_program(P, &gr_P, &gr_PF)) goto error;

  if (!gr_P.n) {
    array_prob_fact_t_free_contents(&gr_PF);
    array_char_free_contents(&gr_P);
    ok = true;
    goto error;
  }

  ok = true;
error:
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  if (C) clingo_control_free(C);
  return ok;
}

/* Do not use. */
bool ground_per_total_choice(program_t *P, total_choice_t *theta, array_prob_fact_t *PF,
    total_choice_t *gamma, char **rules, prob_storage_t *Q) {
  size_t i;
  clingo_control_t *C = NULL;
  array_prob_fact_t gr_PF = {0};
  array_char_t gr_P = {0};
  void *pack[5] = {(void*) &gr_PF, (void*) &gr_P, (void*) P, (void*) true, NULL};
  bool ok = false;

  if (!array_prob_fact_t_init(&gr_PF)) goto error;
  if (!array_char_init(&gr_P)) goto error;

  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto error;
  if (!add_atoms_from_total_choice(C, P, theta)) goto error;
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto error;
  for (i = 0; i < P->PR_n; ++i) {
    if (!P->PR[i].sharing) continue;
    if (!clingo_control_add(C, "base", NULL, 0, P->PR[i].unify)) goto error;
  }

  if (Q) {
    pack[4] = (void*) Q->I_GR;
    for (i = 0; i < Q->pr; ++i) array_uint8_t_clear(&Q->I_GR[i]);
  }

  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, unify_callback, (void*) pack)) goto error;

  /* Nothing to ground. */
  if (!gr_P.n) {
    array_prob_fact_t_free_contents(&gr_PF);
    array_char_free_contents(&gr_P);
    ok = true;
    goto error;
  }
  /* Update PFs to probabilities of their respective PRs. The grounded unify string may not be
   * entirely updated after learning. */
  /*if (Q)*/
    /*for (i = 0; i < Q->pr; ++i) {*/
      /*array_uint8_t *pf_ids = &Q->I_GR[i];*/
      /*for (size_t l = 0; l < pf_ids->n; ++l) gr_PF.d[pf_ids->d[l]].p = P->PR[Q->I_PR[i]].p;*/
    /*}*/

  PF->d = gr_PF.d; PF->c = gr_PF.c; PF->n = gr_PF.n;
  if (!bitvec_init(&gamma->pf, gr_PF.n)) goto error;
  bitvec_zeron(&gamma->pf, gr_PF.n);
  gamma->ad_n = 0; gamma->theta_ad = NULL;
  if (gr_P.n) *rules = gr_P.d;

  ok = true;
error:
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  if (C) clingo_control_free(C);
  return ok;
}

bool ground(program_t *P) {
  clingo_control_t *C = NULL;
  array_prob_fact_t gr_PF = {0};
  array_char_t gr_P = {0};
  void *pack[4] = {(void*) &gr_PF, (void*) &gr_P, (void*) P, NULL};
  bool ok = false;

  if (P->gr_P[0]) return true;

  if (!array_prob_fact_t_init(&gr_PF)) goto error;
  if (!array_char_init(&gr_P)) goto error;

  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto error;
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto error;
  for (size_t i = 0; i < P->PR_n; ++i) {
    if (P->PR[i].is_prop || P->PR[i].sharing) continue;
    if (!clingo_control_add(C, "base", NULL, 0, P->PR[i].unify)) goto error;
  }
  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, unify_callback, (void*) pack)) goto error;
  /* Nothing to ground. */
  if (!gr_P.n) {
    array_prob_fact_t_free_contents(&gr_PF);
    ok = true;
    goto error;
  }
  if (!partial_update_program(P, &gr_P, &gr_PF)) goto error;

  ok = true;
error:
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  array_char_free_contents(&gr_P);
  if (C) clingo_control_free(C);
  return ok;
}

bool needs_ground(program_t *P) {
  for (size_t i = 0; i < P->PR_n; ++i) if (!P->PR[i].is_prop) return true;
  return false;
}
