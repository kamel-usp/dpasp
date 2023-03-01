#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdbool.h>

#include "cutils.h"
#include "cprogram.h"
#include "carray.h"
#include "../bitvector/bitvector.h"

#define CGROUND_MODULE
#include "ground.h"

static inline size_t unique_ground_pfact_id() { static size_t i = 0; return i++; }

#define GROUND_MAX_PROBRULE_LINE_LEN 400
#define GROUND_MAX_SYM_LEN 50

static bool unify_callback(const clingo_location_t *loc, const char *name, const clingo_symbol_t *args,
    size_t argc, void* data, clingo_symbol_callback_t sym_callback, void *sym_data) {
  int b, h, i, j;
  size_t s_n, cursor, _cursor = 0;
  char line[GROUND_MAX_PROBRULE_LINE_LEN], s[GROUND_MAX_SYM_LEN];
  char _line[GROUND_MAX_PROBRULE_LINE_LEN];
  void **pack = (void**) data;
  array_clingo_symbol_t_t *PF = (array_clingo_symbol_t_t*) pack[0];
  array_char_t *S = (array_char_t*) pack[1];
  array_double_t *Pr = (array_double_t*) pack[2];
  bitvec_t *L = (bitvec_t*) pack[3];
  uintptr_t sem = (uintptr_t) pack[4];
  clingo_symbol_t ground_pf;
  const char *cl_str;
  double pr;

  /* Get probability of probabilistic rule. */
  if (!clingo_symbol_string(args[0], &cl_str)) goto error;
  pr = atof(cl_str);

  /* Check if probabilistic rule is learnable. */
  int l;
  if (!clingo_symbol_number(args[2], &l)) goto error;
  bitvec_push(L, l);

  /* Get number of head arguments. */
  if (!clingo_symbol_number(args[3], &h)) goto error;
  /* Get number of body subgoals. */
  if (!clingo_symbol_number(args[4], &b)) goto error;

#define ARGS_START 5

  /* Get rule name. */
  if (!clingo_symbol_to_string_size(args[1], &s_n)) goto error;
  if (!clingo_symbol_to_string(args[1], s, s_n)) goto error;
  memcpy(line, s, s_n); line[s_n-1] = '('; cursor = s_n;
  if (sem) { memcpy(_line+1, s, s_n); _line[0] = '_'; _line[s_n] = '('; _cursor = s_n+1; }
  /* Fill out grounded head arguments. */
  for (i = 0, j = ARGS_START; i < h; ++i) {
    if (!clingo_symbol_to_string_size(args[i+j], &s_n)) goto error;
    if (!clingo_symbol_to_string(args[i+j], s, s_n)) goto error;
    if (i != h-1) { s[s_n-1] = ','; s[s_n++] = ' '; }
    memcpy(line+cursor, s, s_n); cursor += s_n;
    if (sem) { memcpy(_line+_cursor, s, s_n); _cursor += s_n; }
  }
  strcat(line, ") :- "); cursor += 4;
  if (sem) { strcat(_line,  ") :- "); _cursor += 4; }
  /* Fill out grounded body subgoals. */
  for (i = 0, j += h; i < b; i += 2) {
    int pos;
    if (!clingo_symbol_number(args[i+j], &pos)) goto error;
    if (!clingo_symbol_to_string_size(args[i+j+1], &s_n)) goto error;
    if (!clingo_symbol_to_string(args[i+j+1], s, s_n)) goto error;
    if (!pos) {
      memcpy(line+cursor, "not ", 4); cursor += 4;
      if (sem) { memcpy(_line+_cursor, "not ", 4); _cursor += 4; }
    }
    memcpy(line+cursor, s, s_n);
    cursor += s_n;
    line[cursor-1] = ','; line[cursor++] = ' ';
    if (sem) {
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
  if (sem) { memcpy(_line+_cursor, s, s_n); _cursor += s_n; }

  /* Add the newly created probabilistic fact to the set of new PFs. */
  if (!array_clingo_symbol_t_append(PF, ground_pf)) goto error;
  /* Add the grounded rule to the logic part. */
  if (!array_char_writeln(S, line, cursor+1)) goto error;
  if (sem) { if (!array_char_writeln(S, _line, _cursor+1)) goto error; }
  /* Add the probability of this grounded probabilistic rule. */
  if (!array_double_append(Pr, pr)) goto error;

  /* Pass the actual head arguments down to the original rule. */
  return sym_callback(args + ARGS_START, h, sym_data);
error:
  return false;
}

static bool partial_update_program(program_t *P, array_char_t *gr_P,
    array_clingo_symbol_t_t *gr_PF, array_double_t *gr_pr, bitvec_t *L) {
  PyObject *py_gr_P = NULL;
  PyObject *py_mdl = NULL, *py_pf_type = NULL;
  PyObject *py_P_PF = NULL;
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

  /* Add new probabilistic facts to P->PF and update Python side. */
  size_t n = P->PF_n + gr_PF->n;

  py_mdl = PyImport_ImportModule("pasp.program");
  if (!py_mdl) goto cleanup;
  py_pf_type = PyObject_GetAttrString(py_mdl, "ProbFact");
  if (!py_pf_type) goto cleanup;

  py_P_PF = PyObject_GetAttrString(P->py_P, "PF");
  if (!py_P_PF) goto cleanup;

  PF = (prob_fact_t*) realloc(P->PF, n*sizeof(prob_fact_t));
  if (!PF) goto cleanup;
  for (size_t i = 0; i < gr_PF->n; ++i) {
    prob_fact_t *pf = PF + P->PF_n + i;
    pf->p = gr_pr->d[i];
    pf->cl_f = gr_PF->d[i];
    pf->f_obj = NULL;
    pf->learnable = bitvec_GET(L, i);
    if (!clingo_symbol_name(pf->cl_f, &pf->f)) goto cleanup;
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

static bool ground(program_t *P) {
  size_t i;
  clingo_control_t *C = NULL;
  void *pack[5] = {NULL, NULL, NULL, NULL, (void*) ((uintptr_t) P->sem)};
  array_clingo_symbol_t_t gr_PF;
  array_char_t gr_P;
  array_double_t gr_pr;
  bitvec_t L = {0};
  bool ok = false;

  if (!array_char_init(&gr_P)) goto error;
  if (!array_clingo_symbol_t_init(&gr_PF)) goto error;
  if (!array_double_init(&gr_pr)) goto error;
  if (!bitvec_init(&L, 2)) goto error;

  pack[0] = (void*) &gr_PF; pack[1] = (void*) &gr_P; pack[2] = (void*) &gr_pr;
  pack[3] = (void*) &L;

  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto error;
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto error;
  for (i = 0; i < P->PR_n; ++i) {
    if (P->PR[i].is_prop) continue;
    if (!clingo_control_add(C, "base", NULL, 0, P->PR[i].unify)) goto error;
  }

  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, unify_callback, (void*) pack)) goto error;

  if (!partial_update_program(P, &gr_P, &gr_PF, &gr_pr, &L)) goto error;

  ok = true;
error:
  if (clingo_error_code() != clingo_error_success) {
    char err_buff[200];
    sprintf(err_buff, "Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
    PyErr_SetString(PyExc_Exception, err_buff);
  }
  array_clingo_symbol_t_free_contents(&gr_PF);
  array_double_free_contents(&gr_pr);
  array_char_free_contents(&gr_P);
  bitvec_free_contents(&L);
  if (C) clingo_control_free(C);
  return ok;
}

static bool needs_ground(program_t *P) {
  size_t i, n = P->PR_n;
  if (P->gr_P[0]) return false;
  for (i = 0; i < n; ++i) if (!P->PR[i].is_prop) return true;
  return false;
}

static PyObject* py_ground(PyObject *self, PyObject *args) {
  program_t p;
  PyObject *py_P = NULL;

  if (!PyArg_ParseTuple(args, "O", &py_P)) goto cleanup;
  if (!from_python_program(py_P, &p)) goto cleanup;

  if (needs_ground(&p)) if (!ground(&p)) goto cleanup;

cleanup:
  free_program_contents(&p);
  return py_P;
}

static PyMethodDef CgroundMethods[] = {
  {"ground", py_ground, METH_VARARGS, "Pre-grounds probabilistic rules."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef groundmodule = {
  PyModuleDef_HEAD_INIT,
  "ground",
  "Grounding functions from the C side.",
  -1,
  CgroundMethods,
};

PyMODINIT_FUNC PyInit_ground(void) {
  PyObject *m;
  static void* PyCground_API[PyCground_API_pointers];
  PyObject *c_api_object;

  m = PyModule_Create(&groundmodule);
  if (!m) return NULL;

  PyCground_API[PyCground_ground_NUM] = (void*) ground;
  PyCground_API[PyCground_needs_ground_NUM] = (void*) needs_ground;

  c_api_object = PyCapsule_New((void*) PyCground_API, "ground._C_API", NULL);

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
    Py_XDECREF(c_api_object);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}

