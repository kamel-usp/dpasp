#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdbool.h>

#include "cutils.h"
#include "cprogram.h"
#include "carray.h"

#define CGROUND_MODULE
#include "cground.h"

static inline size_t unique_ground_pfact_id() { static size_t i = 0; return i++; }

const clingo_part_t GROUND_DEFAULT_PARTS[] = {{"base", NULL, 0}};

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
  uintptr_t sem = (uintptr_t) pack[3];
  clingo_symbol_t ground_pf;
  const char *cl_str;
  double pr;

  /* Get probability of probabilistic rule. */
  if (!clingo_symbol_string(args[0], &cl_str)) goto error;
  pr = atof(cl_str);

  /* Get number of head arguments. */
  if (!clingo_symbol_number(args[2], &h)) goto error;
  /* Get number of body subgoals. */
  if (!clingo_symbol_number(args[3], &b)) goto error;

  /* Get rule name. */
  if (!clingo_symbol_to_string_size(args[1], &s_n)) goto error;
  if (!clingo_symbol_to_string(args[1], s, s_n)) goto error;
  memcpy(line, s, s_n); line[s_n-1] = '('; cursor = s_n;
  if (sem) { memcpy(_line+1, s, s_n); _line[0] = '_'; _line[s_n] = '('; _cursor = s_n+1; }
  /* Fill out grounded head arguments. */
  for (i = 0, j = 4; i < h; ++i) {
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
  return sym_callback(args + 4, h, sym_data);
error:
  return false;
}

static bool update_program(program_t *P) {
  PyObject *py_gr_P, *py_gr_PF, *py_gr_pr = py_gr_PF = py_gr_P = NULL;
  size_t i;

  py_gr_P = PyUnicode_DecodeUTF8(P->gr_P.d, P->gr_P.n-1, NULL);
  if (!py_gr_P) {
    PyErr_SetString(PyExc_UnicodeDecodeError, "could not decode gr_P as a UTF-8 string!");
    goto error;
  }

  py_gr_PF = PyTuple_New(P->gr_PF.n);
  if (!py_gr_PF) {
    PyErr_SetString(PyExc_MemoryError, "could not create new tuple for gr_PF!");
    goto error;
  }
  for (i = 0; i < P->gr_PF.n; ++i) {
    PyObject *py_sym = PyLong_FromUnsignedLong(P->gr_PF.d[i]);
    if (!py_sym) {
      PyErr_SetString(PyExc_TypeError, "could not build PyObject from clingo_symbol_t!");
      goto error;
    }
    PyTuple_SET_ITEM(py_gr_PF, i, py_sym);
  }

  py_gr_pr = PyTuple_New(P->gr_pr.n);
  if (!py_gr_pr) {
    PyErr_SetString(PyExc_MemoryError, "could not create new tuple for gr_pr!");
    goto error;
  }
  for (i = 0; i < P->gr_pr.n; ++i) {
    PyObject *py_pr = PyFloat_FromDouble(P->gr_pr.d[i]);
    if (!py_pr) {
      PyErr_SetString(PyExc_TypeError, "could not build PyObject from double!");
      goto error;
    }
    PyTuple_SET_ITEM(py_gr_pr, i, py_pr);
  }

  if (PyObject_SetAttrString(P->py_P, "gr_P", py_gr_P)) {
    PyErr_SetString(PyExc_AttributeError, "could not attribute value to Program.gr_P!");
    goto error;
  }
  if (PyObject_SetAttrString(P->py_P, "gr_PF", py_gr_PF)) {
    PyErr_SetString(PyExc_AttributeError, "could not attribute value to Program.gr_PF!");
    goto error;
  }
  if (PyObject_SetAttrString(P->py_P, "gr_pr", py_gr_pr)) {
    PyErr_SetString(PyExc_AttributeError, "could not attribute value to Program.gr_pr!");
    goto error;
  }

  return true;
error:
  Py_XDECREF(py_gr_P);
  Py_XDECREF(py_gr_PF);
  Py_XDECREF(py_gr_pr);
  return false;
}

static bool ground(program_t *P) {
  size_t i;
  clingo_control_t *C = NULL;
  void *pack[4] = {NULL, NULL, NULL, (void*) ((uintptr_t) P->sem)};

  if (!P->gr_PF.d && !array_clingo_symbol_t_init(&P->gr_PF)) goto error;
  if (!P->gr_P.d  && !array_char_init(&P->gr_P)) goto error;
  if (!P->gr_pr.d && !array_double_init(&P->gr_pr)) goto error;

  pack[0] = (void*) &P->gr_PF; pack[1] = (void*) &P->gr_P; pack[2] = (void*) &P->gr_pr;

  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) goto error;
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) goto error;
  for (i = 0; i < P->PR_n; ++i) {
    if (P->PR[i].is_prop) continue;
    if (!clingo_control_add(C, "base", NULL, 0, P->PR[i].unify)) goto error;
  }

  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, unify_callback, (void*) pack)) goto error;

  clingo_control_free(C);

  if (!update_program(P)) goto error;

  return true;
error:
  if (clingo_error_code() != clingo_error_success) {
    wprintf(L"Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
    PyErr_SetString(PyExc_Exception, "Clingo or unknown error!");
  }
  if (C) clingo_control_free(C);
  return false;
}

static bool needs_ground(program_t *P) {
  size_t i, n = P->PR_n;
  if (P->gr_PF.d) return false;
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

static struct PyModuleDef cgroundmodule = {
  PyModuleDef_HEAD_INIT,
  "cground",
  "Grounding functions from the C side.",
  -1,
  CgroundMethods,
};

PyMODINIT_FUNC PyInit_cground(void) {
  PyObject *m;
  static void* PyCground_API[PyCground_API_pointers];
  PyObject *c_api_object;

  m = PyModule_Create(&cgroundmodule);
  if (!m) return NULL;
  if (import_cutils() < 0) return NULL;
  if (import_cprogram() < 0) return NULL;
  if (import_carray() < 0) return NULL;

  PyCground_API[PyCground_ground_NUM] = (void*) ground;
  PyCground_API[PyCground_needs_ground_NUM] = (void*) needs_ground;

  c_api_object = PyCapsule_New((void*) PyCground_API, "cground._C_API", NULL);

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
    Py_XDECREF(c_api_object);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}

#ifdef PASP_DEBUG
int main(void) { return 0; }
#endif
