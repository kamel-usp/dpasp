#include "cground.h"

static PyObject* py_ground(PyObject *self, PyObject *args) {
  program_t p;
  PyObject *py_P = NULL;

  if (!PyArg_ParseTuple(args, "O", &py_P)) goto cleanup;
  if (!from_python_program(py_P, &p)) goto cleanup;

  if (needs_ground(&p)) if (!ground_all(&p, NULL)) goto cleanup;

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

  m = PyModule_Create(&groundmodule);
  if (!m) return NULL;

  return m;
}

