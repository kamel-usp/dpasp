#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define CUTILS_MODULE
#include "cutils.h"

static bool string_from_symbol(clingo_symbol_t sym, string_t *buf) {
  bool r = true;
  char *s;
  size_t n;

  if (!clingo_symbol_to_string_size(sym, &n)) goto error;
  if (buf->n < n) {
    if (!(s = (char*) realloc(buf->s, n*sizeof(char)))) {
      clingo_set_error(clingo_error_bad_alloc, "Could not allocate memory for symbol!");
      goto error;
    }
    buf->s = s;
    buf->n = n;
  }
  if (!clingo_symbol_to_string(sym, buf->s, n)) goto error;
  goto out;
error:
  r = false;
out:
  return r;
}

static void string_free(string_t *s) {
  if (!s->s) return;
  free(s->s);
  s->s = NULL, s->n = 0;
}

static PyMethodDef CutilsMethods[] = {
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef cutilsmodule = {
  PyModuleDef_HEAD_INIT,
  "cutils",
  "Utility functions from the C side.",
  -1,
  CutilsMethods,
};

PyMODINIT_FUNC PyInit_cutils(void) {
  PyObject *m;
  static void* PyCutils_API[PyCutils_API_pointers];
  PyObject *c_api_object;

  m = PyModule_Create(&cutilsmodule);
  if (!m) return NULL;

  PyCutils_API[PyCutils_string_from_symbol_NUM] = (void*) string_from_symbol;
  PyCutils_API[PyCutils_string_free_NUM] = (void*) string_free;

  c_api_object = PyCapsule_New((void*) PyCutils_API, "cutils._C_API", NULL);

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
    Py_XDECREF(c_api_object);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}

#ifdef PASP_DEBUG
int main() { return 0; }
#endif
