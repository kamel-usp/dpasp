#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdarg.h>

#define CARRAY_MODULE
#include "carray.h"

#define ARRAY_MAGIC_MULTIPLIER 1.5
#define ARRAY_MAGIC_INIT_CAP   16

#define CARRAY_ARRAY_INIT_DECLARE(type) \
static bool array_##type##_init(array_##type##_t *a) { \
  a->d = (type*) malloc(ARRAY_MAGIC_INIT_CAP*sizeof(type)); \
  if (!a->d) { \
    a->c = 0; \
    PyErr_SetString(PyExc_MemoryError, "could not allocate memory for carray!"); \
    return false; \
  } \
  a->c = ARRAY_MAGIC_INIT_CAP; \
  a->n = 0; \
  return true; \
}

#define CARRAY_ARRAY_FREE_CONTENTS_DECLARE(type) \
static inline void array_##type##_free_contents(array_##type##_t *a) { \
  if (a->d) { free(a->d); a->d = NULL; a->c = 0; a->n = 0; } \
}

#define CARRAY_ARRAY_FREE_DECLARE(type) \
static inline void array_##type##_free(array_##type##_t *a) { array_##type##_free_contents(a); free(a); }

#define CARRAY_ARRAY_GROW_DECLARE(type) \
static inline bool array_##type##_grow(array_##type##_t *a) { \
  size_t nc = sizeof(type)*ARRAY_MAGIC_MULTIPLIER*a->n; \
  type *d = (type*) realloc(a->d, nc); \
  if (!d) { \
    a->d = NULL; \
    PyErr_SetString(PyExc_MemoryError, "could not allocate more memory for carray!"); \
    return false; \
  } \
  a->d = d; \
  a->c = nc; \
  return true; \
}

#define CARRAY_ARRAY_APPEND_DECLARE(type) \
static bool array_##type##_append(array_##type##_t *a, type o) { \
  if (a->n + 1 > a->c) { \
    if (!array_##type##_grow(a)) return false; \
    a->d[a->n++] = o; \
  } else a->d[a->n++] = o; \
  return true; \
}

#define CARRAY_ARRAY_EXTEND_DECLARE(type, promoted) \
static bool array_##type##_extend(array_##type##_t *a, size_t argc, ...) { \
  va_list args; \
  if ((a->n + argc > a->c)) if (!array_##type##_grow(a)) return false; \
  va_start(args, argc); \
  while (argc--) a->d[a->n++] = va_arg(args, promoted); \
  return true; \
}
#define CARRAY_ARRAY_EXTEND_NP_DECLARE(type) CARRAY_ARRAY_EXTEND_DECLARE(type, type)

CARRAY_ARRAY_INIT_DECLARE(bool);
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(bool);
CARRAY_ARRAY_FREE_DECLARE(bool);
CARRAY_ARRAY_GROW_DECLARE(bool);
CARRAY_ARRAY_APPEND_DECLARE(bool);
/* CARRAY_ARRAY_EXTEND_DECLARE(bool, int); */

CARRAY_ARRAY_INIT_DECLARE(double);
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(double);
CARRAY_ARRAY_FREE_DECLARE(double);
CARRAY_ARRAY_GROW_DECLARE(double);
CARRAY_ARRAY_APPEND_DECLARE(double);
/* CARRAY_ARRAY_EXTEND_NP_DECLARE(double); */

static PyMethodDef CarrayMethods[] = {
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef carraymodule = {
  PyModuleDef_HEAD_INIT,
  "carray",
  "Dynamic array implementation from the C side.",
  -1,
  CarrayMethods,
};

PyMODINIT_FUNC PyInit_carray(void) {
  PyObject *m;
  static void* PyCarray_API[PyCarray_API_pointers];
  PyObject *c_api_object;

  m = PyModule_Create(&carraymodule);
  if (!m) return NULL;

  PyCarray_API[PyCarray_array_double_init_NUM] = (void*) array_double_init;
  PyCarray_API[PyCarray_array_double_free_contents_NUM] = (void*) array_double_free_contents;
  PyCarray_API[PyCarray_array_double_free_NUM] = (void*) array_double_free;
  PyCarray_API[PyCarray_array_double_append_NUM] = (void*) array_double_append;
  PyCarray_API[PyCarray_array_bool_init_NUM] = (void*) array_bool_init;
  PyCarray_API[PyCarray_array_bool_free_contents_NUM] = (void*) array_bool_free_contents;
  PyCarray_API[PyCarray_array_bool_free_NUM] = (void*) array_bool_free;
  PyCarray_API[PyCarray_array_bool_append_NUM] = (void*) array_bool_append;

  c_api_object = PyCapsule_New((void*) PyCarray_API, "carray._C_API", NULL);

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
