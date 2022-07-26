#ifndef Py_CUTILSMODULE_H
#define Py_CUTILSMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <clingo.h>

typedef struct string {
  char *s;
  size_t n;
} string_t;

#define PyCutils_string_from_symbol_NUM 0
#define PyCutils_string_from_symbol_RETURN bool
#define PyCutils_string_from_symbol_PROTO (clingo_symbol_t sym, string_t *buf)

#define PyCutils_string_free_NUM 1
#define PyCutils_string_free_RETURN void
#define PyCutils_string_free_PROTO (string_t *s)

#define PyCutils_API_pointers 2

#ifdef CUTILS_MODULE

static PyCutils_string_from_symbol_RETURN string_from_symbol PyCutils_string_from_symbol_PROTO;
static PyCutils_string_free_RETURN string_free PyCutils_string_free_PROTO;

#else

static void** PyCutils_API;

#define string_from_symbol \
  (*(PyCutils_string_from_symbol_RETURN (*)PyCutils_string_from_symbol_PROTO) PyCutils_API[PyCutils_string_from_symbol_NUM])
#define string_free \
  (*(PyCutils_string_free_RETURN (*)PyCutils_string_free_PROTO) PyCutils_API[PyCutils_string_free_NUM])

static int import_cutils(void) {
  PyCutils_API = (void**) PyCapsule_Import("cutils._C_API", 0);
  return (PyCutils_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
